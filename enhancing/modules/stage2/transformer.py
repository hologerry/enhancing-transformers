# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import Optional, Tuple, Dict, Union, Any
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .layers import *
from ...utils.general import initialize_from_config


class CondTransformer(pl.LightningModule):
    def __init__(self, cond_key: str, cond: OmegaConf, stage1: OmegaConf, transformer: OmegaConf, path: Optional[str] = None, ignore_keys: List[str] = list()) -> None:
        super().__init__()
        
        # get condition key
        self.cond_key = cond_key

        # load condition model
        self.cond_model = initialize_from_config(cond)
        
        # load stage1 model
        self.stage1_model = initialize_from_config(stage1)
        
        # load transformer
        self.transformer = initialize_from_config(transformer)
        if stage1.params.qparams.use_residual:
            assert 'RQTransformer' in transformer.target

        # make the parameters in stage1 model not trainable
        self.stage1_model.eval()
        for p in self.stage1_model.parameters():
            p.requires_grad = False

        # make the parameters in condition model not trainable
        self.cond_model.eval()
        for p in self.cond_model.parameters():
            p.requires_grad = False

        if path is not None:
            self.init_from_ckpt(path, ignore_keys)

    def forward(self,
                codes: torch.LongTensor,
                conds: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        
        conds = conds.view(conds.shape[0], -1)
        logits = self.transformer(codes, conds)

        codes = codes.view(-1, codes.shape[-1])
            
        return logits, codes

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def sample(self,
               conds: torch.LongTensor,
               top_k: Optional[float] = None,
               top_p: Optional[float] = None,
               softmax_temperature: float = 1.0,
               use_fp16: bool = True,
               return_pixels: bool = False) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.LongTensor]]:

        conds = conds.view(conds.shape[0], -1)
        logits, codes = self.transformer.sample(conds=conds, top_k=top_k, top_p=top_p,
                                                softmax_temperature=softmax_temperature,
                                                use_fp16=use_fp16)

        if return_pixels:
            return self.stage1_model.decode_codes(codes).clamp(0, 1)
        else:
            codes = codes.view(-1, codes.shape[-1])
            
            return logits, codes

    def get_input(self, batch: Tuple[Any, Any], key: str) -> torch.FloatTensor:
        x = batch[key]

        if len(x.shape) == 3:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()

        return x.contiguous()

    def shared_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        images = self.get_input(batch, self.stage1_model.image_key)
        conds = self.get_input(batch, self.cond_key)

        codes = self.stage1_model.encode_codes(images).detach()
        conds = self.cond_model.encode_codes(conds).detach()
        
        logits, codes = self(codes, conds) # if torch.bernoulli(self.sample_prob) else self.sampling(conds) # scheduled sampling
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), codes.view(-1))

        return loss

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        loss = self.shared_step(batch, batch_idx)
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        loss = self.shared_step(batch, batch_idx)
        self.log("val/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb_cond')
        no_decay.add('pos_emb_code')

        if hasattr(self.transformer, 'pos_emb_depth'):
            no_decay.add('pos_emb_depth')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay 
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay/ignored set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        
        return optimizer
    
    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        
        conds = self.get_input(batch, self.cond_key).to(self.device)
        cond_codes = self.cond_model.encode_codes(conds).detach()
        
        log["conditions"] = self.cond_model.to_img(conds)
        log["first samples"] = self.sample(cond_codes, return_pixels=True)
        log["second samples"] = self.sample(cond_codes, return_pixels=True)
        
        return log
