import gc
import io
import os
import sys

import cv2
import numpy as np
import PIL
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from matplotlib import pyplot as plt
from PIL import Image
from torchvision.utils import save_image

from vit_vqgan_demo.vitvqgan import ViTVQ


def show_ims(recon, original):
    fig = plt.figure(figsize=(10, 7))

    Image1 = np.array(original)
    Image2 = np.array(recon)
    fig.add_subplot(1, 2, 1)

    plt.imshow(Image1)
    plt.axis("off")
    plt.title("Original")
    fig.add_subplot(1, 2, 2)

    plt.imshow(Image2)
    plt.axis("off")
    plt.title("Reconstructed")


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img):
    s = min(img.size)

    if s < 256:
        raise ValueError(f"min dim for image {s} < 256")

    r = 256 / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [256])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img


to_Pil = T.ToPILImage()
gc.collect()
torch.cuda.empty_cache()

image_path = "https://www.biography.com/.image/ar_1:1%2Cc_fill%2Ccs_srgb%2Cfl_progressive%2Cq_auto:good%2Cw_1200/MTc5OTk2ODUyMTMxNzM0ODcy/gettyimages-1229892983-square.jpg"  # @param {type:"string"}
if "https" in image_path:
    original = download_image(image_path)
else:
    if os.path.exists(image_path):
        original = Image.open(image_path)
    else:
        print("Please check the image path")

encoder = {"dim": 768, "depth": 12, "heads": 12, "mlp_dim": 3072}
decoder = {"dim": 768, "depth": 12, "heads": 12, "mlp_dim": 3072}
quantizer = {"embed_dim": 32, "n_embed": 8192}

image = preprocess(original).cuda()

model = ViTVQ(256, 8, encoder, decoder, quantizer, path="./imagenet_vitvq_base.ckpt").cuda()
recon, _ = model(image)

save_image(image, "original.png")
save_image(recon, "reconstructed.png")

recon = Image.open("reconstructed.png")
image = Image.open("original.png")

show_ims(recon, original)
print("original saved at vit_vqgan/original.png ")
print("reconstructed saved at vit_vqgan/reconstructed.png ")
