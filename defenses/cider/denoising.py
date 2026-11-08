"""Reference CIDER checkpoint image generation (inference only)."""
import os
from math import ceil

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .denoiser import DiffusionRobustModel


def generate_denoised_img_diffusion(
    imgpth: list[str],
    save_dir: str,
    cps: int,
    step=50,
    device: int = 0,
    batch_size=50,
    denoiser=None,
    model_path=None,
    **kwargs,
):

    resized_imgs = []
    denoised_pth = []
    imgpth.sort()

    trans = transforms.Compose([transforms.Resize([224, 224])])
    for filepath in imgpth:
        img = Image.open(filepath).convert("RGB")
        filename = os.path.split(filepath)[1]
        img = trans(img)
        resized_imgs.append((img, filename))
        savename = os.path.splitext(filename)[0] + "_denoised_000times.jpg"
        img.save(os.path.join(save_dir, savename))
        denoised_pth.append(os.path.join(save_dir, savename))
    if cps <= 1:
        return denoised_pth

    device_obj = f"cuda:{device}" if isinstance(device, int) else str(device)
    owns_model = denoiser is None
    model = denoiser or DiffusionRobustModel(
        device_obj=device_obj, model_path=model_path, load_classifier=False
    )
    iterations = range(step, cps * step, step)
    b_num = ceil(len(resized_imgs) / batch_size)
    for b in tqdm(range(b_num), desc="denoising batch"):
        l = b * batch_size
        r = (b + 1) * batch_size if b < b_num - 1 else len(resized_imgs)
        part = resized_imgs[l:r]
        for it in iterations:
            ary = [np.array(_[0], dtype=np.float32) / 255 * 2 - 1 for _ in part]
            ary = np.array(ary)
            ary = torch.tensor(ary).permute(0, 3, 1, 2).to(device_obj)
            denoised_ary = np.array(model.denoise(ary, it).to("cpu"))
            denoised_ary = denoised_ary.transpose(0, 2, 3, 1)
            denoised_ary = (denoised_ary + 1) / 2 * 255
            denoised_ary = denoised_ary.astype(np.uint8)
            for i in range(denoised_ary.shape[0]):
                img = Image.fromarray(denoised_ary[i])
                sn = os.path.splitext(part[i][1])[
                    0
                ] + "_denoised_{:0>3d}times.jpg".format(it)
                img.save(os.path.join(save_dir, sn))
                denoised_pth.append(os.path.join(save_dir, sn))
    if owns_model:
        del model
        torch.cuda.empty_cache()
    return denoised_pth


def get_denoiser(device=0, model_path=None):
    device_obj = f"cuda:{device}" if isinstance(device, int) else str(device)
    return DiffusionRobustModel(device_obj=device_obj, model_path=model_path, load_classifier=False)
