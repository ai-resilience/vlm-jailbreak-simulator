"""Image-only JailGuard mutators without the text mutators' NLTK dependency.

These functions preserve the defaults and ``img_aug_dict`` mapping from the
official JailGuard ``utils/augmentations.py``.  Keeping image mutations in a
small module lets the defense registry load before optional NLP assets exist.
"""

import numpy as np
import torchvision.transforms as T
from PIL import ImageDraw


def sample_float_level(max_level, min_level=0.1):
    return np.random.uniform(low=min_level, high=max_level)


def sample_int_level(max_level, min_level=1):
    return np.random.randint(low=min_level, high=max_level)


def sample_odd_level(max_level, min_level=1):
    return 2 * np.random.randint(low=min_level, high=max_level) - 1


def mask_image(img, level=None, position="rand", mask_type="r", mask_size=(200, 200)):
    del level
    image_size = img.size
    new_mask_size = [
        min(mask_size[index], 0.3 * image_size[index])
        for index in range(len(mask_size))
    ]
    if position == "rand":
        mask_position = (
            np.random.randint(0, int(image_size[0] - new_mask_size[0])),
            np.random.randint(0, int(image_size[1] - new_mask_size[1])),
        )
    else:
        mask_position = (
            min(int(position.split("-")[0]), image_size[0]),
            min(int(position.split("-")[1]), image_size[1]),
        )
    if mask_type != "r":
        raise ValueError(f"Unsupported JailGuard mask type: {mask_type}")
    draw = ImageDraw.Draw(img)
    x, y = mask_position
    width, height = new_mask_size
    draw.rectangle((x, y, x + width, y + height), fill="black")
    return img


def blur_image(img, level=5):
    kernel = (sample_odd_level(level), sample_odd_level(level))
    return T.GaussianBlur(kernel_size=kernel)(img)


def flip_image(img, level=1.0):
    return T.RandomHorizontalFlip(p=sample_float_level(level))(img)


def vflip_image(img, level=1.0):
    return T.RandomVerticalFlip(p=sample_float_level(level))(img)


def resize_crop_image(img, level=500):
    width, height = img.size
    target_width = int(max(sample_int_level(level), 0.8 * width))
    target_height = int(max(sample_int_level(level), 0.8 * height))
    return T.RandomResizedCrop(
        (target_height, target_width), scale=(0.9, 1)
    )(img)


def gray_image(img, level=1):
    if sample_float_level(level) >= 0.5:
        return T.Grayscale(num_output_channels=len(img.split()))(img)
    return img


def rotation_image(img, level=180):
    return T.RandomRotation(degrees=(0, sample_float_level(level)))(img)


def colorjitter_image(img, level1=1, level2=0.5):
    brightness = sample_float_level(level1)
    hue = sample_float_level(level2)
    return T.ColorJitter(brightness=brightness, hue=hue)(img)


def solarize_image(img, level=200):
    return T.RandomSolarize(threshold=sample_float_level(level))(img)


def posterize_image(img, level=3):
    return T.RandomPosterize(bits=sample_int_level(level))(img)


def policy_aug_image(img):
    # Official PL policy: RR / BL / RP with probabilities .34 / .45 / .21.
    sample = np.random.random()
    if sample < 0.34:
        return rotation_image(img)
    if sample < 0.79:
        return blur_image(img)
    return posterize_image(img)


IMAGE_MUTATORS = {
    "RM": mask_image,
    "BL": blur_image,
    "HF": flip_image,
    "CR": resize_crop_image,
    "GR": gray_image,
    "RR": rotation_image,
    "CJ": colorjitter_image,
    "VF": vflip_image,
    "RS": solarize_image,
    "RP": posterize_image,
    "PL": policy_aug_image,
}


def get_image_mutator(method_name):
    try:
        return IMAGE_MUTATORS[method_name]
    except KeyError as exc:
        choices = ", ".join(sorted(IMAGE_MUTATORS))
        raise ValueError(
            f"Unknown JailGuard image mutator {method_name!r}; choose one of {choices}."
        ) from exc
