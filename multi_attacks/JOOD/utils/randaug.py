"""
JOOD
Copyright (c) 2025-present NAVER Corp.
Apache License v2.0
"""
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw

class RandAug:
    def __init__(self, N, r):
        self.N = N
        self.r = r
        self.augmentations = [
            self.gaussian_blur,
            self.rotation,
            self.lower_brightness,
            self.upper_brightness,
            self.lower_contrast,
            self.upper_contrast,
            self.upper_colorjitter,
            self.lower_colorjitter,
            self.solarize,
            self.posterize,
            self.cutout,
            self.random_crop,
            self.shearX,
            self.shearY
        ]

    def apply(self, image, seed=None):
        if type(image) == str:
            image = Image.open(image).convert("RGB")
        if seed is not None:
            # for deterministic sample
            random.seed(seed)
        augmentations_to_apply = random.sample(self.augmentations, self.N)
        for aug in augmentations_to_apply:
            image = aug(image)
        return image

    def gaussian_blur(self, image):
        radius = self.r * 10  # Maximum blur radius
        return image.filter(ImageFilter.GaussianBlur(radius))

    def rotation(self, image):
        angle = self.r * 180  # Maximum rotation angle
        return image.rotate(angle, expand=True)

    def lower_brightness(self, image):
        enhancer = ImageEnhance.Brightness(image)
        factor = 1 - (self.r * 0.9)  # r=0 -> factor=1, r=1 -> factor=0.1 (slightly visible) 
        return enhancer.enhance(factor)

    def upper_brightness(self, image):
        enhancer = ImageEnhance.Brightness(image)
        factor = 1 + (self.r * 2)  # r=0 -> factor=1, r=1 -> factor=3 (much brighter)
        return enhancer.enhance(factor)

    def lower_contrast(self, image):
        enhancer = ImageEnhance.Contrast(image)
        factor = 1 - (self.r * 0.9)  # r=0 -> factor=1, r=1 -> factor=0.1 (slightly perceivable)
        return enhancer.enhance(factor)

    def upper_contrast(self, image):
        enhancer = ImageEnhance.Contrast(image)
        factor = 1 + (self.r * 4)  # r=0 -> factor=1, r=1 -> factor=5 (much higher contrast)
        return enhancer.enhance(factor)

    def upper_colorjitter(self, image):
        enhancer = ImageEnhance.Color(image)
        factor = 1 + (self.r * 49)  # r=0 -> factor=1, r=1 -> factor=50 (extreme color jittering)
        return enhancer.enhance(factor)

    def lower_colorjitter(self, image):
        enhancer = ImageEnhance.Color(image)
        factor = 1 - (self.r * 0.9)  # r=0 -> factor=1, r=1 -> factor=0.1 (extreme desaturation)
        return enhancer.enhance(factor)

    def solarize(self, image):
        image = image.convert("RGB")
        if self.r == 0:
            return image  # No solarization for r=0
        threshold = 255 - int(self.r * 255)  # r=0 -> no solarization, r=1 -> threshold=0 (maximized solarized)
        return ImageOps.solarize(image, threshold)

    def posterize(self, image):
        image = image.convert("RGB")
        if self.r == 0:
            return image
        bits = int(1 + (3 * (1 - self.r)))  # r=0 -> bits=4, r=1 -> bits=1
        return ImageOps.posterize(image.convert("RGB"), bits)

    def cutout(self, image):
        if self.r == 0:
            return image
        
        draw = ImageDraw.Draw(image)
        w, h = image.size
        max_cutout_size = int((3/4) * min(w, h))
        cutout_size = int(self.r * max_cutout_size)
        
        if cutout_size == 0:
            return image
        
        x0 = random.randint(0, w - cutout_size)
        y0 = random.randint(0, h - cutout_size)
        draw.rectangle([x0, y0, x0 + cutout_size, y0 + cutout_size], fill=(0, 0, 0))
        return image

    def random_crop(self, image):
        # TODO: smart crop from foreground objects?
        if self.r == 0:
            return image
        
        w, h = image.size
        crop_factor = 1 - (3/4) * self.r
        crop_size = int(crop_factor * min(w, h))
        
        if crop_size == 0:
            return image  # No cropping if crop_size is 0
        
        left = random.randint(0, w - crop_size)
        top = random.randint(0, h - crop_size)
        return image.crop((left, top, left + crop_size, top + crop_size)).resize((w, h))

    def shearX(self, image):
        shear_factor = self.r  # Shear factor range 0 to 1
        return image.transform(image.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0))

    def shearY(self, image):
        shear_factor = self.r  # Shear factor range 0 to 1
        return image.transform(image.size, Image.AFFINE, (1, 0, 0, shear_factor, 1, 0))