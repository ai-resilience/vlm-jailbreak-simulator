"""Reuse the loaded LLaVA-1.5 encoder without loading a second VLM."""
import abc
import types

import numpy as np
import torch
from PIL import Image


class Encoder:
    model_path = None

    def __init__(self, mdpth) -> types.NoneType:
        self.model_path = mdpth

    @staticmethod
    def compute_cosine(a_vec: np.ndarray, b_vec: np.ndarray):
        """calculate cosine similarity"""
        norms1 = np.linalg.norm(a_vec, axis=1)
        norms2 = np.linalg.norm(b_vec, axis=1)
        dot_products = np.sum(a_vec * b_vec, axis=1)
        cos_similarities = dot_products / (norms1 * norms2)
        return cos_similarities[0]

    @abc.abstractmethod
    def calc_cossim(self, pairs: list[tuple[str, str]]):
        res = []
        for p in pairs:
            text_embed = self.embed_text(p[0])
            img_embed = self.embed_img(p[1])
            cossim = self.compute_cosine(text_embed, img_embed)
            res.append(cossim)
        return res

    @abc.abstractmethod
    def embed_img(self, imgpth) -> np.ndarray:
        pass

    @abc.abstractmethod
    def embed_text(self, text) -> np.ndarray:
        pass


class ReusedLlavaEncoder(Encoder):
    def __init__(self, llava_model):
        backend = getattr(llava_model, "_model", llava_model)
        if not hasattr(backend, "model") or not hasattr(backend, "processor"):
            raise TypeError(
                "CIDER currently requires the registered LLaVA-1.5 target; "
                "the target wrapper does not expose its model and processor."
            )
        self.m = backend.model
        self.processor = backend.processor
        self.device = self.m.device

    def embed_img(self, imgpth) -> np.ndarray:
        with Image.open(imgpth) as opened:
            image = opened.convert("RGB")
        pixel_value = self.processor.image_processor(
            image, return_tensors="pt"
        ).pixel_values.to(self.device, dtype=self.m.dtype)
        with torch.no_grad():
            image_outputs = self.m.vision_tower(
                pixel_value, output_hidden_states=True
            )
            selected = image_outputs.hidden_states[
                self.m.config.vision_feature_layer
            ]
            selected = selected[:, 1:]
            feats = self.m.multi_modal_projector(selected)
            feats = torch.mean(feats, dim=1).to("cpu").float().numpy()
        return feats

    def embed_text(self, text) -> np.ndarray:
        input_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(
            self.device
        )
        with torch.no_grad():
            embeds = self.m.get_input_embeddings()(input_ids)
            embeds = torch.mean(embeds, dim=1).to("cpu").float().numpy()
        return embeds

    def calc_cossim(self, pairs):
        res = []
        for text, imgpth in pairs:
            t = self.embed_text(text)
            im = self.embed_img(imgpth)
            res.append(self.compute_cosine(t, im))
        return res
