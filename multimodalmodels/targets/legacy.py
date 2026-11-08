"""Keep the existing optional wrappers behind the common target interface."""

from .base import TargetModel


class LegacyTarget(TargetModel):
    min_images = 1
    max_images = 1

    def __init__(self, wrapper_name, max_new_tokens=256):
        import multimodalmodels

        self.name = wrapper_name
        self._model = getattr(multimodalmodels, wrapper_name)()
        self.max_new_tokens = max_new_tokens

    def generate(self, text, image_paths):
        if len(image_paths) != 1:
            raise ValueError(f"{self.name} requires exactly one image")
        outputs = self._model.generate(
            [[image_paths[0], text]], image_dir="",
            max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
        )
        if len(outputs) != 1:
            raise RuntimeError(f"{self.name} returned an unexpected response count")
        return outputs[0]
