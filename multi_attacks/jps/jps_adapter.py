"""JPS-specific Qwen2-VL tokenization and differentiable image packing.

These methods are ported from the reference Qwen2Adapter. This adapter is
separate from the common CE profiles and does not import UltraBreak.
"""
import base64
import io
import torch
from PIL import Image

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class Qwen2Adapter:
    def __init__(self, model_id, revision=None, image_size=(448, 448)):
        from multimodalmodels.loaders import load_qwen2vl
        self.model_id, self.revision = model_id, revision
        self.image_size = image_size
        self.model, self.processor = load_qwen2vl(
            model_id, revision=revision, processor_kwargs={"use_fast": False})
        self.model.eval().requires_grad_(False)
        self.device = self.model.device

    def process_jps_target(self, image_path, prompt_text, target_text):
        """Tokenize JPS's exact user(image+prompt) -> assistant(target) chat.

        Unlike UltraBreak's line-wise target appending, official JPS locates the
        assistant-role boundary in Qwen's full chat template and supervises only
        assistant target tokens before ``<|im_end|>``.
        """
        from qwen_vl_utils import process_vision_info

        with Image.open(image_path) as img:
            resized = img.convert("RGB").resize(self.image_size)
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{encoded}"},
                {"type": "text", "text": prompt_text},
            ],
        }
        user_messages = [user]
        full_messages = [user, {
            "role": "assistant",
            "content": [{"type": "text", "text": target_text}],
        }]
        prefix_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(full_messages)
        prefix = self.processor(
            text=[prefix_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)
        full = self.processor(
            text=[full_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        prefix_len = prefix["input_ids"].size(1)
        eos_id = self.processor.tokenizer.eos_token_id
        eos_positions = torch.nonzero(
            full["input_ids"][0, prefix_len:] == eos_id, as_tuple=False
        )
        if eos_positions.numel() == 0:
            raise RuntimeError("JPS target chat has no assistant <|im_end|> token")
        target_stop = prefix_len + int(eos_positions[0].item())
        input_ids = full["input_ids"][:, :target_stop]
        attention_mask = full["attention_mask"][:, :target_stop]
        labels = torch.full_like(input_ids, -100)
        labels[:, prefix_len:target_stop] = input_ids[:, prefix_len:target_stop]
        return (
            input_ids,
            labels,
            full["pixel_values"],
            full["image_grid_thw"],
            attention_mask,
        )

    def preprocess_patched(self, patched_tensor, image_grid_thw, temporal_patch_size=2, patch_size=14, merge_size=2):
        grid_t, grid_h, grid_w = image_grid_thw
        patches = patched_tensor.unsqueeze(0)
        channel = patches.shape[1]
        T = patches.size(0)
        remainder = T % temporal_patch_size
        if remainder != 0:
            pad = temporal_patch_size - remainder
            patches = torch.cat([patches, patches[-1:].expand(pad, *patches.shape[1:])], dim=0)
        patches = patches.reshape(grid_t, temporal_patch_size, channel,
                                  grid_h // merge_size, merge_size, patch_size,
                                  grid_w // merge_size, merge_size, patch_size)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return patches.reshape(grid_t * grid_h * grid_w,
                               channel * temporal_patch_size * patch_size * patch_size)

    @torch.no_grad()
    def generate(self, prompt, image_path, max_new_tokens=512):
        from qwen_vl_utils import process_vision_info

        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
            {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        # Some checkpoints persist sampling-only values in generation_config.
        # Explicit greedy defaults silence Transformers' invalid-flag warning
        # without changing deterministic decoding.
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]
