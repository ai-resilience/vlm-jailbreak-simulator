import torch
import torch.nn as nn
import os
import warnings

os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
from .guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


class Args:
    image_size = 256
    num_channels = 256
    num_res_blocks = 2
    num_heads = 4
    num_heads_upsample = -1
    num_head_channels = 64
    attention_resolutions = "32,16,8"
    channel_mult = ""
    dropout = 0.0
    class_cond = False
    use_checkpoint = False
    use_scale_shift_norm = True
    resblock_updown = True
    use_fp16 = False
    use_new_attention_order = False
    clip_denoised = True
    num_samples = 10000
    batch_size = 16
    use_ddim = False
    model_path = ""
    classifier_path = ""
    classifier_scale = 1.0
    learn_sigma = True
    diffusion_steps = 1000
    noise_schedule = "linear"
    timestep_respacing = None
    use_kl = False
    predict_xstart = False
    rescale_timesteps = False
    rescale_learned_sigmas = False


class DiffusionRobustModel(nn.Module):
    def __init__(
        self,
        classifier_name="beit",
        device_obj="cuda",
        model_path=None,
        load_classifier=False,
    ):
        super().__init__()

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        mdpath = model_path or f"{os.path.dirname(__file__)}/256x256_diffusion_uncond.pt"
        if not os.path.isfile(mdpath):
            raise FileNotFoundError(
                f"CIDER diffusion checkpoint not found: {mdpath}. "
                "Run python -m defenses.cider.setup"
            )
        # weights loaded through multimodalmodels.loaders (all model loading lives there)
        from multimodalmodels.loaders import load_torch_checkpoint
        model.load_state_dict(
            load_torch_checkpoint(mdpath, map_location=device_obj, weights_only=False)
        )
        warnings.warn(f"\033[7m Using trained model from {mdpath} \033[0m", UserWarning)
        model.eval().to(device_obj)

        self.model = model
        self.diffusion = diffusion
        self.device = device_obj

        self.classifier_name = classifier_name
        self.classifier = None
        if load_classifier:
            self._load_classifier()

    def _load_classifier(self):
        """Load the upstream auxiliary ViT only when ``forward`` needs it.

        CIDER detection calls ``denoise`` directly, so eagerly loading this
        classifier wastes memory and does not affect detector outputs.
        """
        if self.classifier is not None:
            return self.classifier
        local_model_path = f"{os.path.dirname(__file__)}/../vit-patch16-224"
        from multimodalmodels.loaders import load_image_classifier
        try:
            classifier = load_image_classifier(local_model_path)
            warnings.warn(
                f"\033[7m Using local ViT model from {local_model_path} \033[0m",
                UserWarning,
            )
        except (OSError, ValueError, RuntimeError) as e:
            # If local model loading fails, fall back to Hugging Face model
            warnings.warn(
                f"\033[7m Failed to load local ViT model from {local_model_path}: {e}\n"
                f"Falling back to Hugging Face model: google/vit-base-patch16-224 \033[0m",
                UserWarning,
            )
            classifier = load_image_classifier("google/vit-base-patch16-224")
        classifier.eval().to(self.device)
        self.classifier = classifier
        return classifier

    def forward(self, x, t):
        x_in = x * 2 - 1
        imgs = self.denoise(x_in, t)

        imgs = torch.nn.functional.interpolate(
            imgs, (224, 224), mode="bicubic", antialias=True
        )

        imgs = torch.as_tensor(imgs).to(self.device)
        classifier = self._load_classifier()
        with torch.no_grad():
            out = classifier(imgs)

        return out

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).to(self.device)

        noise = torch.randn_like(x_start)

        #:param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
                    t_batch = torch.tensor([i] * len(x_start)).to(self.device)
                    out = self.diffusion.p_sample(
                        self.model, out, t_batch, clip_denoised=True
                    )["sample"]
            else:
                out = self.diffusion.p_sample(
                    self.model, x_t_start, t_batch, clip_denoised=True
                )["pred_xstart"]

        return out
