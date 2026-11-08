# Shared model layer

The existing `generate_test_cases.py`, `generate_completions.py` and
`evaluate_completions.py` entry points remain in use.

| Module | Role |
| --- | --- |
| `registry.py` | HF model IDs, default revision pins and aliases |
| `loaders.py` | Shared pretrained model/processor loading |
| `targets/` | Model chat templates, image handling and response decoding |
| `inference.py` | Existing JSON inputs to shared targets; output compatibility |
| `surrogates.py`, `ce_surrogate.py` | Model profiles and differentiable teacher-forced CE interfaces |

The target adapters are ported from our `core/targets/` implementation. Their
common contract is `generate(text, image_paths) -> str`. Image paths are
absolute or repository-relative; `data/...` honors `VLM_JAILBREAK_DATA_ROOT`.
The JSON bridge first resolves generated image filenames against the test-case
file's sibling `images/` directory, the test-case directory, then repository/data-root paths.

```python
from multimodalmodels.targets import load_target

target = load_target("qwen3_vl_2b", max_new_tokens=256)
response = target.generate("Describe the image.", ["/path/to/image.png"])
```

The CLI accepts existing `["image.png", "prompt"]` pairs, multi-image
`[["one.png", "two.png"], "prompt"]` pairs, plain text, and `{text, images}`
objects. A behavior can also map to a list of these cases for multiple attempts.
Results remain `{BehaviorID: [{test_case, generation}, ...]}`.
LLaVA-1.5/1.6/OneVision accept at most one image through these adapters;
Qwen2/2.5 require one. Qwen3 accepts multiple images.

The shared loaders supply registry revisions by default; an explicit loader
revision overrides the default. Local target imports are lazy and do not load
weights.

```python
from multimodalmodels.surrogates import list_surrogate_profiles, load_surrogate

names = list_surrogate_profiles()
surrogate = load_surrogate("qwen3_vl_2b")
pixels = surrogate.to_unit01(image).detach().requires_grad_(True)
loss = surrogate.compute_loss_batch(pixels, ["red"], [image], prompts=["Name the color."])
loss.backward()
```

Surrogate inputs are RGB pixels in `[0, 1]`. Model-specific fast preprocessing
remains differentiable; `compute_loss_batch` returns a scalar target-token CE
loss. Attack update loops remain in the attack implementations. The profile
factory is a model integration API, separate from target inference selection.

Use `python generate_completions.py --help` for the completion CLI options.
Method-specific launchers are documented in [scripts/README.md](../scripts/README.md).
