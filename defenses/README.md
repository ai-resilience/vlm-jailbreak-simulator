# Defenses

Defense implementations live beside the attack packages and are selected in
the existing completion CLI. `load_defense()` resolves YAML through a lazy
registry. A defense implements input preflight, optional asset preparation,
provenance, and `apply_input(text, images, target)`. `validate_target(model_config)`
can reject unsupported targets before loading weights.

The result contains `blocked`, optional transformed `text`/`images`, an optional
final `response` or blocked `default_response`, and detector `meta`. A defense
can query the loaded target internally. The runner returns a blocked response
without querying the original input; an allowed input with no supplied final
response receives one downstream target query. It retains the original
`test_case` so defended and undefended records can be compared by the evaluator.

## Package layout

Keep shared code at the top level and each defense's implementation, helpers
and provenance together in its own package:

```text
defenses/
├── __init__.py           # Lazy registry and config loading
├── base.py               # Common defense interface
├── inference.py          # Shared target routing and call accounting
├── README.md
├── jailguard/
│   ├── __init__.py       # Public JailGuardDefense export
│   ├── defense.py
│   ├── image_mutations.py
│   ├── similarity.py
│   ├── utils.py
│   ├── README.md
│   └── UPSTREAM.md
├── cider/               # Diffusion and encoder-based detector
│   ├── defense.py
│   ├── encoder.py
│   ├── detection.py
│   ├── denoising.py
│   ├── denoiser.py
│   ├── guided_diffusion/
│   ├── setup.py
│   └── ...
├── adashield/           # Static and adaptive shield prompts
│   ├── __init__.py
│   ├── defense.py
│   ├── setup.py
│   ├── README.md
│   └── UPSTREAM.md
├── asd/                 # Protective image overlay and safety prompt
│   └── ...
└── vlmshield/           # CLIP features and released MLP input detector
    ├── __init__.py
    ├── defense.py
    ├── features.py
    ├── setup.py
    ├── README.md
    ├── UPSTREAM.md
    └── LICENSE
```

YAML settings remain in `configs/defense_configs/<Method>.yaml` beside the
other project configurations. Model weights and datasets remain outside the
implementation package.

To add another method, create `defenses/<method>/`, implement `BaseDefense`,
export its class in the package initializer, register it in `DEFENSE_REGISTRY`,
and add its YAML configuration. Helpers used by only one method belong in that
method's package. Add shared modules when multiple implementations need them.

| Defense | Supported target | Input | Setup and usage |
| --- | --- | --- | --- |
| JailGuard | Registered single-image targets | Exactly one image | [JailGuard](jailguard/README.md) |
| CIDER | LLaVA-1.5-7B | Exactly one image | [CIDER](cider/README.md) |
| AdaShield-S | Registered targets; placement configurable | Target's image limits | [AdaShield](adashield/README.md) |
| AdaShield-A | Registered targets; placement configurable | Exactly one image | [AdaShield](adashield/README.md) |
| AsD | Registered single-image targets; released patch transfers across targets | Exactly one image; patch geometry must fit | [AsD](asd/README.md) |
| VLMShield | Registered targets; independent CLIP detector | Zero or one image, within target limits | [VLMShield](vlmshield/README.md) |

AdaShield and AsD are input modifiers, so they supply no detector decision. The
evaluator reports ASR/utility comparisons and leaves detector metrics unavailable.
Each loaded defense also receives its YAML `config_path` for resolving
configuration-relative resources.
