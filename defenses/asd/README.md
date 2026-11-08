# AsD — Attack as Defense

AsD overlays the released universal protective image onto the input image,
then prepends the original safety prompt. The default profile uses seed 42,
target patch area fraction 0.25 and alpha 0.7. It accepts exactly one image.

```bash
python generate_completions.py --model_name llava_v1_5 \
  --test_cases_path test_cases/FigStep_SafeBench/FigStep_test_cases.json \
  --save_path completions/FigStep_AsD.json --defense_name AsD
```

The settings are in `configs/defense_configs/AsD.yaml`. Select another YAML
with `--defense_config PATH`. Evaluate against an undefended completion file
generated from the same original test cases:

```bash
python evaluate_completions.py --method_name FigStep \
  --completions_path completions/FigStep_AsD.json \
  --baseline_results_path completions/FigStep_baseline.json \
  --save_path results/FigStep_AsD.json
```

## Image and prompt operations

For image width `W`, height `H`, patch aspect ratio `r` and area fraction `f`,
the reference geometry is `w = int(sqrt(W * H * f * r))`, `h = int(w / r)`.
The patch uses Pillow's original default resize operation, receives alpha
`int(0.7 * 255) = 178`, and is pasted at two inclusive integer coordinates
sampled from an instance-local `random.Random(seed)`. Integer rounding can
make the actual area slightly less than the requested fraction.

Image dimensions are preserved. Inputs whose patch is empty or cannot fit
are rejected before loading the target; geometry is not silently clamped.
The target receives the transformed image and `safety_prompt + '\n' + query`.
The original `test_case` stays in the result so baseline pairing remains valid.

AsD performs no internal target generation. Every valid input receives one
final generation at the normal token budget. It supplies no detector decision:
`blocked=false` and `detected_adversarial=null` remain true even when the target
refuses a request. Detector DSR/F1 are unavailable. ASR and benign utility use
the actual responses; refusal/severity metrics require suitable annotations.

## Patch setup and artifacts

The verified released image lives outside source at
`data/method_assets/AsD/patches/L2_noNorm_clipgrad_tips_336_LR0_1-1_1.png`.
`VLM_JAILBREAK_DATA_ROOT` redirects that root. The official patch SHA-256 is
`f02b3667c4140e9dc2bc1cafba24d8a3f6764265555904104cefd93b60df37d3`.
The exact upstream prompt is also hash-checked.

Install a local copy or download from the pinned upstream commit:

```bash
python -m defenses.asd.setup --source /path/to/released-patch.png
# Or: python -m defenses.asd.setup
```

By default, transformed images live in a temporary directory removed at
process exit. Library callers can call `defense.close()` to remove it earlier,
after generation. Set `output_dir` to retain PNG artifacts in an experiment's
own directory. Persistent paths are recorded in metadata; default temporary
paths are omitted. Custom relative paths resolve beside the YAML file;
`data/...` follows the selected data root.

Records include the source/transformed image hashes, patch hash, actual size,
coordinates, alpha integer, original/transformed text hashes and call counts.
Metadata binds configuration, code, the patch and Pillow version. Separate
baseline/defended output paths and matching provenance are required for reuse.

The RNG advances only for pending transformations. A complete seeded run uses
a fresh output and a fixed input order; a partial incremental resume does not
replay RNG draws for skipped inputs. Stored coordinates allow individual
transformations to be reconstructed.

This port covers inference with the released protective image. New patch
optimization and `mml_official` composition are outside the adapter. Registered
single-image targets can use the transform; the released target was
`liuhaotian/llava-v1.5-7b`, while validation uses the pinned HF conversion.
See [provenance](UPSTREAM.md).
