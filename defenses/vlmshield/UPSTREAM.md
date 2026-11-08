# Provenance

- Source adapter: `/home/jovyan/jailbreak/vlm-jailbreak-simulator/defenses/vlmshield.py`.
- Source feature/classifier helpers: `defenses/vlmshield_utils.py` in that repository.
- Upstream: https://github.com/pgqihere/VLMShield
- Pinned commit: `344ca14778383aef57315229118f7669454fb4bb`.
- Upstream files: `VLMShield_script/VLMShield.py`, `VLMShield_script/classifier.py`
  and `VLMShield_script/model/vlmshield_classifier.pt`.
- The Apache-2.0 upstream license is preserved byte for byte in [LICENSE](LICENSE).
- Classifier size: 8,401,604 bytes.
- Classifier SHA-256: `b36702877791f347f0b846d93b9b9162c86a8895efe962b6052d322ed0c53c34`.
- CLIP: `openai/clip-vit-large-patch14`, revision
  `32bd64288804d66eefd0ccbe215aa642df71cc41`, through the existing shared loader.

The port keeps the source classifier, feature-extraction functions, float32
inference, label mapping and argmax decision. Adaptations are package imports,
the shared defense interface, lazy registry entry, portable asset paths,
fail-fast preflight and resource/code provenance. It retains the reference
adapter's optional legacy interfaces.

One source-adapter metadata bug is corrected: modality-presence booleans were
computed *after* replacing missing features with zeros, making both flags
always true. Here they are computed before the replacement. The embedding,
probabilities and decision are unchanged; absent-modality flags agree with
the actual inputs and the upstream sample processor. Missing or corrupt
images raise an error instead of silently becoming an absent modality.

The fixed blocked response is simulator plumbing, matching the source
adapter. Upstream reports filter decisions and does not generate that refusal
with a target. Adding downstream target generation and HarmBench evaluation
does not reproduce the paper's full benchmark. `paper_reproduction` is false.
