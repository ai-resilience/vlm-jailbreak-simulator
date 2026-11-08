# JPS provenance

Ported from the local reference simulator's `multi_attacks/jps/` and the
JPS-specific methods of `multi_attacks/ultrabreak/ultrabreak_adapter.py`.
Upstream: https://github.com/thu-coai/JPS
Pinned commit: `beacbc58a359005e00a7e7f7b0b485eae64bb163`.

The upstream README declares MIT licensing, but that pinned repository does
not contain the linked root LICENSE file. This integration retains that
provenance statement and does not invent an upstream license notice.

The dedicated Qwen2-VL adapter retains target-role masking and packed image
geometry. PGD, momentum, threshold stopping, best-image selection and the
feedback/summarization/revision schedule follow the reference implementation.
Finite-value checks, diagnostics, CSV/JSON integration and artifact lineage
are integration code. Only the local HuggingFace agent backend is exposed.

The reference Qwen2 adapter also derives image-packing/generation code from
[kaiyuanCui/UltraBreak](https://github.com/kaiyuanCui/UltraBreak). The reference
adapter describes that source as MIT, but the local upstream Git tree has no
root LICENSE file to copy. JPS-specific target masking is reference simulator
integration code; no missing upstream license notice is fabricated here.

The training CSV comes from `RICommunity/TAP` as linked by JPS's README;
its license and source hashes are in `data/method_assets/JPS/`.
The original upstream `clean.jpeg` is unavailable. The default config uses
an existing local image and explicitly records this substitution.
Paper reproduction is not claimed.
