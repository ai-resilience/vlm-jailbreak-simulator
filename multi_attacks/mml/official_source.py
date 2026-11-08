"""Read prompt literals from a pinned checkout of the upstream MML source.

The upstream repository does not declare a software license.  The simulator
therefore does not redistribute its prompt text.  An official-profile run
instead reads the four string constants from a user-supplied checkout after
verifying the exact ``const.py`` hash reviewed by this project.

Only top-level string assignments are accepted.  The file is parsed as Python
syntax but never imported or executed.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


UPSTREAM_PROMPT_NAMES = {
    "word_replacement": "wr_game_prompt",
    "mirror": "mirror_game_prompt",
    "rotate": "rotate_game_prompt",
    "base64": "base64_game_prompt",
}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_upstream_constants(
    path: str,
    names: set[str],
    *,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    """Load selected released string constants without executing upstream code."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Pinned MML prompt source not found: {source_path}. Clone the "
            "official repository at commit 5966fca5 and set upstream_const_path."
        )
    actual_sha256 = file_sha256(str(source_path))
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            "MML upstream const.py hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    constants: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in names:
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            raise ValueError(f"MML prompt {target.id!r} is not a string literal")
        constants[target.id] = value

    missing = sorted(names - constants.keys())
    if missing:
        raise ValueError(
            f"Pinned MML const.py is missing prompt constant(s): {', '.join(missing)}"
        )
    return constants


def load_upstream_prompts(
    path: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    """Load the four main-experiment prompts from a pinned checkout."""

    constants = load_upstream_constants(
        path, set(UPSTREAM_PROMPT_NAMES.values()),
        expected_sha256=expected_sha256,
    )
    return {
        variant: constants[name]
        for variant, name in UPSTREAM_PROMPT_NAMES.items()
    }


def load_upstream_judge_prompt(
    path: str,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Load the released CLAS ``judge_template`` literal."""

    return load_upstream_constants(
        path, {"judge_template"}, expected_sha256=expected_sha256
    )["judge_template"]
