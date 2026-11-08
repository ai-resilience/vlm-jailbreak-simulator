"""Multi-Modal Linkage (MML) attack for the unified simulator.

MML conceals a typographic harmful title with one of four reversible transforms
(word replacement, horizontal mirroring, 180-degree rotation, or Base64), then
uses a cross-modal reconstruction prompt and a fictional villain-game frame.

The upstream release contains inference/evaluation code and distributes its
already-rendered images separately; it does not publish the image-generation
script.  Consequently this adapter exposes two explicit input modes:

``generated``
    A deterministic clean-room renderer implementing the procedure specified
    in the paper.  Rendering details that were not released are recorded in
    every test case and make this an adapted implementation.

``precomputed``
    Replay the separately downloaded upstream ``dataset`` tree.  Rows are
    matched to simulator behaviors by their original/rewritten prompt, and the
    released image plus metadata are used without modification.

Stage 1 represents the five-query budget as identical attempts. The
method-specific ``run_mml_official.py`` entry point interleaves generation and
CLAS evaluation, stops at score 5, and retains the highest score exactly as the
released runner does. The common three-stage evaluator remains available only
for cross-method comparisons.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import textwrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .schema import Attack, Behavior, TestCase
from data_paths import REPO_ROOT, resolve_path
from benchmark_datasets import portable_path

from .official_source import load_upstream_prompts
from .prompts import MML_PROMPTS


ADJECTIVE_REPLACEMENTS = [
    "sweet", "gentle", "joyful", "charming", "kind", "bright", "adorable",
    "delightful", "friendly", "warm", "lovely", "caring", "whimsical",
    "playful", "radiant", "graceful", "bubbly", "serene", "hopeful",
    "cheerful", "compassionate",
]
NOUN_REPLACEMENTS = [
    "pizza", "burger", "sushi", "pasta", "tacos", "steak", "ramen",
    "donut", "croissant", "brownie", "cheesecake", "pancakes", "waffles",
    "lasagna", "dumplings", "bagel", "paella", "falafel", "muffin",
    "burrito",
]

_VARIANT_ALIASES = {
    "word_replacement": "word_replacement",
    "wr": "word_replacement",
    "images_wr": "word_replacement",
    "mirror": "mirror",
    "mirroring": "mirror",
    "images_mirror": "mirror",
    "images_miror": "mirror",  # spelling in the upstream README
    "rotate": "rotate",
    "rotation": "rotate",
    "images_rotate": "rotate",
    "base64": "base64",
    "b64": "base64",
    "images_base64": "base64",
    "shift": "shift",
    "shift_cipher": "shift",
    "sc": "shift",
}
_VARIANT_DIRECTORIES = {
    "word_replacement": "images_wr",
    "mirror": "images_mirror",
    "rotate": "images_rotate",
    "base64": "images_base64",
}
_DATASET_DIRECTORIES = {
    "safebench": "safebench",
    "mm-safetybench": "mm-safebench",
    "mmsafetybench": "mm-safebench",
    "hades": "hades",
}
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().casefold())
    return value.rstrip(".!?\"' ")


def _safe_stem(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")[:80] or "case"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


class MMLAttack(Attack):
    official_repo = "https://github.com/wangyu-ovo/MML"
    official_commit = "5966fca529011e8b5ccff61d70365997d6466abc"
    # Main-paper runs replay the released images. The paper describes but does
    # not release their generator/rendering settings, so generated mode remains
    # an adapted clean-room implementation.
    algorithm_fidelity = "replay"
    official_signature: dict[str, Any] = {}

    requires_images = False
    interactive = False

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        raw_variant = str(self.config.get("variant", "mirror")).lower()
        if raw_variant not in _VARIANT_ALIASES:
            raise ValueError(
                "MML variant must be one of word_replacement, mirror, rotate, "
                f"base64, or shift; got {raw_variant!r}"
            )
        self.variant = _VARIANT_ALIASES[raw_variant]
        self.input_mode = str(self.config.get("input_mode", "generated")).lower()
        if self.input_mode not in {"generated", "precomputed"}:
            raise ValueError("MML input_mode must be 'generated' or 'precomputed'")
        if self.input_mode == "precomputed" and self.variant == "shift":
            raise ValueError(
                "MML shift cipher is a paper extension without a released image "
                "directory; use input_mode=generated"
            )

        self.prompt_source = str(self.config.get("prompt_source", "clean_room")).lower()
        if self.prompt_source not in {"clean_room", "upstream"}:
            raise ValueError("MML prompt_source must be 'clean_room' or 'upstream'")
        if self.prompt_source == "upstream" and self.variant == "shift":
            raise ValueError("The released MML source has no shift-cipher prompt constant")
        self.upstream_const_path = resolve_path(
            self.config.get("upstream_const_path", "data/method_assets/MML/upstream/const.py")
        )
        expected = self.config.get("upstream_const_sha256")
        self.upstream_const_sha256 = str(expected) if expected else None
        self._upstream_prompts: dict[str, str] | None = None

        self.attempts = int(self.config.get("attempts", 1))
        if self.attempts < 1:
            raise ValueError("MML attempts must be >= 1")
        self.num_attempts = self.attempts

        configured_seed = self.config.get("seed", 0)
        self.seed = None if configured_seed is None else int(configured_seed)
        self.title_source = str(
            self.config.get("title_source", "statement_or_instruction")
        ).lower()
        if self.title_source not in {
            "statement_or_instruction", "statement", "instruction"
        }:
            raise ValueError(
                "MML title_source must be statement_or_instruction, statement, "
                "or instruction"
            )

        self.canvas_width = int(self.config.get("canvas_width", 760))
        self.canvas_height = int(self.config.get("canvas_height", 760))
        self.font_size = int(self.config.get("font_size", 60))
        self.wrap_width = int(self.config.get("wrap_width", 22))
        self.steps = int(self.config.get("steps", 3))
        if min(self.canvas_width, self.canvas_height, self.font_size,
               self.wrap_width, self.steps) < 1:
            raise ValueError("MML rendering dimensions, wrap_width, and steps must be positive")
        self.font_path = self.config.get("font_path", "FreeMonoBold.ttf")
        self.strict_font = bool(self.config.get("strict_font", False))
        self.pos_tagger = str(self.config.get("pos_tagger", "nltk")).lower()
        if self.pos_tagger not in {"nltk", "heuristic"}:
            raise ValueError("MML pos_tagger must be 'nltk' or 'heuristic'")

        self.precomputed_root = resolve_path(
            self.config.get("precomputed_root", "data/method_assets/MML")
        )
        self._font_source: str | None = None
        self._precomputed_indices: dict[str, dict[str, dict[str, Any]]] = {}

    def effective_config(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "input_mode": self.input_mode,
            "prompt_source": self.prompt_source,
            "upstream_const_path": portable_path(self.upstream_const_path),
            "upstream_const_sha256": self.upstream_const_sha256,
            "attempts": self.attempts,
            "seed": self.seed,
            "title_source": self.title_source,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "font_size": self.font_size,
            "wrap_width": self.wrap_width,
            "steps": self.steps,
            "font_path": self.font_path,
            "strict_font": self.strict_font,
            "pos_tagger": self.pos_tagger,
            "precomputed_root": portable_path(self.precomputed_root),
        }

    def provenance(self) -> dict[str, Any]:
        record = super().provenance()
        fidelity = (
            "replay"
            if self.input_mode == "precomputed" and self.prompt_source == "upstream"
            else "adapted"
        )
        record["algorithm_fidelity"] = fidelity
        # Attack.provenance computes this from the class-level value. MML is a
        # released-artifact replay (or an adapted renderer), never a faithful
        # reconstruction of the unpublished image generator.
        record["paper_reproduction"] = False
        return record

    def _prompt(self, mapping: dict[str, str], shuffled: list[str]) -> str:
        if self.prompt_source == "upstream":
            if self._upstream_prompts is None:
                self._upstream_prompts = load_upstream_prompts(
                    self.upstream_const_path,
                    expected_sha256=self.upstream_const_sha256,
                )
            template = self._upstream_prompts[self.variant]
            if self.variant == "word_replacement":
                return template.format(mapping, shuffled)
            return template.format(shuffled)
        return MML_PROMPTS[self.variant].format(
            replacement_map=repr(mapping), shuffled_words=repr(shuffled)
        )

    # -- title, hint, and replacement construction ----------------------
    def _title(self, behavior: Behavior) -> tuple[str, str]:
        if self.title_source == "statement":
            if not behavior.statement:
                raise ValueError(
                    f"MML title_source=statement but {behavior.behavior_id} has no statement"
                )
            return behavior.statement, "statement"
        if self.title_source == "instruction":
            return behavior.instruction, "instruction"
        if behavior.statement:
            return behavior.statement, "statement"
        # MM-SafetyBench exposes the baseline's declarative rewrite as
        # ``Changed Question``. MML encrypts that rewritten query rather than
        # the separate text prompt that refers to an image.
        changed = behavior.meta.get("changed_question")
        if changed:
            return str(changed), "meta.changed_question"
        return behavior.instruction, "instruction"

    @staticmethod
    def _candidate_titles(behavior: Behavior) -> list[str]:
        values = [
            behavior.meta.get("mml_original_prompt"),
            behavior.statement,
            behavior.meta.get("changed_question"),
            behavior.instruction,
        ]
        return [str(value) for value in values if value]

    def _rng(self, behavior_id: str) -> random.Random:
        if self.seed is None:
            return random.Random()
        material = f"{self.seed}\0{behavior_id}".encode("utf-8")
        derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return random.Random(derived)

    @staticmethod
    def _shuffled_words(title: str, rng: random.Random) -> list[str]:
        words = title.split()
        rng.shuffle(words)
        return words

    @staticmethod
    def _heuristic_tags(tokens: list[str]) -> list[tuple[str, str]]:
        """Dependency-free fallback, intentionally marked non-official in meta."""
        adjectives = {
            "illegal", "malicious", "harmful", "dangerous", "violent", "fake",
            "fraudulent", "stolen", "offensive", "private", "secret", "lethal",
        }
        noun_suffixes = ("tion", "ment", "ness", "ity", "ware", "er", "or", "ist")
        tags: list[tuple[str, str]] = []
        for token in tokens:
            lower = token.casefold()
            if lower in adjectives or lower.endswith(("ous", "ful", "ive", "al", "ic")):
                tag = "JJ"
            elif token.isalpha() and (
                lower.endswith(noun_suffixes)
                or lower not in {"to", "a", "an", "the", "and", "or", "for", "of",
                                 "in", "on", "with", "without", "from", "by"}
            ):
                tag = "NN"
            else:
                tag = "OTHER"
            tags.append((token, tag))
        return tags

    def _tag_tokens(self, title: str) -> tuple[list[tuple[str, str]], str]:
        # Treebank tokenization requires no downloaded corpus.  The official
        # NLTK perceptron tagger does; its missing-resource error is made
        # actionable instead of silently changing the algorithm.
        try:
            from nltk import pos_tag
            from nltk.tokenize import TreebankWordTokenizer

            tokens = TreebankWordTokenizer().tokenize(title)
            if self.pos_tagger == "heuristic":
                return self._heuristic_tags(tokens), "heuristic"
            try:
                return pos_tag(tokens), "nltk"
            except LookupError as exc:
                raise RuntimeError(
                    "MML word replacement requires NLTK's English perceptron "
                    "tagger. For the pinned NLTK 3.8.1 run `python -m "
                    "nltk.downloader averaged_perceptron_tagger`; newer NLTK "
                    "releases may require `averaged_perceptron_tagger_eng`. "
                    "Alternatively, explicitly set "
                    "pos_tagger: heuristic for an adapted fallback."
                ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "MML word replacement requires the `nltk` package; install the "
                "project requirements first."
            ) from exc

    def _replace_words(self, title: str) -> tuple[str, dict[str, str], str]:
        tagged, tagger_used = self._tag_tokens(title)
        noun_index = adjective_index = 0
        replaced: list[str] = []
        mapping: dict[str, str] = {}
        for token, tag in tagged:
            if tag.startswith("NN"):
                if noun_index >= len(NOUN_REPLACEMENTS):
                    raise ValueError("MML title contains more nouns than released replacements")
                substitute = NOUN_REPLACEMENTS[noun_index]
                noun_index += 1
                mapping[substitute] = token
                replaced.append(substitute)
            elif tag.startswith("JJ"):
                if adjective_index >= len(ADJECTIVE_REPLACEMENTS):
                    raise ValueError(
                        "MML title contains more adjectives than released replacements"
                    )
                substitute = ADJECTIVE_REPLACEMENTS[adjective_index]
                adjective_index += 1
                mapping[substitute] = token
                replaced.append(substitute)
            else:
                replaced.append(token)

        try:
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            rendered = TreebankWordDetokenizer().detokenize(replaced)
        except ImportError:  # covered by _tag_tokens, defensive only
            rendered = " ".join(replaced)
        # Keep output stable across NLTK versions whose Treebank detokenizer
        # differs on a stand-alone punctuation token.
        rendered = re.sub(r"\s+([.,!?;:])", r"\1", rendered)
        return rendered, mapping, tagger_used

    # -- typography ------------------------------------------------------
    def _font(self) -> ImageFont.ImageFont:
        candidates = [self.font_path]
        if self.font_path and not os.path.isabs(self.font_path):
            candidates.append(os.path.join(REPO_ROOT, self.font_path))
        candidates.extend([
            "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ])
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                self._font_source = candidate
                return ImageFont.truetype(candidate, self.font_size)
        if self.strict_font:
            raise FileNotFoundError(
                "MML strict rendering requires the configured font; set font_path "
                "or disable strict_font"
            )
        self._font_source = "Pillow-default"
        return ImageFont.load_default(size=self.font_size)

    def _render_typography(self, text: str) -> Image.Image:
        body = textwrap.fill(text.removesuffix("\n"), width=self.wrap_width)
        for index in range(1, self.steps + 1):
            body += f"\n{index}. "
        image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        ImageDraw.Draw(image).text(
            (20, 10), body, font=self._font(), fill="black", spacing=11
        )
        return image

    @staticmethod
    def _shift_cipher(text: str) -> str:
        """Paper Section 5.6 Caesar shift: a->b, ..., z->a (case preserving)."""

        output: list[str] = []
        for character in text:
            if "a" <= character <= "z":
                output.append(chr((ord(character) - ord("a") + 1) % 26 + ord("a")))
            elif "A" <= character <= "Z":
                output.append(chr((ord(character) - ord("A") + 1) % 26 + ord("A")))
            else:
                output.append(character)
        return "".join(output)

    def _generated_payload(
        self, behavior: Behavior, out_dir: str
    ) -> tuple[str, str, list[str], dict[str, Any]]:
        title, title_source = self._title(behavior)
        rng = self._rng(behavior.behavior_id)
        shuffled = self._shuffled_words(title, rng)
        mapping: dict[str, str] = {}
        tagger_used: str | None = None

        if self.variant == "word_replacement":
            image_text, mapping, tagger_used = self._replace_words(title)
            image = self._render_typography(image_text)
        elif self.variant == "base64":
            image_text = base64.b64encode(title.encode("utf-8")).decode("ascii")
            image = self._render_typography(image_text)
        elif self.variant == "shift":
            image_text = self._shift_cipher(title)
            image = self._render_typography(image_text)
        else:
            image_text = title
            image = self._render_typography(title)
            if self.variant == "mirror":
                image = ImageOps.mirror(image)
            else:
                image = image.rotate(180)

        os.makedirs(out_dir, exist_ok=True)
        image_path = os.path.join(out_dir, f"{_safe_stem(behavior.behavior_id)}.png")
        image.save(image_path)
        prompt = self._prompt(mapping, shuffled)
        metadata = {
            "input_mode": "generated",
            "variant": self.variant,
            "original_title": title,
            "title_source": title_source,
            "image_text": image_text,
            "replacement_map": mapping,
            "shuffled_words": shuffled,
            "pos_tagger_used": tagger_used,
            "font_source": self._font_source,
            "rendering": {
                "canvas": [self.canvas_width, self.canvas_height],
                "font_size": self.font_size,
                "wrap_width": self.wrap_width,
                "steps": self.steps,
                "spacing": 11,
                "origin": [20, 10],
            },
            "official_precomputed_image": False,
            "prompt_source": self.prompt_source,
        }
        return portable_path(image_path), prompt, shuffled, metadata

    # -- replay of upstream Drive data ----------------------------------
    @staticmethod
    def _dataset_key(name: str) -> str:
        compact = name.casefold().replace("_", "-").replace(" ", "")
        return _DATASET_DIRECTORIES.get(compact, name.casefold())

    def _build_precomputed_index(self, dataset_name: str) -> dict[str, dict[str, Any]]:
        dataset_dir = os.path.join(self.precomputed_root, self._dataset_key(dataset_name))
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(
                f"MML precomputed dataset directory not found: {dataset_dir}. "
                "Download the official Drive folder and preserve its dataset tree."
            )
        index: dict[str, dict[str, Any]] = {}
        for subclass in sorted(os.scandir(dataset_dir), key=lambda item: item.name):
            if not subclass.is_dir() or subclass.name.startswith("."):
                continue
            metadata_path = os.path.join(subclass.path, "data.json")
            if not os.path.isfile(metadata_path):
                continue
            with open(metadata_path, encoding="utf-8") as source:
                rows = json.load(source)
            if not isinstance(rows, list):
                raise ValueError(f"MML metadata must contain a JSON list: {metadata_path}")
            for row in rows:
                if not isinstance(row, dict) or "original_prompt" not in row or "id" not in row:
                    raise ValueError(f"Malformed MML metadata row in {metadata_path}")
                title = str(row["original_prompt"])
                key = _normalize_text(title)
                if key in index:
                    raise ValueError(
                        f"Duplicate normalized MML title {title!r} in {dataset_dir}"
                    )
                index[key] = {
                    "row": row,
                    "subclass": subclass.name,
                    "subclass_path": subclass.path,
                    "metadata_path": metadata_path,
                }
        if not index:
            raise FileNotFoundError(f"No MML data.json rows found below {dataset_dir}")
        return index

    def _precomputed_payload(
        self, behavior: Behavior
    ) -> tuple[str, str, list[str], dict[str, Any]]:
        record = self._find_precomputed_record(behavior)
        if record is None:
            raise KeyError(
                f"No official MML row matches behavior {behavior.behavior_id!r}; "
                "matching checks both statement and instruction text."
            )
        row = record["row"]
        title = str(row["original_prompt"])
        shuffled = self._shuffled_words(title, self._rng(behavior.behavior_id))
        mapping = row.get("replace_map", {})
        if self.variant == "word_replacement" and not isinstance(mapping, dict):
            raise ValueError("Official MML word-replacement row lacks replace_map")

        image_dir = os.path.join(
            record["subclass_path"], _VARIANT_DIRECTORIES[self.variant]
        )
        stem = str(row["id"])
        image_path = next(
            (os.path.join(image_dir, stem + extension)
             for extension in _IMAGE_EXTENSIONS
             if os.path.isfile(os.path.join(image_dir, stem + extension))),
            None,
        )
        if image_path is None:
            raise FileNotFoundError(
                f"Official MML image missing for row {stem!r} below {image_dir}"
            )
        prompt = self._prompt(mapping, shuffled)
        metadata = {
            "input_mode": "precomputed",
            "variant": self.variant,
            "original_title": title,
            "title_source": "official_data.json",
            "replacement_map": mapping,
            "shuffled_words": shuffled,
            "official_subclass": record["subclass"],
            "official_row_id": row["id"],
            "official_metadata": portable_path(record["metadata_path"]),
            "official_precomputed_image": True,
            "prompt_source": self.prompt_source,
        }
        return portable_path(image_path), prompt, shuffled, metadata

    def _find_precomputed_record(self, behavior: Behavior) -> dict[str, Any] | None:
        dataset_key = self._dataset_key(behavior.dataset)
        if dataset_key not in self._precomputed_indices:
            self._precomputed_indices[dataset_key] = self._build_precomputed_index(
                behavior.dataset
            )
        index = self._precomputed_indices[dataset_key]
        return next((
            index.get(_normalize_text(value))
            for value in self._candidate_titles(behavior)
            if _normalize_text(value) in index
        ), None)

    def missing_requirements(self, behavior: Behavior) -> list[str]:
        missing = super().missing_requirements(behavior)
        if self.input_mode != "precomputed" or missing:
            return missing
        try:
            record = self._find_precomputed_record(behavior)
        except FileNotFoundError:
            return ["official_mml_dataset"]
        if record is None:
            # Expected for the 500 MM-SafetyBench rows removed by the paper's
            # CLAS score-4 filtering. Stage 1 records the skip explicitly.
            if self._dataset_key(behavior.dataset) == "mm-safebench":
                return ["official_mml_filtered_row"]
            # Other official datasets are not filtered. Let build() raise so a
            # mismatched or incomplete release cannot silently shrink the run.
            return []
        return []

    # -- unified entry point --------------------------------------------
    def build(self, behavior: Behavior, out_dir: str) -> list[TestCase]:
        if self.input_mode == "precomputed":
            image_path, prompt, _, metadata = self._precomputed_payload(behavior)
        else:
            image_path, prompt, _, metadata = self._generated_payload(behavior, out_dir)

        cases: list[TestCase] = []
        for attempt in range(self.attempts):
            cases.append(TestCase(
                test_case_id=self.test_case_id(
                    self.name, behavior.behavior_id, attempt
                ),
                behavior_id=behavior.behavior_id,
                attack=self.name,
                text=prompt,
                images=[image_path],
                attempt=attempt,
                meta={**metadata, "query_budget": self.attempts},
            ))
        return cases

