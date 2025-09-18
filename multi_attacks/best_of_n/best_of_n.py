import json
import os
import random
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from attacks.baseline import Baseattack


@dataclass
class TextAugmentation:
    seed: int
    word_scrambling: bool
    random_capitalization: bool
    ascii_perturbation: bool
    random_prefix_length: int = 0
    random_suffix_length: int = 0

    def __str__(self) -> str:
        return (
            "seed={seed}, RandomCap: {rand_cap}, ASCIIPerturb: {ascii}, "
            "WordScramble: {scramble}, PrefixLength: {prefix}, SuffixLength: {suffix}"
        ).format(
            seed=self.seed,
            rand_cap=self.random_capitalization,
            ascii=self.ascii_perturbation,
            scramble=self.word_scrambling,
            prefix=self.random_prefix_length,
            suffix=self.random_suffix_length,
        )

    def dict(self) -> Dict[str, Any]:
        return asdict(self)


def _random_attack_string(num_tokens: int, rng: random.Random) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    tokens = []
    for _ in range(max(num_tokens, 0)):
        length = rng.randint(3, 8)
        token = "".join(rng.choice(alphabet) for _ in range(length))
        tokens.append(token)
    return " ".join(tokens)


def _apply_word_scrambling(text: str, sigma: float, rng: random.Random) -> str:
    words = text.split()
    scrambled_words = []
    for word in words:
        if len(word) > 3 and rng.random() < sigma ** 0.5:
            chars = list(word)
            middle = chars[1:-1]
            rng.shuffle(middle)
            scrambled_words.append(chars[0] + "".join(middle) + chars[-1])
        else:
            scrambled_words.append(word)
    return " ".join(scrambled_words)


def _apply_random_capitalization(text: str, sigma: float, rng: random.Random) -> str:
    updated = []
    for char in text:
        if char.isalpha() and rng.random() < sigma ** 0.5:
            if "a" <= char <= "z":
                updated.append(chr(ord(char) - 32))
            else:
                updated.append(chr(ord(char) + 32))
        else:
            updated.append(char)
    return "".join(updated)


def _apply_ascii_noising(text: str, sigma: float, rng: random.Random) -> str:
    updated = []
    for char in text:
        if char.isprintable() and rng.random() < sigma ** 3:
            pert = rng.choice([-1, 1])
            new_char_code = ord(char) + pert
            if 32 <= new_char_code <= 126:
                updated.append(chr(new_char_code))
            else:
                updated.append(char)
        else:
            updated.append(char)
    return "".join(updated)


def process_text_augmentation(
    text: str,
    sigma: float,
    seed: int,
    word_scrambling: bool,
    random_capitalization: bool,
    ascii_perturbation: bool,
    random_prefix_length: int = 0,
    random_suffix_length: int = 0,
) -> Tuple[str, TextAugmentation]:
    rng = random.Random(seed)

    augmentation = TextAugmentation(
        seed=seed,
        word_scrambling=word_scrambling,
        random_capitalization=random_capitalization,
        ascii_perturbation=ascii_perturbation,
        random_prefix_length=random_prefix_length,
        random_suffix_length=random_suffix_length,
    )

    if random_prefix_length > 0:
        prefix = _random_attack_string(random_prefix_length, rng)
        text = prefix + "\n\n" + text
    if random_suffix_length > 0:
        suffix = _random_attack_string(random_suffix_length, rng)
        text = text + "\n\n" + suffix
    if word_scrambling:
        text = _apply_word_scrambling(text, sigma, rng)
    if random_capitalization:
        text = _apply_random_capitalization(text, sigma, rng)
    if ascii_perturbation:
        text = _apply_ascii_noising(text, sigma, rng)

    return text, augmentation


def _augment_msj_prefixes(
    prefixes: Sequence[Sequence[str]],
    sigma: float,
    seed: int,
    word_scrambling: bool,
    random_capitalization: bool,
    ascii_perturbation: bool,
    optim_user: bool,
    optim_assistant: bool,
) -> List[Tuple[str, str]]:
    augmented: List[Tuple[str, str]] = []
    for idx, (user_content, assistant_content) in enumerate(prefixes):
        if optim_user:
            user_content, _ = process_text_augmentation(
                user_content,
                sigma,
                seed + idx,
                word_scrambling,
                random_capitalization,
                ascii_perturbation,
            )
        if optim_assistant:
            assistant_content, _ = process_text_augmentation(
                assistant_content,
                sigma,
                seed + idx + 10_000,
                word_scrambling,
                random_capitalization,
                ascii_perturbation,
            )
        augmented.append((user_content, assistant_content))
    return augmented


def process_decorated_text_with_augmentations(
    text: str,
    prefix: Optional[str],
    suffix: Optional[str],
    optim_harmful_text: bool,
    optim_prefix: bool,
    optim_suffix: bool,
    sigma: float,
    seed: int,
    word_scrambling: bool,
    random_capitalization: bool,
    ascii_perturbation: bool,
    random_prefix_length: int,
    random_suffix_length: int,
    msj_num_shots: int,
    msj_path: Optional[Path],
    msj_shuffle: bool,
    optim_msj_user_content: bool,
    optim_msj_assistant_content: bool,
) -> Tuple[str, Optional[TextAugmentation], Optional[List[Tuple[str, str]]]]:
    if text.strip() == "":
        raise ValueError("Main text cannot be empty")
    if optim_prefix and prefix is None:
        raise ValueError("Cannot optimize prefix when prefix is None")
    if optim_suffix and suffix is None:
        raise ValueError("Cannot optimize suffix when suffix is None")
    if not optim_prefix and not optim_suffix and not optim_harmful_text and msj_num_shots == 0:
        raise ValueError(
            "At least one of optim_prefix, optim_suffix, optim_harmful_text, or msj_num_shots must be > 0"
        )
    if (random_prefix_length > 0 or random_suffix_length > 0) and (prefix or suffix or msj_num_shots > 0):
        raise ValueError(
            "Random prefix/suffix augmentation is only supported when prefix, suffix, and msj_num_shots are unset"
        )

    msj_prefixes: Optional[List[Tuple[str, str]]] = None
    if msj_num_shots > 0:
        if msj_path is None:
            raise ValueError("msj_num_shots>0 requires msj_path to be provided")
        with open(msj_path, "r", encoding="utf-8") as file:
            loaded_prefixes = json.load(file)
        if not isinstance(loaded_prefixes, list):
            raise ValueError("MSJ prompt file must contain a list of [user, assistant] pairs")
        rng = random.Random(seed)
        if msj_shuffle:
            rng.shuffle(loaded_prefixes)
        msj_prefixes = [tuple(entry) for entry in loaded_prefixes[:msj_num_shots]]
        if optim_msj_user_content or optim_msj_assistant_content:
            msj_prefixes = _augment_msj_prefixes(
                msj_prefixes,
                sigma,
                seed + 50_000,
                word_scrambling,
                random_capitalization,
                ascii_perturbation,
                optim_msj_user_content,
                optim_msj_assistant_content,
            )

    processed_parts: List[str] = []
    text_augmentation: Optional[TextAugmentation] = None

    if prefix is not None:
        if optim_prefix:
            prefix, _ = process_text_augmentation(
                prefix,
                sigma,
                seed + 1,
                word_scrambling,
                random_capitalization,
                ascii_perturbation,
            )
        processed_parts.append(prefix)

    if optim_harmful_text:
        text, text_augmentation = process_text_augmentation(
            text,
            sigma,
            seed,
            word_scrambling,
            random_capitalization,
            ascii_perturbation,
            random_prefix_length,
            random_suffix_length,
        )
    processed_parts.append(text)

    if suffix is not None:
        if optim_suffix:
            suffix, _ = process_text_augmentation(
                suffix,
                sigma,
                seed + 2,
                word_scrambling,
                random_capitalization,
                ascii_perturbation,
            )
        processed_parts.append(suffix)

    final_text = " ".join(part for part in processed_parts if part)
    return final_text, text_augmentation, msj_prefixes


class BestOfN(Baseattack):
    """Lightweight integration of the BoN text search strategy."""

    _ALLOWED_STRATEGIES = {"balanced", "diff", "length", "random", "first"}

    def __init__(self, **kwargs: Any) -> None:
        print("=== MULTI_ATTACKS BESTOFN INITIALIZED ===")

        self.num_candidates = int(kwargs.get("num_candidates", 8))
        self.num_steps = int(kwargs.get("num_steps", 1))
        self.seed = int(kwargs.get("seed", 0))

        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")

        self.sigma = float(kwargs.get("sigma", 0.4))
        self.word_scrambling = bool(kwargs.get("word_scrambling", True))
        self.random_capitalization = bool(kwargs.get("random_capitalization", True))
        self.ascii_perturbation = bool(kwargs.get("ascii_perturbation", True))
        self.random_prefix_length = int(kwargs.get("random_prefix_length", 0))
        self.random_suffix_length = int(kwargs.get("random_suffix_length", 0))

        self.optim_harmful_text = bool(kwargs.get("optim_harmful_text", True))
        self.optim_prefix = bool(kwargs.get("optim_prefix", False))
        self.optim_suffix = bool(kwargs.get("optim_suffix", False))
        self.prefix = self._load_optional_prompt(kwargs.get("prefix_path"))
        self.suffix = self._load_optional_prompt(kwargs.get("suffix_path"))
        if self.optim_prefix and self.prefix is None:
            raise ValueError("optim_prefix=True requires prefix_path to be set")
        if self.optim_suffix and self.suffix is None:
            raise ValueError("optim_suffix=True requires suffix_path to be set")

        self.msj_num_shots = int(kwargs.get("msj_num_shots", 0))
        self.msj_shuffle = bool(kwargs.get("msj_shuffle", True))
        self.optim_msj_user_content = bool(kwargs.get("optim_msj_user_content", True))
        self.optim_msj_assistant_content = bool(kwargs.get("optim_msj_assistant_content", True))
        self.msj_path = self._coerce_optional_path(kwargs.get("msj_path"))
        if self.msj_num_shots > 0 and self.msj_path is None:
            raise ValueError("msj_num_shots>0 requires msj_path to be set")

        self.score_strategy = str(kwargs.get("score_strategy", "balanced"))
        if self.score_strategy not in self._ALLOWED_STRATEGIES:
            raise ValueError(
                f"Unsupported score_strategy '{self.score_strategy}'. "
                f"Expected one of {sorted(self._ALLOWED_STRATEGIES)}"
            )
        self.diff_weight = float(kwargs.get("diff_weight", 0.7))
        self.length_weight = float(kwargs.get("length_weight", 0.3))
        self.save_all_candidates = bool(kwargs.get("save_all_candidates", True))

        image_root = kwargs.get("image_root", "data/multimodal_behavior_images")
        if image_root and str(image_root).lower() != "none":
            self.image_root = Path(image_root)
        else:
            self.image_root = None

    def generate_test_cases(
        self, behaviors: List[Dict[str, str]], save_dir: str, verbose: bool = False
    ) -> Dict[str, Any]:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        image_dir: Optional[Path] = None

        test_cases: Dict[str, Any] = {}
        candidate_logs: Dict[str, List[Dict[str, Any]]] = {}

        for idx, behavior_dict in enumerate(tqdm(behaviors, total=len(behaviors))):
            behavior = behavior_dict["Behavior"].strip()
            behavior_id = behavior_dict["BehaviorID"]
            if verbose:
                print(f"Behavior: {behavior}")
                print(f"Behavior ID: {behavior_id}")

            candidates = self._generate_candidates(behavior, idx)
            if not candidates:
                continue

            best_candidate = max(candidates, key=lambda c: c["score"])
            image_reference = self._extract_image_reference(behavior_dict)
            if image_reference is not None:
                if image_dir is None:
                    image_dir = save_dir_path / "images"
                    image_dir.mkdir(parents=True, exist_ok=True)
                source_path = self._resolve_image_source(image_reference)
                image_filename = source_path.name
                destination_path = image_dir / image_filename
                should_copy = True
                if destination_path.exists():
                    try:
                        should_copy = not destination_path.samefile(source_path)
                    except FileNotFoundError:
                        should_copy = True
                if should_copy:
                    copy2(source_path, destination_path)
                test_cases[behavior_id] = [image_filename, best_candidate["prompt"]]
            else:
                test_cases[behavior_id] = best_candidate["prompt"]
            if self.save_all_candidates:
                candidate_logs[behavior_id] = candidates

        test_cases_path = os.path.join(save_dir, "BestOfN_test_cases.json")
        with open(test_cases_path, "w", encoding="utf-8") as file:
            json.dump(test_cases, file, indent=2)

        if self.save_all_candidates and candidate_logs:
            logs_path = os.path.join(save_dir, "BestOfN_candidate_logs.json")
            with open(logs_path, "w", encoding="utf-8") as file:
                json.dump(candidate_logs, file, indent=2)

        return test_cases

    def _generate_candidates(self, behavior: str, behavior_index: int) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        base_seed = self.seed + behavior_index * max(1, self.num_candidates * self.num_steps)

        # We sweep through the original BoN sampling grid and score each prompt locally.
        for step in range(self.num_steps):
            for candidate_idx in range(self.num_candidates):
                candidate_seed = base_seed + step * self.num_candidates + candidate_idx

                prompt, augmentation, msj_prefixes = process_decorated_text_with_augmentations(
                    text=behavior,
                    prefix=self.prefix,
                    suffix=self.suffix,
                    optim_harmful_text=self.optim_harmful_text,
                    optim_prefix=self.optim_prefix,
                    optim_suffix=self.optim_suffix,
                    sigma=self.sigma,
                    seed=candidate_seed,
                    word_scrambling=self.word_scrambling,
                    random_capitalization=self.random_capitalization,
                    ascii_perturbation=self.ascii_perturbation,
                    random_prefix_length=self.random_prefix_length,
                    random_suffix_length=self.random_suffix_length,
                    msj_num_shots=self.msj_num_shots,
                    msj_path=self.msj_path,
                    msj_shuffle=self.msj_shuffle,
                    optim_msj_user_content=self.optim_msj_user_content,
                    optim_msj_assistant_content=self.optim_msj_assistant_content,
                )

                score = self._score_candidate(
                    original=behavior,
                    candidate=prompt,
                    global_index=step * self.num_candidates + candidate_idx,
                    base_seed=base_seed,
                )

                candidate_entry: Dict[str, Any] = {
                    "prompt": prompt,
                    "score": score,
                    "step": step,
                    "candidate_index": candidate_idx,
                    "seed": candidate_seed,
                    "augmentation": augmentation.dict() if augmentation is not None else None,
                }
                if msj_prefixes is not None:
                    candidate_entry["msj_prefixes"] = msj_prefixes

                candidates.append(candidate_entry)

        return candidates

    def _score_candidate(self, original: str, candidate: str, global_index: int, base_seed: int) -> float:
        if self.score_strategy == "first":
            return -float(global_index)
        if self.score_strategy == "random":
            rng = random.Random(base_seed + global_index)
            return rng.random()

        if self.score_strategy == "length":
            return float(len(candidate.split()))

        diff_score = 1.0 - SequenceMatcher(None, original, candidate).ratio()
        if self.score_strategy == "diff":
            return diff_score

        length_score = len(candidate.split()) / max(len(original.split()), 1)
        # Balanced mode rewards both semantic drift and length gains.
        return self.diff_weight * diff_score + self.length_weight * length_score

    @staticmethod
    def _load_optional_prompt(path_like: Optional[str]) -> Optional[str]:
        if not path_like or str(path_like).lower() == "none":
            return None
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        try:
            from bon.utils.utils import get_prompt_template  # type: ignore
        except ModuleNotFoundError:
            return path.read_text(encoding="utf-8").strip()

        try:
            return get_prompt_template(str(path)).render().strip()
        except Exception:
            return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _coerce_optional_path(path_like: Optional[str]) -> Optional[Path]:
        if not path_like or str(path_like).lower() == "none":
            return None
        return Path(path_like)

    @staticmethod
    def _extract_image_reference(behavior_dict: Dict[str, Any]) -> Optional[str]:
        for key in (
            "ImageFileName",
            "image_filename",
            "ImagePath",
            "image_path",
            "image",
            "Image",
        ):
            value = behavior_dict.get(key)
            if value is None:
                continue
            value_str = str(value).strip()
            if not value_str or value_str.lower() == "none":
                continue
            return value_str
        return None

    def _resolve_image_source(self, reference: str) -> Path:
        ref_path = Path(reference)
        candidates = []

        if ref_path.is_absolute():
            candidates.append(ref_path)
        if self.image_root is not None:
            candidates.append(self.image_root / ref_path)
            candidates.append(self.image_root / ref_path.name)
        if not ref_path.is_absolute():
            candidates.append(ref_path)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not locate image '{reference}'. Set image_root to the directory containing the source images."
        )
