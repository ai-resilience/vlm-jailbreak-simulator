import json
import os
import random
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
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


@dataclass
class ImageAugmentation:
    position: Tuple[int, int]
    font: int
    font_scale: float
    color: Tuple[int, int, int]
    thickness: int

    def __str__(self) -> str:
        return (
            "Position: {position}, Font: {font}, FontScale: {scale:.2f}, "
            "Color: {color}, Thickness: {thickness}"
        ).format(
            position=self.position,
            font=self.font,
            scale=self.font_scale,
            color=self.color,
            thickness=self.thickness,
        )

    def dict(self) -> Dict[str, Any]:
        return asdict(self)


_CV2_FONTS: Tuple[int, ...] = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
)


def _generate_block_background(
    height: int,
    width: int,
    rng: random.Random,
    background_color: str,
    block_size_rng: Tuple[int, int],
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    num_blocks = rng.randint(50, 80)
    min_divisor, max_divisor = block_size_rng
    min_divisor = max(1, min_divisor)
    max_divisor = max(min_divisor, max_divisor)

    for _ in range(num_blocks):
        block_height = max(1, rng.randint(height // max_divisor, max(height // min_divisor, 1)))
        block_width = max(1, rng.randint(width // max_divisor, max(width // min_divisor, 1)))
        max_y = max(height - block_height, 0)
        max_x = max(width - block_width, 0)
        y = rng.randint(0, max_y) if max_y > 0 else 0
        x = rng.randint(0, max_x) if max_x > 0 else 0

        if background_color == "bw":
            value = rng.randint(0, 255)
            color = (value, value, value)
        else:
            color = tuple(rng.randint(0, 255) for _ in range(3))

        image[y : y + block_height, x : x + block_width] = color

    return image


def _generate_grid_background(
    height: int,
    width: int,
    rng: random.Random,
    background_color: str,
    block_size_rng: Tuple[int, int],
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    min_blocks, max_blocks = block_size_rng
    min_blocks = max(1, min_blocks)
    max_blocks = max(min_blocks, max_blocks)

    num_rows = rng.randint(min_blocks, max_blocks)
    num_cols = rng.randint(min_blocks, max_blocks)

    block_height = max(1, height // max(num_rows, 1))
    block_width = max(1, width // max(num_cols, 1))

    for row in range(num_rows):
        for col in range(num_cols):
            y = row * block_height
            x = col * block_width

            if background_color == "bw":
                value = rng.randint(0, 255)
                color = (value, value, value)
            else:
                color = tuple(rng.randint(0, 255) for _ in range(3))

            image[y : min(y + block_height, height), x : min(x + block_width, width)] = color

    return image


def _create_background(
    height: int,
    width: int,
    background_color: str,
    background_type: str,
    rng: random.Random,
    np_rng: np.random.Generator,
    block_size_rng: Tuple[int, int],
) -> np.ndarray:
    background_type_normalized = background_type.lower()
    background_color_normalized = background_color.lower()

    if background_type_normalized == "pixels":
        image = np_rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        if background_color_normalized == "bw":
            gray = image[:, :, 0]
            image = np.repeat(gray[:, :, None], 3, axis=2)
        return image

    if background_type_normalized == "grid":
        return _generate_grid_background(height, width, rng, background_color_normalized, block_size_rng)

    if background_type_normalized == "blocks":
        return _generate_block_background(height, width, rng, background_color_normalized, block_size_rng)

    raise ValueError(
        f"Invalid background_type '{background_type}'. Expected one of ['pixels', 'grid', 'blocks']"
    )


def _wrap_text_to_width(
    text: str,
    font: int,
    font_scale: float,
    thickness: int,
    max_width: int,
) -> Optional[List[str]]:
    max_width = max(max_width, 1)
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current_line: List[str] = []

    for word in words:
        candidate_line = " ".join(current_line + [word])
        width, _ = cv2.getTextSize(candidate_line, font, font_scale, thickness)[0]
        if width <= max_width:
            current_line.append(word)
            continue

        if not current_line:
            return None

        lines.append(" ".join(current_line))
        current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def _prepare_wrapped_lines(
    text: str,
    font: int,
    font_scale: float,
    thickness: int,
    max_width: int,
) -> Optional[List[str]]:
    paragraphs = text.splitlines() or [text]
    lines: List[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        wrapped = _wrap_text_to_width(paragraph, font, font_scale, thickness, max_width)
        if wrapped is None:
            return None
        lines.extend(wrapped)
    return lines


def _render_text_on_image(image: np.ndarray, text: str, augmentation: ImageAugmentation) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    x, y = augmentation.position
    if x >= width or y >= height:
        return None

    max_width = width - x - 10
    if max_width <= 0:
        return None

    lines = _prepare_wrapped_lines(
        text,
        augmentation.font,
        augmentation.font_scale,
        augmentation.thickness,
        max_width,
    )
    if lines is None:
        return None

    _, line_height = cv2.getTextSize("Ag", augmentation.font, augmentation.font_scale, augmentation.thickness)[0]
    line_height = max(line_height + augmentation.thickness * 2 + 4, 1)
    total_height = line_height * len(lines)
    if y + total_height > height - 5:
        return None

    output = image.copy()
    current_y = y
    for line in lines:
        if not line:
            current_y += line_height
            continue
        text_size = cv2.getTextSize(line, augmentation.font, augmentation.font_scale, augmentation.thickness)[0]
        baseline_y = current_y + text_size[1]
        cv2.putText(
            output,
            line,
            (x, baseline_y),
            augmentation.font,
            augmentation.font_scale,
            augmentation.color,
            augmentation.thickness,
            lineType=cv2.LINE_AA,
        )
        current_y += line_height

    return output


def _get_uniform_image_kwargs(
    valid_fonts: Sequence[int],
    rng: random.Random,
    np_rng: np.random.Generator,
    background_color: str,
    background_type: str,
    block_size_rng: Tuple[int, int],
) -> Tuple[np.ndarray, ImageAugmentation]:
    height = rng.randint(240, 600)
    width = rng.randint(240, 600)
    image = _create_background(height, width, background_color, background_type, rng, np_rng, block_size_rng)

    font = rng.choice(valid_fonts)
    font_scale = rng.uniform(0.2, 2.0)
    thickness = 1 if font_scale < 0.8 else rng.randint(1, 3)
    color = tuple(rng.randint(0, 255) for _ in range(3))
    pos_x = rng.randint(0, max(width // 2, 1))
    pos_y = rng.randint(0, max(height // 2, 1))
    augmentation = ImageAugmentation(position=(pos_x, pos_y), font=font, font_scale=font_scale, color=color, thickness=thickness)
    return image, augmentation


def _get_gaussian_image_kwargs(
    rng: random.Random,
    np_rng: np.random.Generator,
    sigma: float,
    background_color: str,
    background_type: str,
    block_size_rng: Tuple[int, int],
) -> Tuple[np.ndarray, ImageAugmentation]:
    def _sample_dim() -> int:
        value = int(np.clip(np_rng.normal(0, sigma), -1, 1) * 300 + 600)
        return max(value, 240)

    height = _sample_dim()
    width = _sample_dim()
    image = _create_background(height, width, background_color, background_type, rng, np_rng, block_size_rng)

    font = cv2.FONT_HERSHEY_SIMPLEX
    raw_color = np.clip(np_rng.normal(0, sigma, size=3), -1, 1)
    color = tuple(int(((component + 1) / 2) * 255) for component in raw_color)
    font_scale = float(np.clip(np_rng.normal(0, sigma) + 1.2, 0.2, 2.0))
    thickness = 1 if font_scale < 0.8 else 2

    pos_x = int((np.clip(np_rng.normal(0, sigma), -1, 1) + 1) * (width // 4))
    pos_y = int((np.clip(np_rng.normal(0, sigma), -1, 1) + 1) * (height // 4))
    pos_x = int(np.clip(pos_x, 0, max(width - 1, 0)))
    pos_y = int(np.clip(pos_y, 0, max(height - 1, 0)))

    augmentation = ImageAugmentation(position=(pos_x, pos_y), font=font, font_scale=font_scale, color=color, thickness=thickness)
    return image, augmentation


def process_image_with_text(
    text: str,
    valid_fonts: Sequence[int],
    seed: int,
    sigma: Optional[float] = None,
    background_color: str = "color",
    background_type: str = "pixels",
    block_size_rng: Tuple[int, int] = (3, 8),
    long_request: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[ImageAugmentation]]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    attempts = 0
    max_attempts = 25

    background_color = background_color.lower()
    background_type = background_type.lower()

    while attempts < max_attempts:
        if sigma is not None:
            image, augmentation = _get_gaussian_image_kwargs(
                rng,
                np_rng,
                sigma,
                background_color,
                background_type,
                block_size_rng,
            )
        else:
            image, augmentation = _get_uniform_image_kwargs(
                valid_fonts,
                rng,
                np_rng,
                background_color,
                background_type,
                block_size_rng,
            )

        if long_request:
            augmentation.font_scale = min(augmentation.font_scale, 0.35)
            augmentation.thickness = 1
            augmentation.position = (10, 20)

        rendered = _render_text_on_image(image, text, augmentation)
        if rendered is not None:
            return rendered, augmentation

        attempts += 1

    return None, None


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

        self.generate_images = bool(kwargs.get("generate_images", False))
        image_sigma = kwargs.get("image_sigma")
        self.image_sigma: Optional[float] = None if image_sigma in (None, "None") else float(image_sigma)
        self.image_background_color = str(kwargs.get("image_background_color", "color")).lower()
        self.image_background_type = str(kwargs.get("image_background_type", "pixels")).lower()
        self.image_block_size_rng = self._parse_block_size_rng(kwargs.get("image_block_size_rng", (3, 8)))
        self.image_seed_offset = int(kwargs.get("image_seed_offset", 0))
        image_long_request = kwargs.get("image_long_request")
        self.image_long_request = bool(self.prefix) if image_long_request is None else bool(image_long_request)
        self._image_fonts = self._resolve_image_fonts(kwargs.get("image_fonts"))
        self._validate_image_options()

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

            generated_image_filename, image_aug = self._maybe_generate_image(
                best_candidate["prompt"],
                save_dir_path,
                behavior_id,
                best_candidate["seed"],
            )
            if image_aug is not None:
                best_candidate["image_augmentation"] = image_aug
            if generated_image_filename is not None:
                best_candidate["image_filename"] = generated_image_filename

            image_reference = self._extract_image_reference(behavior_dict)
            if generated_image_filename is not None:
                test_cases[behavior_id] = [generated_image_filename, best_candidate["prompt"]]
            elif image_reference is not None:
                if not self.generate_images:
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
    def _parse_block_size_rng(raw_value: Any) -> Tuple[int, int]:
        if isinstance(raw_value, str):
            parts = [part.strip() for part in raw_value.split(",") if part.strip()]
            if len(parts) != 2:
                raise ValueError(
                    "image_block_size_rng must contain two comma-separated integers when provided as a string"
                )
            values = (int(parts[0]), int(parts[1]))
        elif isinstance(raw_value, Sequence):
            seq = list(raw_value)
            if len(seq) != 2:
                raise ValueError("image_block_size_rng must contain exactly two values")
            values = (int(seq[0]), int(seq[1]))
        else:
            raise ValueError("image_block_size_rng must be a tuple/list of two integers or a comma-separated string")

        low, high = values
        if low <= 0 or high <= 0:
            raise ValueError("image_block_size_rng values must be positive")
        if low > high:
            low, high = high, low
        return (low, high)

    @staticmethod
    def _resolve_image_fonts(raw_fonts: Optional[Sequence[Any]]) -> Tuple[int, ...]:
        if raw_fonts in (None, "None"):
            return _CV2_FONTS

        resolved: List[int] = []
        for font in raw_fonts:
            if isinstance(font, int):
                resolved.append(font)
                continue

            if isinstance(font, str):
                name = font.strip().upper()
                if not name.startswith("FONT_"):
                    name = f"FONT_{name}"
                if not hasattr(cv2, name):
                    raise ValueError(f"Unknown OpenCV font identifier: {font}")
                resolved.append(int(getattr(cv2, name)))
                continue

            raise TypeError("image_fonts entries must be integers or OpenCV font name strings")

        if not resolved:
            raise ValueError("image_fonts must include at least one font")
        return tuple(resolved)

    def _validate_image_options(self) -> None:
        if self.image_background_color not in {"color", "bw"}:
            raise ValueError("image_background_color must be either 'color' or 'bw'")
        if self.image_background_type not in {"pixels", "grid", "blocks"}:
            raise ValueError("image_background_type must be one of ['pixels', 'grid', 'blocks']")
        if not self.image_block_size_rng:
            raise ValueError("image_block_size_rng cannot be empty")

    def _maybe_generate_image(
        self,
        prompt: str,
        output_dir: Path,
        behavior_id: str,
        seed: int,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not self.generate_images:
            return None, None

        actual_seed = seed + self.image_seed_offset
        image, augmentation = process_image_with_text(
            text=prompt,
            valid_fonts=self._image_fonts,
            seed=actual_seed,
            sigma=self.image_sigma,
            background_color=self.image_background_color,
            background_type=self.image_background_type,
            block_size_rng=self.image_block_size_rng,
            long_request=self.image_long_request,
        )

        if image is None or augmentation is None:
            return None, None

        image_dir = output_dir / "generated_images"
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{behavior_id}_{actual_seed}.png"
        image_path = image_dir / filename
        success = cv2.imwrite(str(image_path), image)
        if not success:
            return None, None

        return filename, augmentation.dict()

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
