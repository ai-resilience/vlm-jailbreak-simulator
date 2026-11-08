"""Bridge existing JSON test cases to the shared text/image target interface."""

from pathlib import Path

from .registry import default_revision, resolve_model_id
from .targets import get_target_factory, load_target, validate_image_count
from .targets.paths import resolve_path


def is_single_case(value):
    if isinstance(value, (str, dict)):
        return True
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[1], str)
            and (value[0] is None or isinstance(value[0], (str, list, tuple))))


def case_groups(data):
    if not isinstance(data, dict):
        raise ValueError("Test cases must be a JSON object keyed by BehaviorID")
    groups = {}
    for behavior_id, value in data.items():
        if is_single_case(value):
            groups[behavior_id] = [value]
        elif isinstance(value, list) and value and all(is_single_case(v) for v in value):
            groups[behavior_id] = value
        else:
            raise ValueError(f"Invalid test case(s) for {behavior_id!r}")
    return groups


def prepare_case(case, image_dir):
    if isinstance(case, str):
        return case, []
    if isinstance(case, dict):
        text, references = case.get("text", case.get("prompt")), case.get("images", [])
    elif is_single_case(case):
        references, text = case
    else:
        raise ValueError("Expected text, [image(s), prompt], or {text, images}")
    if not isinstance(text, str):
        raise ValueError("The test-case prompt must be a string")
    if references is None or references == "":
        references = []
    elif isinstance(references, str):
        references = [references]
    if not isinstance(references, (list, tuple)) or not all(isinstance(p, str) for p in references):
        raise ValueError("Image references must be a filename or list of filenames")
    paths = []
    for reference in references:
        path = Path(reference).expanduser()
        candidates = ([path] if path.is_absolute() else
                      [Path(image_dir) / path, Path(image_dir).parent / path,
                       Path(resolve_path(path))])
        resolved = next((p.resolve() for p in candidates if p.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(f"Test-case image not found: {reference!r}")
        paths.append(str(resolved))
    return text, paths


def load_generation_function(model_config, max_new_tokens, test_cases_path):
    name = model_config["model_name_or_path"]
    get_target_factory(name)
    image_dir = Path(test_cases_path).resolve().parent / "images"
    target = None

    def generate(test_cases, progress_bar=False):
        nonlocal target
        prepared = [prepare_case(case, image_dir) for case in test_cases]
        for _, images in prepared:
            validate_image_count(name, len(images))
        if not prepared:
            return []
        if target is None:
            target = load_target(name, max_new_tokens=max_new_tokens)
            options = model_config.get("generation", {})
            for key, value in options.items():
                if key not in {"do_sample", "temperature", "top_p", "retry_limit"}:
                    raise ValueError(f"Unsupported generation option: {key!r}")
                if not hasattr(target, key):
                    raise ValueError(f"{name} does not support generation option {key!r}")
                setattr(target, key, value)
        if progress_bar:
            from tqdm import tqdm
            responses = [target.generate(text, images) for text, images in tqdm(prepared)]
        else:
            responses = target.generate_batch(prepared)
        if len(responses) != len(prepared) or not all(isinstance(r, str) for r in responses):
            raise RuntimeError("Target must return one string response per test case")
        return responses

    return generate


def model_metadata(model_config, max_new_tokens):
    name = model_config["model_name_or_path"]
    return {"model_id": resolve_model_id(name), "revision": default_revision(name),
            "max_new_tokens": max_new_tokens,
            "generation": model_config.get("generation", {})}
