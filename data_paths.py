"""Data-root resolution and compatibility for the archived ai-resilience inputs."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# Historical completion records can still refer to the pre-archive locations.
_ARCHIVED_PATHS = {
    'behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv':
        'original/advbench/advbench_behaviors.csv',
    'behavior_datasets/extra_behavior_datasets/tdc2023_test_phase_behaviors.csv':
        'original/TDC2023/tdc2023_test_phase_behaviors.csv',
    'behavior_datasets/extra_behavior_datasets/adv_training_behaviors.csv':
        'original/adv_training/adv_training_behaviors.csv',
    'behavior_datasets': 'original/HarmBench/behaviors',
    'multimodal_behavior_images': 'original/HarmBench/images',
    'advbench': 'original/advbench',
    'optimizer_targets': 'original/optimizer_targets',
    'classifier_val_sets': 'original/classifier_val_sets',
    'copyright_classifier_hashes': 'original/copyright_classifier_hashes',
    'clean.jpeg': 'original/clean.jpeg',
    'derogatory_corpus.csv': 'original/derogatory_corpus.csv',
}


def get_data_root():
    root = Path(os.path.expandvars(os.environ.get('VLM_JAILBREAK_DATA_ROOT', str(REPO_ROOT / 'data')))).expanduser()
    return root if root.is_absolute() else REPO_ROOT / root


def resolve_path(path):
    """Resolve data/... under the selected root; preserve external absolute paths."""
    expanded = Path(os.path.expandvars(os.fspath(path))).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    if expanded.parts and expanded.parts[0] == 'data':
        relative = Path(*expanded.parts[1:]).as_posix()
        for old, new in _ARCHIVED_PATHS.items():
            if relative == old or relative.startswith(old + '/'):
                relative = new + relative[len(old):]
                break
        return str(get_data_root() / relative)
    return str(REPO_ROOT / expanded)
