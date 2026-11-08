"""Lazy defense registry; optional assets are checked only when selected."""
import importlib
from pathlib import Path

import yaml

from data_paths import REPO_ROOT, resolve_path

DEFENSE_REGISTRY = {
}


def load_defense(defense_name, config_dir='configs/defense_configs', config_path=None):
    if defense_name not in DEFENSE_REGISTRY:
        raise ValueError(f'Unknown defense {defense_name!r}; choose {sorted(DEFENSE_REGISTRY)}')
    path = Path(resolve_path(config_path or str(Path(config_dir) / f'{defense_name}.yaml')))
    with path.open(encoding='utf-8') as source:
        config = yaml.safe_load(source) or {}
    if not isinstance(config, dict):
        raise ValueError(f'Defense config must be a YAML mapping: {path}')
    if config.get('temp_dir'):
        config['temp_dir'] = resolve_path(config['temp_dir'])
    module, name = DEFENSE_REGISTRY[defense_name]
    defense = getattr(importlib.import_module(module), name)(config)
    defense.config_path = path.resolve()
    return defense
