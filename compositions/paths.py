"""Use the existing data root and portable-path contract."""
from pathlib import Path
import hashlib

from data_paths import REPO_ROOT, resolve_path
from benchmark_datasets import portable_path


def file_sha256(path):
    path = Path(resolve_path(path))
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()
