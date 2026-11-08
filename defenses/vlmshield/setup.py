"""Install the hash-pinned released VLMShield classifier."""
import argparse
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

from data_paths import resolve_path
from .defense import (DEFAULT_CLASSIFIER_PATH, OFFICIAL_CLASSIFIER_NAME, OFFICIAL_CLASSIFIER_SHA256,
                      OFFICIAL_CLASSIFIER_SIZE, OFFICIAL_COMMIT, _sha256_file)

URL = (f'https://raw.githubusercontent.com/pgqihere/VLMShield/{OFFICIAL_COMMIT}/'
       f'VLMShield_script/model/{OFFICIAL_CLASSIFIER_NAME}')


def verify(path):
    if path.stat().st_size != OFFICIAL_CLASSIFIER_SIZE:
        raise ValueError(f'VLMShield classifier size mismatch: {path}')
    if _sha256_file(path) != OFFICIAL_CLASSIFIER_SHA256:
        raise ValueError(f'VLMShield classifier SHA-256 mismatch: {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='Copy a local released classifier instead of downloading')
    parser.add_argument('--destination', default=str(DEFAULT_CLASSIFIER_PATH))
    args = parser.parse_args(argv)
    destination = Path(resolve_path(args.destination))
    if destination.is_file():
        verify(destination)
        print(f'VLMShield classifier verified: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.vlmshield-', suffix='.pt', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        if args.source:
            shutil.copyfile(Path(args.source).expanduser(), temporary)
        else:
            urllib.request.urlretrieve(URL, temporary)
        verify(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'VLMShield classifier installed and verified: {destination}')


if __name__ == '__main__':
    main()
