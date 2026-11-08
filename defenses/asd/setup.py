"""Install and verify AsD's released protective patch."""
import argparse
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

from data_paths import resolve_path
from .defense import DEFAULT_PATCH_PATH, OFFICIAL_COMMIT, OFFICIAL_PATCH_NAME, OFFICIAL_PATCH_SHA256, _sha256_file

URL = f'https://raw.githubusercontent.com/AngelAlita/AsD/{OFFICIAL_COMMIT}/images/{OFFICIAL_PATCH_NAME}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='Copy a local released patch instead of downloading')
    parser.add_argument('--destination', default=str(DEFAULT_PATCH_PATH))
    args = parser.parse_args(argv)
    destination = Path(resolve_path(args.destination))
    if destination.is_file():
        if _sha256_file(destination) != OFFICIAL_PATCH_SHA256:
            raise ValueError(f'Existing AsD patch SHA-256 mismatch: {destination}')
        print(f'AsD patch verified: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.asd-', suffix='.png', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        if args.source:
            shutil.copyfile(Path(args.source).expanduser(), temporary)
        else:
            urllib.request.urlretrieve(URL, temporary)
        if _sha256_file(temporary) != OFFICIAL_PATCH_SHA256:
            raise ValueError('AsD patch SHA-256 mismatch')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'AsD patch installed and verified: {destination}')


if __name__ == '__main__':
    main()
