"""Install the verified CIDER diffusion checkpoint under the configured data root."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

from data_paths import resolve_path
from .defense import DEFAULT_DENOISER_PATH, OFFICIAL_DENOISER_MD5, OFFICIAL_DENOISER_SIZE

URL = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'


def verify(path):
    if path.stat().st_size != OFFICIAL_DENOISER_SIZE:
        raise ValueError(f'CIDER checkpoint size mismatch: {path}')
    md5 = hashlib.md5()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            md5.update(block)
    if md5.hexdigest() != OFFICIAL_DENOISER_MD5:
        raise ValueError(f'CIDER checkpoint MD5 mismatch: {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='Copy a local checkpoint instead of downloading')
    parser.add_argument('--destination', default=DEFAULT_DENOISER_PATH)
    args = parser.parse_args(argv)
    destination = Path(resolve_path(args.destination))
    if destination.is_file():
        verify(destination)
        print(f'CIDER checkpoint verified: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.cider-', suffix='.pt', dir=destination.parent)
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
    print(f'CIDER checkpoint installed and verified: {destination}')


if __name__ == '__main__':
    main()
