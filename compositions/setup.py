"""Copy and verify the existing font, JPS transfer assets and optional CS-DJ cache."""
import argparse
import json
from pathlib import Path
import shutil

import yaml

from data_paths import resolve_path
from .five_methods import digest, json_hash, atomic_json

FONT_FILES = {
    'NotoSansCJKsc-Bold.otf': 'b5f0d1a190a7f9b43c310a8850630af12553df32c4c050543f9059732d9b4c0a',
    'NotoSansCJK-OFL.txt': '6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2',
}


def copy_verified(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if digest(source) != expected:
        raise ValueError(f'Source asset hash mismatch: {source}')
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f'Existing composition asset changed: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.copying')
    try:
        shutil.copyfile(source, temporary)
        if digest(temporary) != expected:
            raise ValueError(f'Copied asset hash mismatch: {destination}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference_root', required=True, help='Existing original simulator checkout')
    parser.add_argument('--cs_cache_run', help='Completed original five-method run; optional verified cache reuse')
    args = parser.parse_args(argv)
    source = Path(args.reference_root).expanduser().resolve()
    config = yaml.safe_load((source / 'configs/compositions/five_methods_3000.yaml').read_text())
    output = Path(resolve_path('data/method_assets/compositions'))
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in FONT_FILES.items():
        copy_verified(source / 'data/method_assets/fonts' / name,
                      Path(resolve_path('data/method_assets/fonts')) / name, expected)
    jps = config['jps']
    run = source / jps['source_run']
    manifest_path = run / 'result_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name in ('image', 'steer', 'training_csv'):
        relative = jps[name]
        copy_verified(run / relative, output / 'jps' / relative, manifest[relative])
    copy_verified(manifest_path, output / 'jps/result_manifest.json', digest(manifest_path))
    cache_count = 0
    if args.cs_cache_run:
        cached = Path(args.cs_cache_run).expanduser().resolve()
        identity = json.loads((cached / 'run_identity.json').read_text())
        completed = json.loads((cached / 'full/manifest.json').read_text())
        if (completed['status'] != 'complete' or completed['identity'] != identity['identity']
                or digest(cached / 'full/test_cases.jsonl') != completed['test_cases_sha256']
                or digest(cached / 'parents.json') != completed['parents_sha256']):
            raise ValueError('Original CS-DJ cache needs an intact completed run')
        parents = {b['behavior_id']: b for b in json.loads((cached / 'parents.json').read_text())}
        entries = {}
        for line in (cached / 'full/test_cases.jsonl').read_text().splitlines():
            case = json.loads(line)
            data = case.get('meta', {}).get('details', {}).get('cs_dj')
            if data is not None:
                previous = entries.get(case['behavior_id'])
                if previous is not None and previous != data:
                    raise ValueError('Original CS-DJ results differ for the same parent')
                entries[case['behavior_id']] = data
        checked_images, imported_hashes = {}, {}
        for behavior_id, data in sorted(entries.items()):
            path = cached / '_cache/cs_dj' / (behavior_id + '.json')
            parent = parents[behavior_id]
            if data['key'] != json_hash([identity['identity'], parent, data['config']]):
                raise ValueError(f'Original CS-DJ cache identity mismatch: {path}')
            if len(data['subquestions']) != 3 or len(data['contrast_images']) != 9:
                raise ValueError(f'Invalid original CS-DJ decomposition/retrieval: {path}')
            for image, expected in zip(data['contrast_images'], data['contrast_hashes']):
                original = Path(image) if Path(image).is_absolute() else source / image
                local = Path(resolve_path(image))
                if checked_images.setdefault(str(local), digest(local) if str(local) not in checked_images else checked_images[str(local)]) != expected:
                    raise ValueError(f'CS-DJ contrast asset differs: {image}')
            data['source_identity'] = identity['identity']
            data['source_cache_sha256'] = json_hash(data)
            data['source_instruction_sha256'] = json_hash(parent['instruction'])
            data['record_sha256'] = json_hash(parent)
            destination = output / 'cs_dj' / path.name
            if destination.exists() and json.loads(destination.read_text()) != data:
                raise ValueError(f'Existing imported CS-DJ cache differs: {destination}')
            atomic_json(destination, data)
            imported_hashes[path.name] = digest(destination)
            cache_count += 1
        atomic_json(output / 'cs_dj/manifest.json', {'files': imported_hashes, 'source_manifest_sha256': digest(cached / 'full/manifest.json')})
    atomic_json(output / 'manifest.json', {'source_root': str(source),
        'jps_source_manifest_sha256': digest(manifest_path), 'font_files': FONT_FILES,
        'cs_cache_records': cache_count, 'source_config_sha256': digest(source / 'configs/compositions/five_methods_3000.yaml')})
    print(json.dumps({'status': 'prepared', 'assets': str(output), 'cs_cache_records': cache_count}))


if __name__ == '__main__':
    main()
