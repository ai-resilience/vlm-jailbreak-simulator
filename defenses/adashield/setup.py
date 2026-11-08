"""Import locally refined AdaShield pools and their images without training or API calls."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import yaml

from data_paths import REPO_ROOT, resolve_path
from .defense import AdaShieldDefense, OFFICIAL_CLIP_SHA256, _sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference_root', required=True, help='Reference simulator checkout with runs/adashield_official_repro')
    parser.add_argument('--clip_checkpoint', default=str(Path.home() / '.cache/clip/ViT-B-32.pt'))
    args = parser.parse_args(argv)
    source_root = Path(args.reference_root).expanduser().resolve()
    source_run = source_root / 'runs/adashield_official_repro'
    destination = Path(resolve_path('data/method_assets/AdaShield'))
    checkpoint = Path(args.clip_checkpoint).expanduser()
    if not checkpoint.is_file() or _sha256(checkpoint) != OFFICIAL_CLIP_SHA256:
        raise ValueError('Provide the original OpenAI CLIP ViT-B/32 checkpoint via --clip_checkpoint')
    checkpoint_out = destination / 'checkpoints/ViT-B-32.pt'
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_out.exists():
        if _sha256(checkpoint_out) != OFFICIAL_CLIP_SHA256:
            raise ValueError(f'Existing CLIP checkpoint differs: {checkpoint_out}')
    else:
        shutil.copyfile(checkpoint, checkpoint_out)
    if (source_run / 'README.md').is_file():
        shutil.copyfile(source_run / 'README.md', destination / 'SOURCE_RUN_README.md')
    summaries = {}
    for source_name, name in (('figstep', 'figstep'), ('qr', 'query_related')):
        tables = sorted((source_run / source_name / 'tables').glob('*/final_table.csv'))
        if not tables:
            raise FileNotFoundError(f'No local refinement tables found: {source_run / source_name}')
        profile = destination / name
        manifest = {'schema_version': 1,
            'provenance': {'kind': 'local_reference_refinement', 'training_target': 'liuhaotian/llava-v1.5-13b',
                           'source_run': str(source_run / source_name), 'official_released_pool': False},
            'tables': [], 'images': {}, 'raw_rows': 0, 'retained_rows': 0}
        scenarios = []
        for table in tables:
            scene = table.parent.name
            scenarios.append(scene)
            raw = profile / 'source_tables' / scene / 'final_table.csv'
            portable = profile / 'tables' / scene / 'wandb/local-reference/files/final_table.csv'
            raw.parent.mkdir(parents=True, exist_ok=True)
            portable.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(table, raw)
            with table.open(encoding='utf-8-sig', newline='') as source:
                reader = csv.DictReader(source)
                fields = reader.fieldnames
                rows = list(reader)
            for row in rows:
                if row.get('image'):
                    image = Path(row['image']).expanduser()
                    if not image.is_absolute():
                        image = source_root / image
                    if not image.is_file():
                        raise FileNotFoundError(f'AdaShield refinement image missing: {image}')
                    image_hash = _sha256(image)
                    relative = Path('images') / f'{image_hash}{image.suffix.lower()}'
                    image_out = profile / relative
                    image_out.parent.mkdir(parents=True, exist_ok=True)
                    if image_out.exists():
                        if _sha256(image_out) != image_hash:
                            raise ValueError(f'Existing AdaShield image differs: {image_out}')
                    else:
                        shutil.copyfile(image, image_out)
                    manifest['images'][str(relative)] = {'source': str(image), 'sha256': image_hash}
                    row['image'] = str(relative)
                retained = AdaShieldDefense._successful(row.get('final_judge_scores')) and bool(row.get('defense_prompt_list', '').strip())
                if source_name == 'figstep' and row.get('defense_improv_list') == "['rephrase prompt']":
                    retained = False
                manifest['retained_rows'] += int(retained)
            with portable.open('w', encoding='utf-8', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            manifest['raw_rows'] += len(rows)
            manifest['tables'].append({'scenario': scene, 'source': str(table),
                'original_copy': str(raw.relative_to(profile)), 'original_sha256': _sha256(table),
                'portable_table': str(portable.relative_to(profile)), 'portable_sha256': _sha256(portable)})
        (profile / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        summaries[name] = {'tables': len(tables), 'rows': manifest['raw_rows'],
                           'retained_rows': manifest['retained_rows'], 'images': len(manifest['images'])}
        config = {'defense_type': 'AdaShield-A', 'experiment_profile': 'local_refined_pool_transfer',
                  'prompt_placement': 'sandwich', 'retrieval_type': 'sample-wise', 'clip_model': 'ViT-B/32',
                  'clip_checkpoint_path': 'data/method_assets/AdaShield/checkpoints/ViT-B-32.pt',
                  'table_dir': f'data/method_assets/AdaShield/{name}/tables',
                  'wandb_subdir': 'local-reference', 'scenarios': scenarios,
                  'pool_image_root': f'data/method_assets/AdaShield/{name}',
                  'pool_manifest_path': f'data/method_assets/AdaShield/{name}/manifest.json',
                  'filter_rephrase_prompt': source_name == 'figstep', 'seed': 0}
        # Write a data-side config for portable imports; repository configs are maintained separately.
        (profile / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({'status': 'imported', 'destination': str(destination), 'profiles': summaries}, indent=2))


if __name__ == '__main__':
    main()
