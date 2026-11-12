"""CLI for SED-L zero-shot evaluation on R2S100K train."""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import yaml
from .core import ROOT, prepare, prepare_single_image, prepare_image_manifest, prepare_labels_for_manifest, evaluate, read_json, write_json, validate_manifests, sha256, materialize_vocabulary
from .inference import infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'benchmark', 'infer', 'evaluate', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--dataset-root', type=Path)
        p.add_argument('--checkpoint', type=Path)
        p.add_argument('--output', type=Path)
        p.add_argument('--device')
        p.add_argument('--workers', type=int)
        p.add_argument('--max-images', type=int)
        p.add_argument('--image', type=Path, help='Use one image from the train split for a smoke test')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for field in ('dataset_root', 'checkpoint', 'output', 'device', 'workers'):
        value = getattr(args, field)
        if value is not None: cfg[field] = value
    root = Path(cfg['dataset_root']); output = Path(cfg['output']); checkpoint = Path(cfg['checkpoint'])
    vocabulary = materialize_vocabulary(output) if args.command in ('infer', 'benchmark', 'run') else None
    images = output / 'manifests' / 'train_images.json'; labels = output / 'manifests' / 'train_labels.json'
    if args.command in ('prepare', 'run'):
        report = prepare_single_image(root, output, args.image) if args.image else prepare(root, output)
        print(json.dumps(report, indent=2))
    if args.command == 'prepare': return
    if args.command == 'infer':
        if args.image:
            rows = prepare_image_manifest(root, output, args.image)
            print(f'[infer] Created one-image manifest for {rows[0]["image_id"]}; labels are not needed.', flush=True)
        elif not images.is_file():
            rows = prepare_image_manifest(root, output)
            print(f'[infer] Created image-only manifest: {len(rows)} images; labels are not needed.', flush=True)
        else:
            rows = read_json(images)
            if not isinstance(rows, list) or not rows:
                raise ValueError(f'Image manifest is empty or invalid: {images}')
        result = infer(images, checkpoint, vocabulary, output, cfg['device'], cfg['workers'], max_images=args.max_images)
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.command == 'evaluate' and args.image:
        print(json.dumps(prepare_single_image(root, output, args.image), indent=2), flush=True)
    if args.command == 'evaluate' and images.is_file() and not labels.is_file():
        image_rows = read_json(images)
        print(f'[evaluate] Preparing labels for {len(image_rows)} manifest image(s).', flush=True)
        label_rows = prepare_labels_for_manifest(root, output, image_rows)
    if args.command in ('evaluate', 'benchmark') and (not images.is_file() or not labels.is_file()):
        print('[prepare] Required image/label manifests are missing; preparing them from dataset.', flush=True)
        print(json.dumps(prepare(root, output), indent=2), flush=True)
    rows, label_rows = validate_manifests(images, labels)
    if args.command == 'evaluate':
        eval_images, eval_labels = images, labels
        if args.max_images is not None:
            count = min(args.max_images, len(rows))
            eval_dir = output / 'manifests'
            eval_images = eval_dir / 'evaluate_subset_images.json'
            eval_labels = eval_dir / 'evaluate_subset_labels.json'
            write_json(eval_images, rows[:count]); write_json(eval_labels, label_rows[:count])
        eval_rows = read_json(eval_images)
        missing_predictions = [row['image_id'] for row in eval_rows if not (output / 'predictions' / f"{row['image_id']}.png").is_file()]
        if missing_predictions:
            missing = missing_predictions[0]
            hint = f".venv/bin/python -m sed_r2s100k infer --image '{eval_rows[0]['image_path']}' --output '{output}' --workers 0 --device {cfg['device']}" if len(eval_rows) == 1 else f".venv/bin/python -m sed_r2s100k infer --output '{output}'"
            raise SystemExit(f'Evaluation needs predictions, but {len(missing_predictions)} are missing (first: {missing}). Run inference first:\n{hint}')
        result = evaluate(eval_images, eval_labels, output / 'predictions', output / 'evaluation', cfg['seed'], cfg.get('ignore_id'))
        print(json.dumps({k: v for k, v in result.items() if k not in ('per_class', 'confusion_matrix')}, indent=2)); return
    if args.command in ('benchmark', 'run'):
        begin = time.perf_counter()
        checkpoint_hash = sha256(checkpoint)
        manifest_hash = sha256(images)
        vocabulary_hash = sha256(vocabulary)
        prior_path = output / 'benchmark.json'
        prior = read_json(prior_path) if prior_path.exists() else None
        reuse = bool(prior and prior.get('checkpoint_sha256') == checkpoint_hash and prior.get('manifest_sha256') == manifest_hash and prior.get('vocabulary_sha256') == vocabulary_hash and prior.get('device') == cfg['device'] and prior.get('gpu') and prior.get('within_budget'))
        if reuse:
            report = prior
            print('[benchmark] Reusing matching completed pilot', flush=True)
        else:
            n = len(rows); count = min(int(cfg['benchmark_images']), n); warm = min(int(cfg['warmup_images']), n)
            indices = sorted({round(i*(n-1)/max(1,count-1)) for i in range(count)})
            warm_indices = indices[:warm]
            settings = dict(manifest=images, checkpoint=checkpoint, vocabulary=vocabulary, device=cfg['device'], workers=cfg['workers'])
            pilot_dir = output / f'pilot_{int(time.time())}'
            infer(**settings, output=pilot_dir/'warmup', indices=warm_indices)
            measured = infer(**settings, output=pilot_dir/'benchmark', indices=indices)
            pilot_eval_start = time.perf_counter()
            from PIL import Image
            from .core import decode_mask, confusion
            import numpy as np
            label_rows = read_json(labels)
            cm = np.zeros((15,15), np.int64)
            for index in indices:
                with Image.open(label_rows[index]['gt_path']) as gt, Image.open(pilot_dir/'benchmark'/'predictions'/f"{rows[index]['image_id']}.png") as pred:
                    cm += confusion(np.asarray(pred), decode_mask(gt), cfg.get('ignore_id'))
            eval_per_image = (time.perf_counter()-pilot_eval_start)/len(indices)
            pilot_elapsed = time.perf_counter()-begin
            per_image = measured['throughput_seconds_per_image'] + eval_per_image
            estimate = pilot_elapsed + 1.2*n*per_image
            report = {'images_total': n, 'warmup_count': warm, 'benchmark_count': len(indices), 'pilot_elapsed_seconds': pilot_elapsed, 'inference_mean_seconds': measured['latency_mean_seconds'], 'inference_p95_seconds': measured['latency_p95_seconds'], 'evaluation_mean_seconds': eval_per_image, 'peak_vram_bytes': measured['peak_vram_bytes'], 'estimate_with_20pct_margin_seconds': estimate, 'budget_seconds': cfg['budget_seconds'], 'within_budget': estimate <= cfg['budget_seconds'], 'gpu': measured['gpu'], 'checkpoint_sha256': checkpoint_hash, 'manifest_sha256': manifest_hash, 'vocabulary_sha256': vocabulary_hash, 'device': cfg['device']}
            write_json(prior_path, report)
        print(json.dumps(report, indent=2), flush=True)
        if args.command == 'benchmark': return
        if not report['within_budget']: raise SystemExit('Pilot forecasts >3 h; full run stopped. See benchmark.json')
        deadline = begin + cfg['budget_seconds']
    else: deadline = None
    if args.command in ('infer', 'run'):
        result = infer(images, checkpoint, vocabulary, output, cfg['device'], cfg['workers'], max_images=args.max_images, deadline=deadline)
        print(json.dumps(result, indent=2), flush=True)
        if args.command == 'infer': return
        if result['status'] != 'complete': raise SystemExit('Budget expired; incomplete run can be resumed')
    if args.command == 'run':
        result = evaluate(images, labels, output/'predictions', output/'evaluation', cfg['seed'], cfg.get('ignore_id'))
        if time.perf_counter() > deadline:
            write_json(output/'evaluation'/'budget_status.json', {'met': False, 'reason': 'Evaluation completed after deadline'})
            raise SystemExit('Evaluation completed, but 3-hour budget exceeded')
        write_json(output/'evaluation'/'budget_status.json', {'met': True, 'elapsed_seconds': time.perf_counter()-begin})
        print(json.dumps({k: v for k, v in result.items() if k not in ('per_class', 'confusion_matrix')}, indent=2))

if __name__ == '__main__': main()
