"""Data preparation, score aggregation and reproducibility helpers."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .taxonomy import CLASSES, BACKGROUND_RGB, IMAGE_SUFFIXES, model_vocabulary

ROOT = Path(__file__).resolve().parents[1]
NAMES = [CLASSES[i][0] for i in range(15)]
PALETTE = np.asarray([CLASSES[i][2] for i in range(15)], dtype=np.uint8)
LOOKUP = np.full(1 << 24, 255, dtype=np.uint8)
KEYS = (PALETTE[:, 0].astype(np.uint32) << 16) | (PALETTE[:, 1].astype(np.uint32) << 8) | PALETTE[:, 2].astype(np.uint32)
LOOKUP[KEYS] = np.arange(15, dtype=np.uint8)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text())

def materialize_vocabulary(output):
    """Freeze the taxonomy prompts used by this run beside its predictions."""
    path = Path(output) / 'vocabulary.json'
    expected = model_vocabulary()
    if path.is_file():
        if read_json(path) != expected:
            raise RuntimeError(f'Prompt taxonomy changed for {output}; choose a new output directory')
    else:
        write_json(path, expected)
    return path

def decode_mask(rgb, ignore_id=None):
    rgb = np.asarray(rgb.convert('RGB') if isinstance(rgb, Image.Image) else rgb)
    packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2].astype(np.uint32)
    result = LOOKUP[packed]
    if np.any(result == 255):
        bad = np.unique(packed[result == 255])[:8]
        colors = [(int(v >> 16), int((v >> 8) & 255), int(v & 255)) for v in bad]
        raise ValueError(f'Unknown GT colors: {colors}')
    return result

def prepare(dataset_root, output):
    root = Path(dataset_root).resolve(); images_dir = root / 'train'; labels_dir = root / 'train_labels'
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f'Missing train or train_labels under {root}')
    images = sorted(p for p in images_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images: raise ValueError('No train images')
    if len({p.name for p in images}) != len(images): raise ValueError('Duplicate image basename')
    labels = {}
    for p in labels_dir.rglob('*___fuse.png'):
        key = p.name.removesuffix('___fuse.png')
        if key in labels: raise ValueError(f'Duplicate label: {key}')
        labels[key] = p
    im_manifest, gt_manifest = [], []
    for index, image in enumerate(images, 1):
        label = labels.get(image.name)
        if label is None: raise FileNotFoundError(f'Missing GT for {image.name}')
        with Image.open(image) as im, Image.open(label) as gt:
            if im.size != gt.size: raise ValueError(f'Size mismatch: {image.name}: {im.size} != {gt.size}')
            # Unique colors are cheap to check without materializing a full image array.
            colors = gt.convert('RGB').getcolors(maxcolors=1 << 16)
            if colors is None: raise ValueError(f'Too many colors: {label}')
            unknown = [color for _, color in colors if tuple(color) not in map(tuple, PALETTE)]
            if unknown: raise ValueError(f'Unknown GT colors in {label}: {unknown[:8]}')
        im_manifest.append({'image_id': image.name, 'image_path': str(image.resolve())})
        gt_manifest.append({'image_id': image.name, 'gt_path': str(label.resolve())})
        if index % 1000 == 0: print(f'[prepare] {index}/{len(images)}', flush=True)
    extra = set(labels) - {p.name for p in images}
    if extra: raise ValueError(f'{len(extra)} unpaired labels, first: {sorted(extra)[0]}')
    out = Path(output) / 'manifests'; out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'train_images.json', im_manifest)
    write_json(out / 'train_labels.json', gt_manifest)
    report = {'images': len(images), 'labels': len(gt_manifest), 'class_count': 15, 'dataset_root': str(root)}
    write_json(out / 'audit.json', report)
    return report


def _label_for_image(root, image):
    labels_dir = Path(root) / 'train_labels'
    candidates = [labels_dir / f'{image.name}___fuse.png', labels_dir / 'Train-Labels' / f'{image.name}___fuse.png']
    label = next((path for path in candidates if path.is_file()), None)
    if label is None and labels_dir.is_dir():
        matches = list(labels_dir.rglob(f'{image.name}___fuse.png'))
        if len(matches) > 1: raise ValueError(f'Duplicate label for {image.name}')
        label = matches[0] if matches else None
    if label is None: raise FileNotFoundError(f'Missing GT for {image.name} under {labels_dir}')
    return label


def _validate_pair(image, label):
    with Image.open(image) as im, Image.open(label) as gt:
        if im.size != gt.size: raise ValueError(f'Size mismatch: {image.name}: {im.size} != {gt.size}')
        colors = gt.convert('RGB').getcolors(maxcolors=1 << 16)
        if colors is None: raise ValueError(f'Too many colors: {label}')
        allowed = {tuple(color) for color in PALETTE}
        unknown = [color for _, color in colors if tuple(color) not in allowed]
        if unknown: raise ValueError(f'Unknown GT colors in {label}: {unknown[:8]}')


def prepare_single_image(dataset_root, output, image_path):
    root = Path(dataset_root).resolve()
    image = Path(image_path).resolve()
    if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
        raise FileNotFoundError(f'Not a supported image: {image}')
    try: image.relative_to((root / 'train').resolve())
    except ValueError: raise ValueError(f'Image must be under the train split: {image}')
    label = _label_for_image(root, image)
    _validate_pair(image, label)
    image_rows = [{'image_id': image.name, 'image_path': str(image)}]
    label_rows = [{'image_id': image.name, 'gt_path': str(label.resolve())}]
    manifests = Path(output) / 'manifests'
    write_json(manifests / 'train_images.json', image_rows)
    write_json(manifests / 'train_labels.json', label_rows)
    report = {'images': 1, 'labels': 1, 'class_count': 15, 'dataset_root': str(root), 'selected_image': str(image)}
    write_json(manifests / 'audit.json', report)
    return report


def prepare_labels_for_manifest(dataset_root, output, image_rows):
    root = Path(dataset_root).resolve()
    label_rows = []
    for row in image_rows:
        image = Path(row['image_path']).resolve()
        label = _label_for_image(root, image)
        _validate_pair(image, label)
        label_rows.append({'image_id': row['image_id'], 'gt_path': str(label.resolve())})
    write_json(Path(output) / 'manifests' / 'train_labels.json', label_rows)
    return label_rows

def prepare_image_manifest(dataset_root, output, image_path=None):
    root = Path(dataset_root).resolve()
    images_dir = root / 'train'
    if not images_dir.is_dir():
        raise FileNotFoundError(f'Missing train image directory: {images_dir}')
    if image_path is not None:
        image = Path(image_path).resolve()
        try: image.relative_to(images_dir.resolve())
        except ValueError: raise ValueError(f'Image must be under the train split: {image}')
        images = [image]
    else:
        images = sorted(p for p in images_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f'No images found under {images_dir}')
    if len({p.name for p in images}) != len(images):
        raise ValueError('Duplicate image basename')
    records = [{'image_id': image.name, 'image_path': str(image.resolve())} for image in images]
    path = Path(output) / 'manifests' / 'train_images.json'
    write_json(path, records)
    return records

def validate_manifests(images, labels):
    a = read_json(images); b = read_json(labels)
    if not isinstance(a, list) or not isinstance(b, list) or not a or len(a) != len(b): raise ValueError('Manifest length mismatch')
    if len({r['image_id'] for r in a}) != len(a): raise ValueError('Duplicate image ID')
    if [r['image_id'] for r in a] != [r['image_id'] for r in b]: raise ValueError('Manifest ID mismatch')
    return a, b

def confusion(pred, gt, ignore_id=None):
    if pred.shape != gt.shape: raise ValueError('Prediction/GT size mismatch')
    valid = np.ones(gt.shape, dtype=bool) if ignore_id is None else gt != ignore_id
    if np.any((pred[valid] < 0) | (pred[valid] >= 15)): raise ValueError('Invalid prediction ID')
    if np.any((gt[valid] < 0) | (gt[valid] >= 15)): raise ValueError('Invalid GT ID')
    return np.bincount(gt[valid].astype(np.int64).ravel() * 15 + pred[valid].astype(np.int64).ravel(), minlength=225).reshape(15, 15)

def div(num, den): return None if den == 0 else float(num / den)

def metrics(cm):
    tp = cm.diagonal(); gt = cm.sum(axis=1); pred = cm.sum(axis=0)
    rows = []
    for i in range(15):
        rows.append({'id': i, 'name': NAMES[i], 'support': int(gt[i]), 'predicted_pixels': int(pred[i]), 'iou': div(tp[i], gt[i] + pred[i] - tp[i]), 'dice': div(2 * tp[i], gt[i] + pred[i]), 'precision': div(tp[i], pred[i]), 'recall': div(tp[i], gt[i])})
    valid = lambda seq: [x for x in seq if x is not None]
    foreground = cm[1:, 1:].sum(); fp = cm[0, 1:].sum(); fn = cm[1:, 0].sum()
    warnings = []
    if gt[0] > 0 and pred[0] == 0:
        warnings.append('Ground-truth background is present, but the model predicted no background pixels.')
    return {'images': None, 'miou_15': div(sum(valid([r['iou'] for r in rows])), len(valid([r['iou'] for r in rows]))), 'miou_foreground_14': div(sum(valid([r['iou'] for r in rows[1:]])), len(valid([r['iou'] for r in rows[1:]]))), 'pixel_accuracy': div(int(tp.sum()), int(cm.sum())), 'binary_foreground_iou': div(foreground, foreground + fp + fn), 'binary_foreground_dice': div(2 * foreground, 2 * foreground + fp + fn), 'quality_warnings': warnings, 'per_class': rows, 'confusion_matrix': cm.tolist()}

def evaluate(images_path, labels_path, predictions, output, seed=42, ignore_id=None, examples=24):
    images, labels = validate_manifests(images_path, labels_path)
    pred_dir = Path(predictions); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    cm = np.zeros((15, 15), np.int64)
    import random
    chosen = set(random.Random(seed).sample(range(len(images)), min(examples, len(images))))
    gallery = out / 'examples'; gallery.mkdir(exist_ok=True)
    for index, (im, lab) in enumerate(zip(images, labels)):
        path = pred_dir / f"{im['image_id']}.png"
        if not path.is_file(): raise FileNotFoundError(path)
        with Image.open(path) as p, Image.open(lab['gt_path']) as g:
            if p.mode not in ('L', 'P'): raise ValueError(f'Not class-ID PNG: {path}')
            prediction = np.asarray(p); ground = decode_mask(g)
        cm += confusion(prediction, ground, ignore_id)
        if index in chosen:
            with Image.open(im['image_path']) as source:
                src = np.asarray(source.convert('RGB'))
            blend = np.clip(src.astype(np.float32) * .55 + PALETTE[prediction].astype(np.float32) * .45, 0, 255).astype(np.uint8)
            panel = np.concatenate([src, PALETTE[ground], PALETTE[prediction], blend], axis=1)
            Image.fromarray(panel).save(gallery / f'{index:05d}.jpg', quality=88)
        if (index + 1) % 1000 == 0: print(f'[evaluate] {index+1}/{len(images)}', flush=True)
    result = metrics(cm); result['images'] = len(images); result['split'] = 'train'; result['zero_shot'] = True
    write_json(out / 'metrics.json', result)
    with (out / 'per_class.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=result['per_class'][0]); w.writeheader(); w.writerows(result['per_class'])
    np.savetxt(out / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    return result
