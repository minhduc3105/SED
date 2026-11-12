"""Frozen upstream SED-L adapter and timed inference."""
from __future__ import annotations
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .core import ROOT, PALETTE, read_json, write_json, sha256
from .taxonomy import model_vocabulary, model_to_dataset_ids

UPSTREAM = ROOT / 'third_party' / 'SED'
COMMIT = 'db20d93c0ec8b0ce7fae267a10f826dfc7da15f5'


def verify_upstream():
    if not (UPSTREAM / 'sed').is_dir(): raise FileNotFoundError('Run setup/source.sh to clone official SED')
    actual = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT: raise RuntimeError(f'SED commit mismatch: {actual} != {COMMIT}')


def load_model(checkpoint, vocabulary, device):
    verify_upstream()
    sys.path.insert(0, str(UPSTREAM))
    import torch
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.checkpoint import DetectionCheckpointer
    from sed import add_sed_config  # registers model and decoder
    from sed.modeling.transformer.sed_predictor import SEDPredictor
    import sed  # noqa: F401
    names = read_json(vocabulary)
    if names != model_vocabulary(): raise ValueError('Vocabulary differs from taxonomy prompts')
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    cfg = get_cfg(); add_deeplab_config(cfg); add_sed_config(cfg)
    cfg.merge_from_file(str(UPSTREAM / 'configs' / 'convnextL_768.yaml'))
    # Upstream builds text features for both training and test lists even in eval mode.
    # Use the same fixed target vocabulary to avoid encoding unrelated COCO classes.
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = str(Path(vocabulary).resolve())
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = str(Path(vocabulary).resolve())
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [1, 1]  # official evaluation override
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(names)
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.DEVICE = device
    cfg.TEST.FAST_INFERENCE = False
    cfg.TEST.SLIDING_WINDOW = False
    cfg.freeze()
    # Upstream encodes 80 templates separately for every class. Batch the same
    # prompts to avoid hundreds of tiny ConvNeXt-L text encoder launches.
    embedding_cache = {}
    def batched_class_embeddings(self, classnames, templates, clip_model):
        cache_key = (tuple(classnames), tuple(templates), str(next(clip_model.parameters()).device))
        if cache_key in embedding_cache:
            return embedding_cache[cache_key]
        device = next(clip_model.parameters()).device
        ranges = []
        prompts = []
        for classname in classnames:
            variants = classname.split(', ') if ', ' in classname else [classname]
            start = len(prompts)
            prompts.extend(template.format(variant) for template in templates for variant in variants)
            ranges.append((start, len(prompts), len(variants)))
        encoded = []
        for start in range(0, len(prompts), 256):
            tokens = self.tokenizer(prompts[start:start + 256]).to(device)
            with torch.inference_mode():
                part = clip_model.encode_text(tokens).float()
                part = part / part.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            encoded.append(part)
        all_embeddings = torch.cat(encoded)
        output = []
        for start, end, variants in ranges:
            embedding = all_embeddings[start:end].reshape(len(templates), variants, -1).mean(dim=1)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            output.append(embedding)
        result = torch.stack(output, dim=1)
        embedding_cache[cache_key] = result
        return result
    SEDPredictor.class_embeddings = batched_class_embeddings
    # Building upstream model downloads its OpenCLIP initialization weights once.
    model = build_model(cfg)
    features = model.sem_seg_head.predictor.text_features_test
    if features.shape[0] != len(names) or not torch.isfinite(features).all():
        raise RuntimeError(f'Invalid text features for {len(names)} SED concepts: {tuple(features.shape)}')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = state.get('model', state)
    own = model.state_dict()
    critical = [k for k in own if (k.startswith('sem_seg_head.predictor.transformer.') or k.startswith('sem_seg_head.predictor.clip_model.visual.'))]
    missing = [k for k in critical if k not in state or tuple(state[k].shape) != tuple(own[k].shape)]
    if missing: raise RuntimeError(f'Checkpoint lacks compatible SED-L encoder/decoder weights: {missing[:8]} ({len(missing)} total)')
    DetectionCheckpointer(model).load(str(checkpoint))
    model.to(device).float().eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, cfg


def fingerprint(checkpoint, vocabulary, manifest, cfg):
    return {'upstream_commit': COMMIT, 'checkpoint_sha256': sha256(checkpoint), 'vocabulary_sha256': sha256(vocabulary), 'manifest_sha256': sha256(manifest), 'model_config_sha256': sha256(UPSTREAM / 'configs' / 'convnextL_768.yaml'), 'adapter_sha256': sha256(Path(__file__)), 'taxonomy_sha256': sha256(ROOT / 'sed_r2s100k' / 'taxonomy.py'), 'fast_inference': False, 'precision': 'fp32', 'device': cfg['device']}


def input_dataset(manifest, indices=None):
    import torch
    from torch.utils.data import Dataset
    rows = read_json(manifest)
    if indices is not None: rows = [rows[i] for i in indices]
    class ImageDataset(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            with Image.open(row['image_path']) as im:
                rgb = np.asarray(im.convert('RGB')).copy()
            h, w = rgb.shape[:2]
            return {'image': torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float(), 'height': h, 'width': w, 'image_id': row['image_id'], 'read_end': time.perf_counter()}
    return ImageDataset()


def valid_prediction(path, shape):
    try:
        with Image.open(path) as im:
            if im.format != 'PNG' or im.mode != 'L' or im.size != (shape[1], shape[0]): return False
            arr = np.asarray(im)
            return arr.min() >= 0 and arr.max() < 15
    except (OSError, ValueError): return False



def colorize_ids(ids):
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.size == 0 or ids.min() < 0 or ids.max() >= len(PALETTE):
        raise ValueError('Expected a nonempty 2D mask with class IDs 0–14')
    return PALETTE[ids.astype(np.uint8)]


def save_color_prediction(path, ids):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.png')
    Image.fromarray(colorize_ids(ids), mode='RGB').save(tmp, format='PNG')
    os.replace(tmp, path)


def valid_color_prediction(path, ids):
    try:
        with Image.open(path) as im:
            return im.format == 'PNG' and im.mode == 'RGB' and im.size == (ids.shape[1], ids.shape[0]) and np.array_equal(np.asarray(im), colorize_ids(ids))
    except (OSError, ValueError):
        return False

def infer(manifest, checkpoint, vocabulary, output, device='cuda', workers=4, indices=None, max_images=None, resume=True, deadline=None):
    import torch
    from torch.utils.data import DataLoader
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    if read_json(vocabulary) != model_vocabulary(): raise ValueError('Vocabulary differs from taxonomy prompts')
    channel_to_id = torch.as_tensor(model_to_dataset_ids(), dtype=torch.uint8, device=device)
    out = Path(output); pred_dir = out / 'predictions'; color_dir = out / 'colored_predictions'
    pred_dir.mkdir(parents=True, exist_ok=True); color_dir.mkdir(parents=True, exist_ok=True)
    config = {'device': device}
    fp = fingerprint(checkpoint, vocabulary, manifest, config)
    meta_path = out / 'run_metadata.json'
    if meta_path.exists():
        old = read_json(meta_path)
        if old['fingerprint'] != fp: raise RuntimeError('Output fingerprint mismatch; choose another output directory')
    elif any(pred_dir.iterdir()):
        raise RuntimeError('Predictions exist without metadata; choose another output directory')
    else:
        write_json(meta_path, {'fingerprint': fp, 'vocabulary': model_vocabulary(), 'channel_to_dataset_id': model_to_dataset_ids(), 'status': 'incomplete', 'started_at': time.time()})
    dataset = input_dataset(manifest, indices)
    if max_images is not None: dataset = torch.utils.data.Subset(dataset, range(min(max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers, pin_memory=device.startswith('cuda'), persistent_workers=workers > 0, prefetch_factor=2 if workers else None, collate_fn=lambda batch: batch[0])
    model, _ = load_model(checkpoint, vocabulary, device)
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    times = []; skipped = 0; start = time.perf_counter(); completed = 0; observed = 0
    with torch.inference_mode():
        for item in loader:
            if deadline and time.perf_counter() >= deadline: break
            path = pred_dir / f"{item['image_id']}.png"
            color_path = color_dir / f"{item['image_id']}.png"
            shape = (item['height'], item['width'])
            if path.exists():
                if not resume or not valid_prediction(path, shape): raise RuntimeError(f'Invalid or unresumable prediction: {path}')
                with Image.open(path) as existing:
                    existing_ids = np.asarray(existing, dtype=np.uint8)
                if not valid_color_prediction(color_path, existing_ids):
                    save_color_prediction(color_path, existing_ids)
                skipped += 1; completed += 1; continue
            t0 = time.perf_counter()
            data = {'image': item['image'].to(device, non_blocking=True), 'height': shape[0], 'width': shape[1]}
            score = model([data])[0]['sem_seg']
            if score.shape != (len(channel_to_id), *shape) or not torch.isfinite(score).all(): raise RuntimeError(f'Bad SED output: {item["image_id"]}: {tuple(score.shape)}')
            ids = channel_to_id[score.argmax(0)].cpu().numpy()
            tmp = path.with_name(path.name + '.tmp')
            Image.fromarray(ids, mode='L').save(tmp, format='PNG')
            os.replace(tmp, path)
            save_color_prediction(color_path, ids)
            if device.startswith('cuda'): torch.cuda.synchronize()
            times.append(time.perf_counter() - t0); completed += 1; observed += 1
            if completed % 100 == 0:
                eta = (len(dataset) - completed) * statistics.mean(times[-100:])
                print(f'[infer] {completed}/{len(dataset)}; ETA {eta/60:.1f} min', flush=True)
                write_json(meta_path, {'fingerprint': fp, 'vocabulary': model_vocabulary(), 'channel_to_dataset_id': model_to_dataset_ids(), 'status': 'incomplete', 'completed': completed, 'skipped': skipped, 'elapsed_seconds': time.perf_counter()-start})
    complete = completed == len(dataset)
    result = {'fingerprint': fp, 'vocabulary': model_vocabulary(), 'channel_to_dataset_id': model_to_dataset_ids(), 'status': 'complete' if complete else 'incomplete', 'completed': completed, 'expected': len(dataset), 'skipped': skipped, 'elapsed_seconds': time.perf_counter()-start, 'throughput_seconds_per_image': (time.perf_counter()-start)/max(1, observed), 'latency_mean_seconds': statistics.mean(times) if times else None, 'latency_p95_seconds': sorted(times)[math.ceil(.95*len(times))-1] if times else None, 'peak_vram_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else None, 'gpu': torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU'}
    write_json(meta_path, result)
    return result
