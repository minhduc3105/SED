import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from sed_r2s100k.core import PALETTE, prepare, prepare_image_manifest, prepare_single_image, decode_mask, confusion, metrics, evaluate, materialize_vocabulary
from sed_r2s100k.inference import valid_prediction, colorize_ids, save_color_prediction, valid_color_prediction


def test_palette_and_metrics():
    gt = np.arange(15, dtype=np.uint8).reshape(3, 5)
    assert np.array_equal(decode_mask(PALETTE[gt]), gt)
    cm = confusion(gt, gt)
    result = metrics(cm)
    assert result['miou_15'] == 1 and result['miou_foreground_14'] == 1
    assert result['binary_foreground_iou'] == 1
    assert metrics(np.zeros((15, 15), dtype=np.int64))['miou_15'] is None
    no_background = confusion(np.ones((2, 2), np.uint8), np.zeros((2, 2), np.uint8))
    assert metrics(no_background)['per_class'][0]['predicted_pixels'] == 0
    assert metrics(no_background)['quality_warnings']
    pred = gt.copy(); pred[0, 1] = 0
    cm = confusion(pred, gt)
    assert metrics(cm)['pixel_accuracy'] == 14 / 15
    assert metrics(cm)['binary_foreground_iou'] == 13 / 14
    assert confusion(pred, gt, ignore_id=1).sum() == 14
    with pytest.raises(ValueError, match='Unknown'):
        decode_mask(np.array([[[255, 254, 253]]], dtype=np.uint8))


def test_prepare_and_evaluate(tmp_path):
    root = tmp_path / 'data'; (root/'train').mkdir(parents=True); (root/'train_labels'/'Train-Labels').mkdir(parents=True)
    name = 'frame with spaces.jpg'
    Image.new('RGB', (5, 3)).save(root/'train'/name)
    mask = PALETTE[np.zeros((3, 5), np.uint8)]
    Image.fromarray(mask).save(root/'train_labels'/'Train-Labels'/f'{name}___fuse.png')
    out = tmp_path/'out'; assert prepare(root, out)['images'] == 1
    pred = out/'predictions'; pred.mkdir()
    Image.fromarray(np.zeros((3, 5), np.uint8)).save(pred/f'{name}.png')
    assert valid_prediction(pred/f'{name}.png', (3, 5))
    result = evaluate(out/'manifests'/'train_images.json', out/'manifests'/'train_labels.json', pred, out/'evaluation')
    assert result['miou_15'] == 1 and result['images'] == 1
    (pred/f'{name}.png').unlink()
    with pytest.raises(FileNotFoundError):
        evaluate(out/'manifests'/'train_images.json', out/'manifests'/'train_labels.json', pred, out/'evaluation')
    (root/'train_labels'/'Train-Labels'/f'{name}___fuse.png').unlink()
    with pytest.raises(FileNotFoundError): prepare(root, tmp_path/'other')


def test_prepare_rejects_bad_data(tmp_path):
    root = tmp_path/'data'; (root/'train').mkdir(parents=True); (root/'train_labels'/'Train-Labels').mkdir(parents=True)
    image = root/'train'/'one.jpg'; label = root/'train_labels'/'Train-Labels'/'one.jpg___fuse.png'
    Image.new('RGB', (2, 2)).save(image)
    Image.fromarray(np.full((2, 2, 3), 254, np.uint8)).save(label)
    with pytest.raises(ValueError, match='Unknown'): prepare(root, tmp_path/'out')
    Image.fromarray(PALETTE[np.zeros((2, 3), np.uint8)]).save(label)
    with pytest.raises(ValueError, match='Size mismatch'): prepare(root, tmp_path/'out')


def test_color_prediction_png(tmp_path):
    ids = np.array([[0, 1, 14]], dtype=np.uint8)
    path = tmp_path / 'colored_predictions' / 'mask.png'
    save_color_prediction(path, ids)
    assert valid_color_prediction(path, ids)
    with Image.open(path) as mask:
        assert mask.format == 'PNG' and mask.mode == 'RGB'
        assert np.array_equal(np.asarray(mask), PALETTE[ids])
    assert not valid_color_prediction(path, ids[:, :2])
    with pytest.raises(ValueError, match='class IDs'):
        colorize_ids(np.array([[15]], dtype=np.uint8))


def test_image_manifest_does_not_require_labels(tmp_path):
    root = tmp_path/'data'; (root/'train').mkdir(parents=True)
    Image.new('RGB', (4, 3)).save(root/'train'/'frame.jpg')
    rows = prepare_image_manifest(root, tmp_path/'out')
    assert rows[0]['image_id'] == 'frame.jpg'
    assert (tmp_path/'out'/'manifests'/'train_images.json').is_file()
    assert not (tmp_path/'out'/'manifests'/'train_labels.json').exists()


def test_taxonomy_and_model_vocabulary_stay_aligned(tmp_path, monkeypatch):
    from sed_r2s100k.taxonomy import CLASSES, DESCRIPTION_PROMPTS, BACKGROUND_PROMPT_VARIANTS, model_vocabulary, model_to_dataset_ids
    path = materialize_vocabulary(tmp_path/'first')
    vocabulary = json.loads(path.read_text())
    assert set(CLASSES) == set(range(15))
    assert vocabulary == model_vocabulary() == [DESCRIPTION_PROMPTS[i] for i in range(1, 15)] + list(BACKGROUND_PROMPT_VARIANTS)
    assert model_to_dataset_ids() == list(range(1, 15)) + [0] * len(BACKGROUND_PROMPT_VARIANTS)
    assert CLASSES[0][2] == (0, 0, 0)
    assert 'not ignored' in CLASSES[0][1]
    monkeypatch.setitem(DESCRIPTION_PROMPTS, 1, 'rain-soaked roadway')
    with pytest.raises(RuntimeError, match='new output directory'):
        materialize_vocabulary(tmp_path/'first')
    assert json.loads(materialize_vocabulary(tmp_path/'second').read_text())[0] == 'rain-soaked roadway'


def test_single_image_manifest_and_label_pair(tmp_path):
    root = tmp_path/'data'; (root/'train').mkdir(parents=True); (root/'train_labels'/'Train-Labels').mkdir(parents=True)
    image = root/'train'/'chosen image.jpg'
    label = root/'train_labels'/'Train-Labels'/'chosen image.jpg___fuse.png'
    Image.new('RGB', (4, 3)).save(image)
    Image.fromarray(PALETTE[np.zeros((3, 4), np.uint8)]).save(label)
    out = tmp_path/'out'
    report = prepare_single_image(root, out, image)
    assert report['images'] == 1
    assert len(json.loads((out/'manifests'/'train_images.json').read_text())) == 1
    assert len(json.loads((out/'manifests'/'train_labels.json').read_text())) == 1
    assert prepare_image_manifest(root, tmp_path/'infer_only', image)[0]['image_id'] == image.name
    outside = tmp_path/'outside.jpg'; Image.new('RGB', (2, 2)).save(outside)
    with pytest.raises(ValueError, match='under the train split'):
        prepare_single_image(root, out, outside)
