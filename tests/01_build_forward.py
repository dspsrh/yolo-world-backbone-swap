#!/usr/bin/env python
"""Architecture smoke test for every backbone-swapped YOLO-World config.

For each configs/yoloworld_s_*.py this:
  1. builds the full YOLOWorldDetector (backbone swap + text branch + neck + head),
  2. runs a forward pass (mode='tensor') on a synthetic image+text batch and
     checks the head emits 3 pyramid levels at strides 8/16/32,
  3. runs a real loss + backward on synthetic GT boxes (exercises the assigner,
     contrastive head and gradient flow),
  4. reports parameter count.

No COCO images are needed (inputs are random tensors); this validates the
channel wiring of each swapped backbone end-to-end before any training.
Run inside the venv with scripts/env.sh sourced:
    python tests/01_build_forward.py
"""
import glob
import json
import os
import sys
import traceback

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.structures import InstanceData
from mmyolo.registry import MODELS
from yolo_world.datasets.utils import yolow_collate

ROOT = "/home/huanghai/yolo_world_test"
CONFIGS = sorted(glob.glob(f"{ROOT}/configs/yoloworld_s_*.py"))
TEXTS_JSON = f"{ROOT}/data/texts/coco_class_texts.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, IMG = 2, 640


def make_batch(model, num_classes=80):
    """Build a batch the *real* way: per-image pipeline outputs -> yolow_collate
    -> data_preprocessor. Images are random; GT boxes are synthetic. This also
    exercises the collate fn and the YOLOWDetDataPreprocessor (which injects the
    'img_metas' the head loss relies on)."""
    with open(TEXTS_JSON) as f:
        raw = json.load(f)  # [["person"], ["bicycle"], ...]
    # the text backbone wants a flat list[str] per image (one prompt per class)
    texts = [syn[0] for syn in raw]
    items = []
    for _ in range(B):
        ds = DetDataSample()
        ds.set_metainfo(dict(
            img_shape=(IMG, IMG), ori_shape=(IMG, IMG),
            batch_input_shape=(IMG, IMG), pad_param=(0, 0, 0, 0),
            scale_factor=(1.0, 1.0), flip=False))
        n = 3
        cx, cy = torch.rand(n) * IMG, torch.rand(n) * IMG
        w, h = torch.rand(n) * 80 + 20, torch.rand(n) * 80 + 20
        bboxes = torch.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        gt = InstanceData()
        gt.bboxes = HorizontalBoxes(bboxes)
        gt.labels = torch.randint(0, num_classes, (n,))
        ds.gt_instances = gt
        ds.texts = texts
        img = torch.randint(0, 256, (3, IMG, IMG), dtype=torch.uint8)
        items.append(dict(inputs=img, data_samples=ds))
    batch = yolow_collate(items)
    batch = model.data_preprocessor(batch, training=True)
    return batch["inputs"], batch["data_samples"]


def test_config(path):
    name = os.path.basename(path)
    cfg = Config.fromfile(path)
    init_default_scope(cfg.get("default_scope", "mmyolo"))

    model = MODELS.build(cfg.model)
    model.init_weights()
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    imgs, samples = make_batch(model)
    H = imgs.shape[-1]

    # --- (a) forward (tensor mode): checks backbone->neck->head shape wiring
    model.eval()
    with torch.no_grad():
        outs = model._forward(imgs, samples)
    cls_scores = outs[0]
    assert len(cls_scores) == 3, f"expected 3 pyramid levels, got {len(cls_scores)}"
    strides = [H // c.shape[-1] for c in cls_scores]
    assert strides == [8, 16, 32], f"unexpected strides {strides}"

    # --- (b) real loss + backward: checks assigner / contrastive head / grads
    model.train()
    losses = model.loss(imgs, samples)
    total = sum(v.sum() for v in losses.values() if isinstance(v, torch.Tensor))
    total.backward()
    finite = bool(torch.isfinite(total).all())
    loss_keys = ",".join(sorted(losses.keys()))

    return dict(name=name, ok=True, params=n_params, strides=strides,
                loss=float(total.detach()), finite=finite, loss_keys=loss_keys)


def main():
    print(f"device={DEVICE}  torch={torch.__version__}")
    print(f"found {len(CONFIGS)} configs\n")
    results, failed = [], 0
    for path in CONFIGS:
        try:
            r = test_config(path)
            print(f"[PASS] {r['name']:34s} params={r['params']/1e6:6.2f}M  "
                  f"strides={r['strides']}  loss={r['loss']:8.3f}  "
                  f"finite={r['finite']}  ({r['loss_keys']})")
            results.append(r)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {os.path.basename(path):34s} {type(e).__name__}: {e}")
            traceback.print_exc()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print(f"\n{len(results)}/{len(CONFIGS)} configs passed; {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
