#!/usr/bin/env python
"""End-to-end *data pipeline* smoke test on real COCO images.

Unlike 01_build_forward.py (synthetic tensors), this exercises the parts only
real data can reach:
  * the multimodal train pipeline (MultiModalMosaic, CopyPaste, RandomAffine,
    MixUp, RandomLoadText) reading real JPGs from disk,
  * yolow_collate + YOLOWDetDataPreprocessor,
  * a cfg-built optimiser (YOLOWv5OptimizerConstructor + AdamW) doing real
    train_step updates (forward+loss+backward+step),
  * the val/predict path + mmdet CocoMetric.

It runs on val2017 (present on disk) as a stand-in "train" set, so it does not
wait for train2017. Only a couple of iterations / batches are run.

Usage (venv + scripts/env.sh sourced):
    python tests/02_pipeline_smoke.py [config.py] [--train-iters N] [--val-batches N]
"""
import argparse
import copy
import os

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.evaluator import Evaluator
from mmyolo.registry import MODELS

ROOT = "/home/huanghai/yolo_world_test"


def point_at_val2017(dataloader_cfg):
    """Repoint a (train) dataloader's inner COCO dataset at val2017."""
    dl = copy.deepcopy(dataloader_cfg)
    inner = dl["dataset"]["dataset"]  # MultiModalDataset -> YOLOv5CocoDataset
    inner["ann_file"] = "annotations/instances_val2017.json"
    inner["data_prefix"] = dict(img="val2017/")
    dl["batch_size"] = 2
    dl["num_workers"] = 2
    dl["persistent_workers"] = False
    return dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?",
                    default=f"{ROOT}/configs/yoloworld_s_yolov8s_coco.py")
    ap.add_argument("--train-iters", type=int, default=2)
    ap.add_argument("--val-batches", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmyolo"))
    print(f"config={os.path.basename(args.config)}  device={device}")

    # random-init weights almost never clear the default score_thr=0.001, which
    # makes CocoMetric see an empty result set. Drop the threshold so the metric
    # computation path is genuinely exercised (the mAP itself is meaningless here).
    if "test_cfg" in cfg.model and cfg.model.test_cfg is not None:
        cfg.model.test_cfg.score_thr = 0.0
        cfg.model.test_cfg.nms_pre = 1000
        cfg.model.test_cfg.max_per_img = 100

    model = MODELS.build(cfg.model)
    model.init_weights()
    model = model.to(device)

    # ---------------------------------------------------------------- TRAIN
    train_dl_cfg = point_at_val2017(cfg.train_dataloader)
    train_loader = Runner.build_dataloader(train_dl_cfg)
    optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
    print(f"[train] dataset={len(train_loader.dataset)} imgs  "
          f"batch={train_dl_cfg['batch_size']}  running {args.train_iters} iters")

    model.train()
    it = iter(train_loader)
    for i in range(args.train_iters):
        data = next(it)
        log_vars = model.train_step(data, optim_wrapper=optim_wrapper)
        msg = "  ".join(f"{k}={float(v):.3f}" for k, v in log_vars.items())
        print(f"  iter {i}: {msg}")
        assert all(torch.isfinite(torch.as_tensor(float(v))) for v in log_vars.values())

    # ----------------------------------------------------------------- VAL
    val_loader = Runner.build_dataloader(cfg.val_dataloader)
    evaluator = Evaluator(cfg.val_evaluator)
    evaluator.dataset_meta = val_loader.dataset.metainfo
    print(f"[val] dataset={len(val_loader.dataset)} imgs  "
          f"feeding {args.val_batches} batches to CocoMetric")

    model.eval()
    n_seen = 0
    vit = iter(val_loader)
    for _ in range(args.val_batches):
        data = next(vit)
        with torch.no_grad():
            outputs = model.val_step(data)
        evaluator.process(data_samples=outputs, data_batch=data)
        n_seen += len(outputs)
    metrics = evaluator.evaluate(n_seen)
    print(f"[val] processed {n_seen} imgs; CocoMetric keys: {list(metrics.keys())}")
    print(f"[val] bbox_mAP (random-init, expected ~0): "
          f"{metrics.get('coco/bbox_mAP', 'n/a')}")
    print("\nPIPELINE SMOKE OK")


if __name__ == "__main__":
    main()
