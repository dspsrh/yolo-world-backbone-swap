#!/usr/bin/env python
"""Thin single-GPU training launcher (mmengine Runner) for the backbone-swap
YOLO-World configs. Equivalent to mmdet/mmyolo tools/train.py but self-contained
in yolo_world_test (no repo tools are used).

Examples:
    python scripts/train.py configs/yoloworld_s_yolov8s_coco.py
    python scripts/train.py configs/yoloworld_s_yolo11s_coco.py --amp \
        --work-dir work_dirs/yolo11s --cfg-options train_cfg.max_epochs=40
"""
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

ROOT = "/home/huanghai/yolo_world_test"


def parse_args():
    p = argparse.ArgumentParser(description="Train a backbone-swapped YOLO-World")
    p.add_argument("config", help="path to a configs/yoloworld_s_*.py")
    p.add_argument("--work-dir", default=None,
                   help="output dir (default: work_dirs/<config stem>)")
    p.add_argument("--amp", action="store_true", help="enable mixed precision")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="resume from a checkpoint (or 'auto' for latest)")
    p.add_argument("--cfg-options", nargs="+", action=DictAction, default=None,
                   help="override config entries, e.g. train_cfg.max_epochs=40")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        stem = osp.splitext(osp.basename(args.config))[0]
        cfg.work_dir = osp.join(ROOT, "work_dirs", stem)

    if args.amp:
        # wrap the optimiser in AmpOptimWrapper (fp16)
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.setdefault("loss_scale", "dynamic")

    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
