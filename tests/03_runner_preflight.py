#!/usr/bin/env python
"""Full-Runner pre-flight before committing to a multi-day training job.

tests/01 and 02 validated the model and the data/optim/eval pieces. This drives
the *actual* mmengine Runner used by scripts/train.py end-to-end, so it exercises
the parts only the real training loop touches:
  * AmpOptimWrapper (mixed precision, as used by `--amp`),
  * EMAHook, YOLOv5ParamSchedulerHook across real epoch boundaries,
  * mmdet.PipelineSwitchHook -> the close-mosaic *stage-2* pipeline,
  * the validation loop + CocoMetric + checkpoint saving.

It runs 2 tiny epochs on a small val2017 subset (close_mosaic forces the stage-2
switch at epoch 1) in a throwaway work_dir, so it finishes in ~1-2 minutes.

Usage (venv + scripts/env.sh):  python tests/03_runner_preflight.py
"""
import os
import shutil

from mmengine.config import Config
from mmengine.runner import Runner

ROOT = "/home/huanghai/yolo_world_test"
CFG = f"{ROOT}/configs/yoloworld_s_yolov8s_coco.py"
WORK = f"{ROOT}/work_dirs/_preflight"
N_TRAIN, N_VAL = 256, 128


def main():
    cfg = Config.fromfile(CFG)
    cfg.work_dir = WORK
    if os.path.exists(WORK):
        shutil.rmtree(WORK)

    # tiny subset of val2017 (present on disk) as a stand-in train+val set
    for dl in (cfg.train_dataloader, cfg.val_dataloader):
        inner = dl["dataset"]["dataset"]
        inner["ann_file"] = "annotations/instances_val2017.json"
        inner["data_prefix"] = dict(img="val2017/")
    cfg.train_dataloader.dataset.dataset["indices"] = N_TRAIN
    cfg.val_dataloader.dataset.dataset["indices"] = N_VAL
    cfg.train_dataloader.batch_size = 8
    cfg.train_dataloader.num_workers = 4
    cfg.val_dataloader.batch_size = 8
    cfg.val_dataloader.num_workers = 4

    # 2 epochs; close-mosaic switch fires at epoch (2-1)=1 -> exercises stage-2
    max_epochs, close_mosaic = 2, 1
    cfg.train_cfg.max_epochs = max_epochs
    cfg.train_cfg.val_interval = 1
    cfg.train_cfg.dynamic_intervals = None
    cfg.default_hooks.param_scheduler.max_epochs = max_epochs
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.save_best = None
    for h in cfg.custom_hooks:
        if h["type"] == "mmdet.PipelineSwitchHook":
            h["switch_epoch"] = max_epochs - close_mosaic

    # AMP path (same as scripts/train.py --amp)
    cfg.optim_wrapper.type = "AmpOptimWrapper"
    cfg.optim_wrapper.loss_scale = "dynamic"

    # let random-ish predictions through so CocoMetric is exercised
    if cfg.model.get("test_cfg"):
        cfg.model.test_cfg.score_thr = 0.0
        cfg.model.test_cfg.max_per_img = 100

    runner = Runner.from_cfg(cfg)
    runner.train()

    ckpts = [f for f in os.listdir(WORK) if f.endswith(".pth")]
    print(f"\nPREFLIGHT OK  (checkpoints saved: {ckpts})")


if __name__ == "__main__":
    main()
