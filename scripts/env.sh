#!/usr/bin/env bash
# Source this to configure the YOLO-World backbone-swap environment.
# Uses PYTHONPATH (not pip install) for mmyolo + yolo_world so nothing outside
# yolo_world_test is modified.
export YWT=/home/huanghai/yolo_world_test
export YOLO_WORLD_REPO=/home/huanghai/cv_proj/yolo_world/YOLO-World
export MMYOLO_DIR=$YWT/third_party/mmyolo
export PY=$YWT/venv/bin/python

# mmyolo (fork) + yolo_world (patched copy) + our adapters/configs, all via PYTHONPATH.
# yw_src holds a copy of the yolo_world package with two minimal fixes to bugs in
# the cloned fork (the original repo is never modified; it stays on the path only
# as a fallback):
#   1. detectors/yolo_world.py:61  `None` -> `_`  (a SyntaxError in the fork)
#   2. dense_heads/yolo_world_head.py YOLOWorldHead.loss  now forwards `txt_masks`
#      to loss_by_feat (the fork's loss_by_feat requires batch_text_masks but the
#      loss method didn't pass it -> "missing positional argument: batch_img_metas")
export YW_SRC=$YWT/yw_src
export PYTHONPATH=$YW_SRC:$MMYOLO_DIR:$YWT:$YOLO_WORLD_REPO:${PYTHONPATH:-}

# This box cannot reach HuggingFace at runtime; force offline use of local CLIP.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Ultralytics: keep its config/weights inside the project, run offline/headless.
export YOLO_CONFIG_DIR=$YWT/.ultralytics
export MPLBACKEND=Agg
