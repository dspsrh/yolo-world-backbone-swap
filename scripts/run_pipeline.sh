#!/usr/bin/env bash
# Two-stage backbone-swap pipeline on a single GPU (strictly sequential).
#
#   Stage 1 (screen): train each non-baseline backbone for 40 epochs.
#       40e screen == exact prefix of the 80e schedule: we cap ONLY
#       train_cfg.max_epochs=40, while default_hooks.param_scheduler.max_epochs
#       stays 80 (YOLOv5ParamSchedulerHook reads max_epochs from its own kwarg),
#       so the LR curve & augmentation match the v8 baseline's first 40 epochs.
#
#   Stage 2 (promote): scripts/promote.py resumes each winner (mAP@40 within
#       --margin of v8's mAP@40) from its epoch-40 checkpoint to the full 80
#       epochs -- no wasted compute, identical recipe (close-mosaic at ep70).
#
# Idempotent: a variant whose epoch_40.pth already exists is skipped; a partly
# screened variant is resumed. Safe to re-launch if the box reboots.
set -u
cd /home/huanghai/yolo_world_test
source scripts/env.sh

ARCHS="yolov9s yolov10s yolo11s yolo12s yolo26s"
SCREEN_EPOCHS=40
PROMOTE_MARGIN=0.005
mkdir -p logs
MASTER=logs/pipeline_master.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

log "=== PIPELINE START (screen ${SCREEN_EPOCHS}e: ${ARCHS}) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$MASTER"

for a in $ARCHS; do
  cfg="configs/yoloworld_s_${a}_coco.py"
  wd="work_dirs/${a}"
  slog="logs/screen_${a}.log"
  if [ -f "${wd}/epoch_${SCREEN_EPOCHS}.pth" ]; then
    log "SKIP ${a}: ${wd}/epoch_${SCREEN_EPOCHS}.pth exists (already screened)"
    continue
  fi
  resume=""
  if [ -f "${wd}/last_checkpoint" ]; then
    resume="--resume auto"
    log "RESUME ${a} screen from partial checkpoint"
  else
    log "START ${a} screen (${SCREEN_EPOCHS}e) -> ${slog}"
  fi
  "$PY" scripts/train.py "$cfg" --amp --work-dir "$wd" $resume \
      --cfg-options train_cfg.max_epochs=${SCREEN_EPOCHS} \
      > "$slog" 2>&1
  rc=$?
  log "DONE ${a} screen rc=${rc}"
  "$PY" scripts/collect_results.py --quiet 2>>"$MASTER"
done

log "=== SCREENING COMPLETE -- current table ==="
"$PY" scripts/collect_results.py | tee -a "$MASTER"

log "=== STAGE 2: PROMOTE winners (margin ${PROMOTE_MARGIN}) ==="
"$PY" scripts/promote.py --margin ${PROMOTE_MARGIN} 2>&1 | tee -a "$MASTER"

log "=== PIPELINE COMPLETE -- final table ==="
"$PY" scripts/collect_results.py | tee -a "$MASTER"
