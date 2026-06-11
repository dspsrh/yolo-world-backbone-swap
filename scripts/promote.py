#!/usr/bin/env python
"""Resume 'winner' backbones from the 40-epoch screen to the full 80 epochs.

A *winner* is a screened variant whose best mAP within 40 epochs is within
``--margin`` of (or above) the yolov8s baseline's mAP@40. Because the screen is
an exact prefix of the 80e schedule (param_scheduler.max_epochs stays 80, only
train_cfg.max_epochs was capped), we simply ``--resume`` the epoch-40 checkpoint
with the unmodified 80e config: the LR curve continues and close-mosaic fires at
epoch 70 -- identical recipe to the v8 baseline, no compute wasted.

Single GPU => promotions run sequentially. Run this AFTER screening finishes
(run_pipeline.sh calls it automatically).

Usage:
    python scripts/promote.py --list                 # decision only, no training
    python scripts/promote.py                         # resume all winners to 80e
    python scripts/promote.py --margin 0.005          # inclusiveness knob
    python scripts/promote.py --only yolo11s yolov9s  # force a specific set
"""
import argparse
import os
import subprocess
import sys

ROOT = "/home/huanghai/yolo_world_test"
PY = os.path.join(ROOT, "venv", "bin", "python")
TRAIN = os.path.join(ROOT, "scripts", "train.py")
LOGS = os.path.join(ROOT, "logs")

CANDIDATES = ["yolov9s", "yolov10s", "yolo11s", "yolo12s", "yolo26s"]
BASELINE = "yolov8s"
FULL_EPOCHS = 80


def get_summary():
    """(Re)build results/map_results.json and return its parsed summary."""
    subprocess.run([PY, os.path.join(ROOT, "scripts", "collect_results.py"),
                    "--quiet"], cwd=ROOT, check=False)
    import json
    with open(os.path.join(ROOT, "results", "map_results.json")) as f:
        return json.load(f)


def decide(data, margin):
    summ = data["summary"]
    base40 = summ.get(BASELINE, {}).get("mAP_at40")
    if base40 is None:
        print(f"[promote] WARNING: no baseline mAP@40 for {BASELINE}; "
              f"using absolute threshold 0.0")
        base40 = 0.0
    thr = base40 - margin
    winners, decisions = [], []
    for a in CANDIDATES:
        s = summ.get(a, {})
        m40 = s.get("mAP_at40")
        if m40 is None:
            decisions.append((a, None, "no 40e result -> SKIP"))
            continue
        if m40 >= thr:
            winners.append(a)
            decisions.append((a, m40, f"PROMOTE (>= {thr:.3f})"))
        else:
            decisions.append((a, m40, f"drop (< {thr:.3f})"))
    return base40, thr, winners, decisions


def resume_to_full(arch):
    cfg = os.path.join(ROOT, "configs", f"yoloworld_s_{arch}_coco.py")
    wd = os.path.join(ROOT, "work_dirs", arch)
    log = os.path.join(LOGS, f"promote_{arch}.log")
    last = os.path.join(wd, "last_checkpoint")
    if not os.path.exists(last):
        print(f"[promote] {arch}: no checkpoint in {wd}; cannot resume -> SKIP")
        return 1
    print(f"[promote] resuming {arch} 40->{FULL_EPOCHS}  (log: {log})")
    with open(log, "a") as lf:
        # default config max_epochs=80; --resume auto continues from epoch 40.
        proc = subprocess.run(
            [PY, TRAIN, cfg, "--amp", "--work-dir", wd, "--resume", "auto"],
            cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[promote] {arch} finished rc={proc.returncode}")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.005,
                    help="promote if mAP@40 >= baseline_mAP@40 - margin")
    ap.add_argument("--list", action="store_true", help="print decision only")
    ap.add_argument("--only", nargs="+", default=None,
                    help="force this set of archs (ignores the rule)")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    data = get_summary()
    base40, thr, winners, decisions = decide(data, args.margin)

    print(f"\n[promote] baseline {BASELINE} mAP@40 = {base40:.3f} | "
          f"promotion threshold (>= {thr:.3f}, margin {args.margin})")
    for a, m40, msg in decisions:
        m = "  -  " if m40 is None else f"{m40:.3f}"
        print(f"   {a:10s} mAP@40={m}  ->  {msg}")

    if args.only is not None:
        winners = list(args.only)
        print(f"[promote] --only override: {winners}")
    print(f"[promote] winners -> {winners or '(none)'}\n")

    if args.list:
        return
    for a in winners:
        resume_to_full(a)
    # refresh final table
    subprocess.run([PY, os.path.join(ROOT, "scripts", "collect_results.py")],
                   cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
