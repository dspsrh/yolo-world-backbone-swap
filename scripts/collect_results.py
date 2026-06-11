#!/usr/bin/env python
"""Aggregate accuracy (COCO mAP) + speed/size for every backbone variant.

Accuracy is read straight from the mmengine ``scalars.json`` files under each
``work_dirs/<arch>/<timestamp>/vis_data/`` (one per launched / resumed run), so
it works mid-training and after a 40->80 resume. Speed/size come from
``results/benchmark_speed.json`` (see scripts/benchmark.py).

Outputs:
    results/map_results.json   machine-readable, per-variant
    results/map_results.md     human-readable comparison table

Usage:
    python scripts/collect_results.py            # all variants
    python scripts/collect_results.py --quiet    # no stdout table
"""
import argparse
import glob
import json
import os

ROOT = "/home/huanghai/yolo_world_test"
WORK = os.path.join(ROOT, "work_dirs")
RESULTS = os.path.join(ROOT, "results")

VARIANTS = ["yolov8s", "yolov9s", "yolov10s", "yolo11s", "yolo12s", "yolo26s"]

# yolov8s is the trained baseline; screening compares each variant @epoch40 to
# v8 @epoch40, and the full result @best to v8 @best.
BASELINE_ARCH = "yolov8s"


def read_val_points(arch):
    """Return list of dicts {epoch, mAP, mAP_50, mAP_75, s, m, l} for one arch,
    merged across all run dirs (screening + any resume), deduped by epoch
    (latest run wins)."""
    pattern = os.path.join(WORK, arch, "*", "vis_data", "scalars.json")
    files = sorted(glob.glob(pattern))  # chronological by timestamp dir name
    by_epoch = {}
    for fp in files:
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line or "coco/bbox_mAP" not in line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "coco/bbox_mAP" not in d:
                        continue
                    ep = int(d.get("step", -1))
                    by_epoch[ep] = dict(
                        epoch=ep,
                        mAP=d.get("coco/bbox_mAP"),
                        mAP_50=d.get("coco/bbox_mAP_50"),
                        mAP_75=d.get("coco/bbox_mAP_75"),
                        mAP_s=d.get("coco/bbox_mAP_s"),
                        mAP_m=d.get("coco/bbox_mAP_m"),
                        mAP_l=d.get("coco/bbox_mAP_l"),
                    )
        except OSError:
            continue
    return [by_epoch[e] for e in sorted(by_epoch)]


def summarize(arch):
    pts = read_val_points(arch)
    if not pts:
        return dict(arch=arch, status="not started", points=[])
    best = max(pts, key=lambda p: (p["mAP"] if p["mAP"] is not None else -1))
    last = pts[-1]
    # value at/closest-below epoch 40 (the screening checkpoint)
    le40 = [p for p in pts if p["epoch"] <= 40]
    at40 = max(le40, key=lambda p: p["mAP"]) if le40 else None
    return dict(
        arch=arch,
        status="running/done",
        last_epoch=last["epoch"],
        best_mAP=best["mAP"],
        best_epoch=best["epoch"],
        best_mAP_50=best["mAP_50"],
        best_mAP_75=best["mAP_75"],
        mAP_at40=(at40["mAP"] if at40 else None),
        epoch_at40=(at40["epoch"] if at40 else None),
        points=pts,
    )


def load_speed():
    fp = os.path.join(RESULTS, "benchmark_speed.json")
    if not os.path.exists(fp):
        return {}
    with open(fp) as f:
        rows = json.load(f)
    return {r["arch"]: r for r in rows}


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    speed = load_speed()
    summary = {a: summarize(a) for a in VARIANTS}

    base = summary.get(BASELINE_ARCH, {})
    base_at40 = base.get("mAP_at40")
    base_best = base.get("best_mAP")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "map_results.json"), "w") as f:
        json.dump(dict(baseline=BASELINE_ARCH, summary=summary, speed=speed),
                  f, indent=2)

    lines = []
    lines.append("# Backbone-swap results (YOLO-World-S, COCO finetune)\n")
    lines.append(f"Baseline = **{BASELINE_ARCH}**  |  "
                 f"v8 mAP@40 = {fmt(base_at40)}  |  "
                 f"v8 best mAP = {fmt(base_best)}\n")
    lines.append("Screening = first 40 epochs of the 80e schedule (exact prefix; "
                 "mosaic on, 80e LR curve). Winners are resumed 40->80.\n")
    lines.append("| backbone | status | mAP@40 | best mAP | best ep | "
                 "AP50 | AP75 | vis params (M) | GFLOPs | FPS fp16 | vs v8@40 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(
        VARIANTS,
        key=lambda a: (summary[a].get("best_mAP") or -1),
        reverse=True,
    )
    for a in order:
        s = summary[a]
        sp = speed.get(a, {})
        delta = "-"
        if s.get("mAP_at40") is not None and base_at40 is not None:
            d = s["mAP_at40"] - base_at40
            delta = f"{d:+.3f}"
            if a == BASELINE_ARCH:
                delta = "baseline"
        lines.append(
            f"| {a} | {s.get('status','-')} | {fmt(s.get('mAP_at40'))} | "
            f"{fmt(s.get('best_mAP'))} | {s.get('best_epoch','-')} | "
            f"{fmt(s.get('best_mAP_50'))} | {fmt(s.get('best_mAP_75'))} | "
            f"{fmt(sp.get('params_vision_M'),2)} | {fmt(sp.get('gflops_vision'),1)} | "
            f"{fmt(sp.get('fps_fp16'),1)} | {delta} |"
        )
    md = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS, "map_results.md"), "w") as f:
        f.write(md)

    if not args.quiet:
        print(md)


if __name__ == "__main__":
    main()
