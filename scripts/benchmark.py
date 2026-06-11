#!/usr/bin/env python
"""Static + speed benchmark for the backbone-swapped YOLO-World variants.

Reports, per config:
  * parameters: full model (incl. frozen CLIP text encoder) and the deployable
    vision path (backbone+neck+head, what actually runs per image once text
    embeddings are cached),
  * GFLOPs of the vision path at 640x640 (best-effort via mmengine FlopAnalyzer),
  * GPU latency / FPS at batch=1, 640x640, for fp32 and fp16 (text features are
    pre-extracted/cached, i.e. open-vocab "reparameterized" inference - the
    realistic deployment path).

mAP is measured separately by training/validation (CocoMetric); this script
covers model size + speed. A checkpoint is optional (speed is weight-agnostic).

Usage:
    python scripts/benchmark.py --all
    python scripts/benchmark.py configs/yoloworld_s_yolo11s_coco.py [--ckpt x.pth]
"""
import argparse
import glob
import json
import os
import time

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmyolo.registry import MODELS

ROOT = "/home/huanghai/yolo_world_test"
TEXTS_JSON = f"{ROOT}/data/texts/coco_class_texts.json"
RESULTS = f"{ROOT}/results"


class VisionPath(nn.Module):
    """Image-only inference path with cached text features (deployment view)."""

    def __init__(self, model, txt_feats):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.bbox_head
        self.txt = txt_feats

    def forward(self, x):
        feats = self.backbone.forward_image(x)
        feats = self.neck(feats, self.txt)
        return self.head(feats, self.txt, None)


def load_texts():
    with open(TEXTS_JSON) as f:
        raw = json.load(f)
    return [syn[0] for syn in raw]  # 80 class strings


def count_params(module):
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def measure_latency(vp, device, half, iters=100, warmup=20):
    vp = vp.to(device).eval()
    if half:
        vp = vp.half()
    x = torch.rand(1, 3, 640, 640, device=device)
    x = x.half() if half else x
    for _ in range(warmup):
        vp(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        vp(x)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0, 1.0 / dt  # ms, fps


def gflops_vision(vp, device):
    try:
        from mmengine.analysis import FlopAnalyzer
        vp = vp.to(device).eval().float()
        x = torch.rand(1, 3, 640, 640, device=device)
        fa = FlopAnalyzer(vp, x)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total() / 1e9
    except Exception as e:  # noqa: BLE001
        return f"n/a ({type(e).__name__})"


def bench_one(path, ckpt=None):
    name = os.path.basename(path)
    cfg = Config.fromfile(path)
    init_default_scope(cfg.get("default_scope", "mmyolo"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MODELS.build(cfg.model)
    model.init_weights()
    if ckpt and os.path.exists(ckpt):
        from mmengine.runner import load_checkpoint
        load_checkpoint(model, ckpt, map_location="cpu")
    model = model.to(device).eval()

    texts = load_texts()
    model.reparameterize([texts])  # cache text features (batch of 1)
    txt_feats = model.text_feats

    vp = VisionPath(model, txt_feats)

    p_full = count_params(model)
    p_text = count_params(model.backbone.text_model)
    # deployable vision path = image backbone + neck + head (text encoder is
    # frozen and its embeddings are precomputed, so it is NOT counted here).
    p_vision = (count_params(model.backbone.image_model)
                + count_params(model.neck)
                + count_params(model.bbox_head))
    gf = gflops_vision(vp, device)

    # speed (rebuild a fresh cast each time to avoid dtype bleed)
    ms32, fps32 = measure_latency(VisionPath(model.float(), txt_feats.float()),
                                  device, half=False)
    ms16, fps16 = measure_latency(VisionPath(model, txt_feats.half()),
                                  device, half=True)

    return dict(
        name=name,
        arch=cfg.backbone_arch,
        channels=list(cfg.neck_in_channels),
        params_full_M=round(p_full / 1e6, 2),
        params_vision_M=round(p_vision / 1e6, 2),
        params_text_M=round(p_text / 1e6, 2),
        gflops_vision=round(gf, 1) if isinstance(gf, float) else gf,
        ms_fp32=round(ms32, 2), fps_fp32=round(fps32, 1),
        ms_fp16=round(ms16, 2), fps_fp16=round(fps16, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    if args.all:
        configs = sorted(glob.glob(f"{ROOT}/configs/yoloworld_s_*.py"))
    else:
        assert args.config, "pass a config or --all"
        configs = [args.config]

    rows = []
    for c in configs:
        r = bench_one(c, args.ckpt)
        rows.append(r)
        print(f"[bench] {r['arch']:9s} "
              f"params(full/vision/text)={r['params_full_M']}/{r['params_vision_M']}/"
              f"{r['params_text_M']}M  GFLOPs={r['gflops_vision']}  "
              f"fp16={r['ms_fp16']}ms ({r['fps_fp16']} FPS)  "
              f"fp32={r['ms_fp32']}ms ({r['fps_fp32']} FPS)")

    if args.all:
        os.makedirs(RESULTS, exist_ok=True)
        with open(f"{RESULTS}/benchmark_speed.json", "w") as f:
            json.dump(rows, f, indent=2)
        hdr = ("| arch | P3/P4/P5 | params full (M) | params vision (M) | "
               "GFLOPs (vision) | fp16 ms | fp16 FPS | fp32 ms | fp32 FPS |\n"
               "|---|---|---|---|---|---|---|---|---|\n")
        lines = [hdr]
        for r in rows:
            lines.append(
                f"| {r['arch']} | {r['channels']} | {r['params_full_M']} | "
                f"{r['params_vision_M']} | {r['gflops_vision']} | {r['ms_fp16']} | "
                f"{r['fps_fp16']} | {r['ms_fp32']} | {r['fps_fp32']} |\n")
        with open(f"{RESULTS}/benchmark_speed.md", "w") as f:
            f.writelines(lines)
        print(f"\n[bench] wrote {RESULTS}/benchmark_speed.{{json,md}}")


if __name__ == "__main__":
    main()
