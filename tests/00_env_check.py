"""Smoke test #0: verify the environment, GPU, registries, adapter, and CLIP.

Run:  source scripts/env.sh && $PY tests/00_env_check.py
(GPU check requires running outside the sandbox.)
"""
import os
import sys

YWT = os.environ.get("YWT", "/home/huanghai/yolo_world_test")
ok = True


def check(label, fn):
    global ok
    try:
        res = fn()
        print(f"[ OK ] {label}: {res}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")


import torch  # noqa: E402

print("=" * 70)
print("python:", sys.version.split()[0])
check("torch", lambda: f"{torch.__version__} (cuda build {torch.version.cuda})")
check("cuda available", lambda: f"{torch.cuda.is_available()} "
      f"[{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'}]")


def _gpu_matmul():
    if not torch.cuda.is_available():
        return "skipped (no gpu visible)"
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    return f"matmul on {(a @ b).device} ok"


check("gpu compute", _gpu_matmul)

import mmcv, mmengine, mmdet, mmyolo  # noqa: E402
check("openmmlab", lambda: f"mmcv {mmcv.__version__} | mmengine {mmengine.__version__} | "
      f"mmdet {mmdet.__version__} | mmyolo {mmyolo.__version__}")

import ultralytics, transformers  # noqa: E402
check("ultralytics", lambda: ultralytics.__version__)
check("transformers", lambda: transformers.__version__)

# Registration: import model packages, then probe the registry
import mmyolo.models  # noqa: E402,F401
import mmdet.models  # noqa: E402,F401


def _import_yw():
    import yolo_world  # noqa: F401
    import yolo_world.models  # noqa: F401
    return "imported"


check("import yolo_world", _import_yw)

from mmyolo.registry import MODELS  # noqa: E402


def _reg(name):
    return lambda: ("registered" if MODELS.get(name) is not None else "MISSING")


for n in ["YOLOv8CSPDarknet", "YOLOWorldDetector", "MultiModalYOLOBackbone",
          "YOLOWorldPAFPN", "YOLOWorldHeadModule", "HuggingCLIPLanguageBackbone"]:
    check(f"registry:{n}", _reg(n))

import adapters  # noqa: E402,F401
check("registry:UltralyticsYOLOBackbone", _reg("UltralyticsYOLOBackbone"))


def _clip():
    from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPTextConfig
    d = os.path.join(YWT, "data/weights/clip-vit-base-patch32")
    tok = AutoTokenizer.from_pretrained(d)
    cfg = CLIPTextConfig.from_pretrained(d)
    m = CLIPTextModelWithProjection.from_pretrained(d, config=cfg)
    n = tok(["a photo of a cat", "dog"], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = m(**n)
    return f"loaded, proj_dim={m.config.projection_dim}, text_embeds={tuple(out.text_embeds.shape)}"


check("CLIP text encoder (offline)", _clip)

print("=" * 70)
print("RESULT:", "ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
