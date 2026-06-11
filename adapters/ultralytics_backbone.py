# Copyright (c) 2026. YOLO-World backbone-swap experiment.
"""Adapter that exposes an **Ultralytics** YOLO backbone (v8/v9/v10/v11/v12/v26)
as an **MMYOLO** backbone usable inside YOLO-World's ``MultiModalYOLOBackbone``.

Why this is needed
------------------
YOLO-World builds its vision backbone from the MMYOLO ``MODELS`` registry and the
neck (``YOLOWorldPAFPN``) consumes three feature maps at strides 8/16/32 (P3/P4/P5).
MMYOLO only ships v5/v6/v7/v8. v9..v26 live in the ``ultralytics`` package, which is a
different framework. This adapter instantiates an Ultralytics model, keeps **only the
backbone** part (everything before the head section in the model YAML), and returns the
P3/P4/P5 feature maps - which are exactly what the YOLO-World neck expects.

No YOLO-World or MMYOLO source is modified: configs pull this class in via
``custom_imports`` and set ``model.backbone.image_model.type='UltralyticsYOLOBackbone'``.
"""
from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


def _build_ultralytics_model(arch: str, checkpoint: str = None, num_classes: int = 80):
    """Return an Ultralytics ``DetectionModel`` (nn.Module).

    * ``checkpoint`` given -> load COCO-pretrained weights (architecture taken from
      the checkpoint's own YAML).
    * else -> build randomly-initialised architecture from ``<arch>.yaml`` (bundled
      with the ultralytics package, no download required).
    """
    import os
    import warnings

    from ultralytics import YOLO

    if checkpoint and os.path.exists(checkpoint):
        src = checkpoint  # COCO-pretrained weights (arch read from the ckpt)
    else:
        if checkpoint:
            warnings.warn(f"UltralyticsYOLOBackbone: checkpoint '{checkpoint}' not "
                          f"found; using random init from {arch}.yaml")
        src = f"{arch}.yaml"
    model = YOLO(src, task="detect").model  # DetectionModel
    return model.float()


@MODELS.register_module()
class UltralyticsYOLOBackbone(BaseModule):
    """Wrap an Ultralytics backbone and emit multi-scale features for YOLO-World.

    Args:
        arch (str): ultralytics architecture stem, e.g. ``'yolov8s'``, ``'yolov9s'``,
            ``'yolov10s'``, ``'yolo11s'``, ``'yolo12s'``, ``'yolo26s'``.
        checkpoint (str, optional): absolute path to an ultralytics ``*.pt`` for
            COCO-pretrained backbone initialisation. If ``None``, the architecture is
            randomly initialised.
        out_strides (tuple): strides of the feature maps to return (default 8/16/32).
        num_classes (int): nc used to build the (unused) detection head.
        freeze_stages (int): freeze the first N backbone layers (-1 = none).
        norm_eval (bool): keep BatchNorm layers in eval mode during training.
    """

    def __init__(self,
                 arch: str,
                 checkpoint: str = None,
                 out_strides: Sequence[int] = (8, 16, 32),
                 num_classes: int = 80,
                 freeze_stages: int = -1,
                 norm_eval: bool = False,
                 probe_size: int = 256,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.arch = arch
        self.out_strides = tuple(out_strides)
        self.freeze_stages = freeze_stages
        self.norm_eval = norm_eval

        full_model = _build_ultralytics_model(arch, checkpoint, num_classes)
        yaml_cfg = getattr(full_model, "yaml", None)
        if not yaml_cfg or "backbone" not in yaml_cfg:
            raise RuntimeError(f"Could not read backbone definition for arch={arch}")
        n_backbone = len(yaml_cfg["backbone"])
        # The parsed nn.Sequential lays out backbone modules first (indices
        # 0..n_backbone-1), then the head. Keep only the backbone portion.
        self.backbone_layers = full_model.model[:n_backbone]

        # Discover which layer outputs correspond to the requested strides by a
        # dummy forward pass (architecture-agnostic: works for CSPDarknet, GELAN, etc.)
        self.out_layer_indices, self.out_channels = self._probe(probe_size)
        self._freeze_stages()

    # ------------------------------------------------------------------ helpers
    def _forward_layers(self, x: Tensor) -> List[Tensor]:
        """Run the backbone layers honouring each module's ``.f`` (from) wiring,
        saving every intermediate output (backbones are short, so this is cheap)."""
        y: List[Tensor] = []
        for m in self.backbone_layers:
            if getattr(m, "f", -1) != -1:
                f = m.f
                x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]
            x = m(x)
            y.append(x)
        return y

    def _probe(self, size: int) -> Tuple[List[int], List[int]]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            outs = self._forward_layers(torch.zeros(1, 3, size, size))
        strides = [size // o.shape[-2] for o in outs]
        sel_idx: List[int] = []
        for s in self.out_strides:
            matches = [i for i, st in enumerate(strides) if st == s]
            if not matches:
                raise RuntimeError(
                    f"arch={self.arch}: no backbone layer with stride {s}; "
                    f"observed strides={strides}")
            sel_idx.append(matches[-1])  # last (deepest) feature at this stride
        sel_ch = [outs[i].shape[1] for i in sel_idx]
        if was_training:
            self.train()
        return sel_idx, sel_ch

    def _freeze_stages(self):
        if self.freeze_stages < 0:
            return
        for i, m in enumerate(self.backbone_layers):
            if i <= self.freeze_stages:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    # -------------------------------------------------------------------- mmengine
    def init_weights(self):
        # Weights come from ultralytics (pretrained .pt or its own init); do not
        # let BaseModule re-initialise and clobber them.
        return

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs = self._forward_layers(x)
        return tuple(outs[i] for i in self.out_layer_indices)


if __name__ == "__main__":
    # Introspection helper: print the P3/P4/P5 channel counts for a given arch,
    # which are needed to configure the YOLO-World neck. Uses random YAML init
    # (no weight download).  Usage: python ultralytics_backbone.py yolo11s yolo26s
    import sys

    archs = sys.argv[1:] or ["yolov8s", "yolov9s", "yolov10s", "yolo11s", "yolo12s", "yolo26s"]
    for a in archs:
        try:
            bb = UltralyticsYOLOBackbone(arch=a)
            x = torch.zeros(1, 3, 640, 640)
            feats = bb(x)
            shapes = [tuple(f.shape) for f in feats]
            print(f"{a:10s} out_channels={bb.out_channels}  strides={bb.out_strides}  "
                  f"layer_idx={bb.out_layer_indices}  shapes={shapes}")
        except Exception as e:  # noqa: BLE001
            print(f"{a:10s} FAILED: {type(e).__name__}: {e}")
