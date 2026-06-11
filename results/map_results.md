# Backbone-swap results (YOLO-World-S, COCO finetune)

Baseline = **yolov8s**  |  v8 mAP@40 = 0.421  |  v8 best mAP = 0.428

Screening = first 40 epochs of the 80e schedule (exact prefix; mosaic on, 80e LR curve). Winners are resumed 40->80.

| backbone | status | mAP@40 | best mAP | best ep | AP50 | AP75 | vis params (M) | GFLOPs | FPS fp16 | vs v8@40 |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo12s | running/done | 0.462 | 0.470 | 71 | 0.643 | 0.509 | 17.05 | 30.0 | 64.6 | +0.041 |
| yolo26s | running/done | 0.455 | 0.469 | 77 | 0.639 | 0.512 | 17.04 | 28.6 | 83.8 | +0.034 |
| yolo11s | running/done | 0.452 | 0.458 | 69 | 0.628 | 0.497 | 17.04 | 28.6 | 85.0 | +0.031 |
| yolov10s | running/done | 0.444 | 0.448 | 65 | 0.619 | 0.488 | 11.57 | 16.0 | 84.8 | +0.023 |
| yolov8s | running/done | 0.421 | 0.428 | 69 | 0.591 | 0.464 | 12.76 | 16.4 | 95.6 | baseline |
| yolov9s | running/done | 0.415 | 0.415 | 40 | 0.580 | 0.452 | 7.09 | 13.3 | 58.8 | -0.006 |
