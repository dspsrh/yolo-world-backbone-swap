#!/usr/bin/env bash
# Download all external assets that are NOT on the Cursor sandbox allowlist.
# MUST be run with full_network (or all) permission. Uses scripts/fetch.sh
# (AliDNS + curl --resolve) because this box has no working DNS.
#
# Priority order (so smoke-testing can start before the 18 GB train set lands):
#   1. mmcv prebuilt wheel (needed to build the venv)
#   2. CLIP text model (needed by every YOLO-World variant)
#   3. COCO annotations + val2017 (needed for eval + quick smoke training)
#   4. COCO train2017 (18 GB, the long pole; only needed for full training)
set -uo pipefail
ROOT=/home/huanghai/yolo_world_test
FETCH="$ROOT/scripts/fetch.sh"
W="$ROOT/data/weights"
COCO="$ROOT/data/coco"
CLIP="$W/clip-vit-base-patch32"
HFBASE="https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main"
CDN="cdn-lfs.hf-mirror.com cdn-lfs.huggingface.co"

log() { echo "[$(date +%H:%M:%S)] $*"; }

dl()  { bash "$FETCH" "$1" "$2" ${3:-} ; }

unzip_to() {  # zip, dest, sentinel-dir
  local zip="$1" dest="$2" sentinel="$3"
  if [ -d "$sentinel" ] && [ "$(ls -A "$sentinel" 2>/dev/null | head -1)" ]; then
    log "already extracted: $sentinel"; return 0; fi
  log "unzip $zip -> $dest"; mkdir -p "$dest"; unzip -q -o "$zip" -d "$dest" && log "unzip done"
}

stage="${1:-all}"

if [ "$stage" = "mmcv" ] || [ "$stage" = "all" ]; then
  log "=== mmcv wheel ==="
  dl "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl" \
     "$W/wheels/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl"
fi

if [ "$stage" = "clip" ] || [ "$stage" = "all" ]; then
  log "=== CLIP text model (hf-mirror) ==="
  for f in config.json merges.txt vocab.json tokenizer.json tokenizer_config.json \
           special_tokens_map.json preprocessor_config.json; do
    dl "$HFBASE/$f" "$CLIP/$f"
  done
  # model weights (LFS). openai/clip-vit-base-patch32 on the mirror ships
  # pytorch_model.bin (no safetensors -> that path 404s with a 15-byte stub).
  rm -f "$CLIP/model.safetensors"
  dl "$HFBASE/pytorch_model.bin" "$CLIP/pytorch_model.bin" "$CDN"
  # sanity: a valid checkpoint is hundreds of MB, not a 404 stub
  sz=$(stat -c%s "$CLIP/pytorch_model.bin" 2>/dev/null || echo 0)
  if [ "$sz" -lt 100000000 ]; then echo "[clip] ERROR: pytorch_model.bin too small ($sz bytes)"; fi
fi

if [ "$stage" = "coco-small" ] || [ "$stage" = "all" ]; then
  log "=== COCO annotations + val2017 ==="
  dl "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$COCO/annotations_trainval2017.zip"
  unzip_to "$COCO/annotations_trainval2017.zip" "$COCO" "$COCO/annotations"
  dl "http://images.cocodataset.org/zips/val2017.zip" "$COCO/val2017.zip"
  unzip_to "$COCO/val2017.zip" "$COCO" "$COCO/val2017"
fi

if [ "$stage" = "coco-train" ] || [ "$stage" = "all" ]; then
  log "=== COCO train2017 (18 GB) ==="
  dl "http://images.cocodataset.org/zips/train2017.zip" "$COCO/train2017.zip"
  unzip_to "$COCO/train2017.zip" "$COCO" "$COCO/train2017"
fi

log "=== download_data.sh ($stage) complete ==="
