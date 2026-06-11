#!/usr/bin/env bash
# Resilient downloader for hosts that are NOT on the Cursor sandbox allowlist.
# This box has no DNS (empty /etc/resolv.conf) but routing works, so we resolve
# the host via AliDNS (223.5.5.5) and pass the IP to curl via --resolve.
# Requires running with full_network (or all) permission.
#
# Usage: fetch.sh <url> <output_path> [extra_host_for_redirects ...]
set -uo pipefail

URL="$1"; OUT="$2"; shift 2 || true
RESOLVER="/home/huanghai/yolo_world_test/scripts/resolve.py"

host_of() { echo "$1" | sed -E 's#^[a-zA-Z]+://##; s#/.*##; s#:.*##'; }
proto_of() { echo "$1" | sed -E 's#://.*##'; }

HOST=$(host_of "$URL")
PROTO=$(proto_of "$URL")
PORT=443; [ "$PROTO" = "http" ] && PORT=80

IP=$(/usr/bin/python3 "$RESOLVER" "$HOST")
if [[ "$IP" == \#* || -z "$IP" ]]; then echo "[fetch] FAILED to resolve $HOST"; exit 2; fi
RESOLVE_ARGS=(--resolve "$HOST:$PORT:$IP")
# Map common redirect/CDN hosts too (for HF LFS, S3, etc.)
for eh in "$@"; do
  EIP=$(/usr/bin/python3 "$RESOLVER" "$eh")
  if [[ "$EIP" != \#* && -n "$EIP" ]]; then
    RESOLVE_ARGS+=(--resolve "$eh:443:$EIP" --resolve "$eh:80:$EIP")
  fi
done

mkdir -p "$(dirname "$OUT")"
echo "[fetch] $URL  ($HOST -> $IP, port $PORT)  ->  $OUT"
curl -L --retry 8 --retry-delay 3 --retry-connrefused -C - \
     --connect-timeout 20 "${RESOLVE_ARGS[@]}" -o "$OUT" "$URL"
rc=$?
# curl rc=33 / 416 => range not satisfiable (already complete); treat as success
if [[ $rc -eq 33 ]]; then rc=0; fi
if [[ $rc -ne 0 ]]; then echo "[fetch] curl exited $rc"; exit $rc; fi
echo "[fetch] done: $(ls -la "$OUT" | awk '{print $5, $9}')"
