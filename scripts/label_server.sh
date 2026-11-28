#!/usr/bin/env bash
# Run LabelMe on this headless VM and use it from a browser.
#   LabelMe draws into Qt's built-in VNC display (no real screen needed), and noVNC shows
#   that display in a web page. Both listen on 127.0.0.1 only; VS Code forwards the port.
# Usage: scripts/label_server.sh [batch folder under data/]   (default: self_batch_2)
#        SIZE=1920x1080 scripts/label_server.sh                (window size, default 1600x900)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
V="$REPO/.venv-labelme"
BATCH="${1:-self_batch_2}"
IMAGES="$REPO/data/$BATCH"
LABELS="$REPO/data/labels/$BATCH"      # LabelMe .json files go here, one per photo
SIZE="${SIZE:-1600x900}"
VNC_PORT=5901 WEB_PORT=6080

[ -d "$IMAGES" ] || { echo "no folder $IMAGES" >&2; exit 1; }
mkdir -p "$LABELS"

# bindlocal.so keeps Qt's password-less VNC server on loopback (it would otherwise listen on all interfaces)
# OSAM_ONNX_PROVIDERS: run the click-to-outline model on the CPU (it otherwise picks the VM's virtual GPU)
OSAM_ONNX_PROVIDERS=CPUExecutionProvider LD_PRELOAD="$V/lib/bindlocal.so" QT_QPA_PLATFORM="vnc:size=$SIZE:port=$VNC_PORT" \
  "$V/bin/labelme" "$IMAGES" --output "$LABELS" --labels floor,not_floor,ignore --validate-label exact --flags no_floor,exclude \
  --config "{display_label_popup: false}" &   # no label pop-up: with no window manager it can open off-screen and look frozen
LABELME=$!
"$V/bin/websockify" --web "$V/novnc" "127.0.0.1:$WEB_PORT" "127.0.0.1:$VNC_PORT" >/dev/null 2>&1 &
WEB=$!
trap 'kill $LABELME $WEB 2>/dev/null' EXIT INT TERM

echo "Labelling $IMAGES -> $LABELS"
echo "Open in your browser:  http://localhost:$WEB_PORT/vnc.html?autoconnect=1&resize=scale"
echo "(If it doesn't load, add port $WEB_PORT in VS Code's PORTS tab.)  Ctrl+C here to stop."
wait $LABELME
