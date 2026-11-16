#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PATCH="$ROOT/live_prediction_patch"
STAMP="$(date +%Y%m%d_%H%M%S)"

for f in "$ROOT/server.py" "$ROOT/dashboard/index.html" \
         "$PATCH/server.py" "$PATCH/dashboard/index.html" \
         "$PATCH/src/live_predict.py" "$PATCH/src/validate_live_prediction.py"; do
  if [ ! -f "$f" ]; then
    echo "Required file missing: $f" >&2
    exit 1
  fi
done

cp "$ROOT/server.py" "$ROOT/server.py.before_live_prediction_${STAMP}.bak"
cp "$ROOT/dashboard/index.html" "$ROOT/dashboard/index.html.before_live_prediction_${STAMP}.bak"
cp "$PATCH/server.py" "$ROOT/server.py"
cp "$PATCH/dashboard/index.html" "$ROOT/dashboard/index.html"
cp "$PATCH/src/live_predict.py" "$ROOT/src/live_predict.py"
cp "$PATCH/src/validate_live_prediction.py" "$ROOT/src/validate_live_prediction.py"

echo "WaveLab live prediction patch applied."
echo "Backups: server.py.before_live_prediction_${STAMP}.bak"
echo "         dashboard/index.html.before_live_prediction_${STAMP}.bak"
