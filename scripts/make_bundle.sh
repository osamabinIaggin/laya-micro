#!/usr/bin/env bash
# Assemble everything a target device needs, and nothing it does not.
#
# The ONNX graph carries the weights, so model.safetensors stays behind; the
# device needs only the tokenizer, the two config files, the graph and the
# torch-free runner.
#
#   scripts/make_bundle.sh laya-multilingual-en laya-micro.int8.onnx

set -euo pipefail

CKPT="${1:?usage: make_bundle.sh <pruned-checkpoint-dir> <quantized.onnx> [out-dir]}"
ONNX="${2:?usage: make_bundle.sh <pruned-checkpoint-dir> <quantized.onnx> [out-dir]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${3:-$ROOT/out/pi-bundle}"

for f in "$CKPT/tokenizer/tokenizer.json" "$CKPT/tokenizer/tokenizer_config.json" \
         "$CKPT/encoder/config.json" "$CKPT/rl_agent_config.json" "$ONNX"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

rm -rf "$OUT"
mkdir -p "$OUT/model/tokenizer" "$OUT/model/encoder" "$OUT/cases"

cp "$CKPT/tokenizer/tokenizer.json" "$CKPT/tokenizer/tokenizer_config.json" "$OUT/model/tokenizer/"
cp "$CKPT/encoder/config.json" "$OUT/model/encoder/"
cp "$CKPT/rl_agent_config.json" "$OUT/model/"
# Keep the graph's filename: an ONNX with external weights refers to the .data file
# by name, so renaming one without rewriting the other makes the model unloadable.
cp "$ONNX" "$OUT/$(basename "$ONNX")"
[ -f "$ONNX.data" ] && cp "$ONNX.data" "$OUT/$(basename "$ONNX").data"
cp "$ROOT/runtime.py" "$ROOT/task.py" "$ROOT/bench_runtime.py" "$OUT/"
cp "$ROOT/cases/zeus_cases.json" "$ROOT/cases/zeus_task.json" "$OUT/cases/"
cp "$ROOT/requirements-runtime.txt" "$OUT/"

echo "bundle: $OUT  ($(du -sh "$OUT" | cut -f1))"
find "$OUT" -type f | sed "s|$OUT/|  |" | sort
echo
echo "next: scripts/deploy_pi.sh user@host"
