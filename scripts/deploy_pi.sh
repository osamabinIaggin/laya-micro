#!/usr/bin/env bash
# Ship the bundle to a Pi over a link that drops, and run the benchmark detached.
#
# rsync resumes partial transfers, so a dropped connection costs only the retry.
# The benchmark runs under nohup and writes to a file, so it survives losing the
# shell entirely — reconnect later and collect.
#
# Build the bundle first with scripts/make_bundle.sh.
#
#   scripts/deploy_pi.sh pi@raspberrypi.local
#   scripts/deploy_pi.sh pi@raspberrypi.local --status
#   scripts/deploy_pi.sh pi@raspberrypi.local --fetch

set -uo pipefail

HOST="${1:?usage: deploy_pi.sh user@host [--status|--fetch|--run]}"
ACTION="${2:---all}"
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/out/pi-bundle"
REMOTE="laya-micro"
SSH_OPTS=(-o ConnectTimeout=20 -o ServerAliveInterval=5 -o ServerAliveCountMax=3
          -o StrictHostKeyChecking=accept-new)

say() { printf '\n== %s ==\n' "$1"; }

# Apple ships openrsync (2.6.9-compatible), which has no --append-verify.
rsync_flags() {
  local flags="--partial --progress"
  # --append-verify resumes without re-reading; openrsync lacks it, --inplace is the fallback.
  if rsync --append-verify --version >/dev/null 2>&1; then
    flags="$flags --append-verify"
  else
    flags="$flags --inplace"
  fi
  printf '%s' "$flags"
}

push() {
  if [ ! -d "$BUNDLE" ]; then
    echo "no bundle at $BUNDLE" >&2
    echo "build one first:  scripts/make_bundle.sh <pruned-checkpoint> <quantized.onnx>" >&2
    return 1
  fi
  local flags; flags="$(rsync_flags)"
  say "syncing $(du -sh "$BUNDLE" | cut -f1) to $HOST:$REMOTE  [$flags]"
  for attempt in $(seq 1 20); do
    # shellcheck disable=SC2086
    rsync -az $flags --timeout=60 \
          -e "ssh ${SSH_OPTS[*]}" "$BUNDLE/" "$HOST:$REMOTE/"
    local rc=$?
    case $rc in
      0)  echo "sync complete after $attempt attempt(s)"; return 0 ;;
      1|2|4) echo "rsync usage/protocol error ($rc) — not retrying" >&2; return $rc ;;
      *)  echo "attempt $attempt interrupted (rc=$rc); resuming in 10s" ;;
    esac
    sleep 10
  done
  echo "sync failed after 20 attempts" >&2
  return 1
}

setup() {
  say "installing runtime dependencies"
  ssh "${SSH_OPTS[@]}" "$HOST" "LC_ALL=C bash -lc '
    cd $REMOTE || exit 1
    python3 -c \"import onnxruntime, numpy, tokenizers\" 2>/dev/null && {
      echo already installed; exit 0; }
    pip install --break-system-packages --quiet onnxruntime numpy tokenizers
  '"
}

run() {
  say "starting benchmark (detached)"
  ssh "${SSH_OPTS[@]}" "$HOST" "LC_ALL=C bash -lc '
    cd $REMOTE || exit 1
    rm -f results.json bench.log
    nohup python3 bench_runtime.py --threads \${THREADS:-4} --out results.json \
      > bench.log 2>&1 &
    echo started pid \$!
  '"
  echo "poll with: $0 $HOST --status"
}

status() {
  ssh "${SSH_OPTS[@]}" "$HOST" "LC_ALL=C bash -lc '
    cd $REMOTE 2>/dev/null || { echo \"not deployed\"; exit 1; }
    pgrep -f bench_runtime.py >/dev/null && echo \"RUNNING\" || echo \"not running\"
    echo \"--- last lines ---\"; tail -12 bench.log 2>/dev/null
    [ -f results.json ] && echo \"--- results.json present ---\"
  '"
}

fetch() {
  say "fetching results"
  mkdir -p "$(dirname "$BUNDLE")/pi-results"
  rsync -avz --timeout=60 -e "ssh ${SSH_OPTS[*]}" \
    "$HOST:$REMOTE/results.json" "$HOST:$REMOTE/bench.log" \
    "$(dirname "$BUNDLE")/pi-results/" && echo "into out/pi-results/"
}

case "$ACTION" in
  --status) status ;;
  --fetch)  fetch ;;
  --run)    run ;;
  --all)    push && setup && run ;;
  *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac
