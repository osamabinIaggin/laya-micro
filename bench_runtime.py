#!/usr/bin/env python3
"""Score and time a quantized Laya graph through the torch-free runtime.

This is the deployment path: onnxruntime, numpy and tokenizers, nothing else.
Runs anywhere; on a Raspberry Pi it also records thermal and under-voltage state,
because a latency number taken from a throttled board is worse than none.

Self-contained: onnxruntime, numpy and tokenizers only. Records thermal and
under-voltage state alongside every latency sample, because an edge benchmark
taken from a throttled or browning-out board is worse than no benchmark.

    python3 bench_runtime.py --threads 4 --out results.json
"""

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from runtime import LayaRuntime  # noqa: E402
from task import Task, load_cases, majority_baseline  # noqa: E402


def find_graph(directory):
    """The bundle keeps whatever name the graph was built with."""
    import glob

    found = [f for f in sorted(glob.glob(os.path.join(directory, "*.onnx")))
             if not f.endswith(".data")]
    if not found:
        raise SystemExit(f"no .onnx in {directory} — build one with scripts/make_bundle.sh")
    return found[0]


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024)


def telemetry():
    """Throttle flags, temperature and free memory — the context a number is only valid within."""
    out = {}
    for key, cmd in (("throttled", ["vcgencmd", "get_throttled"]),
                     ("temp", ["vcgencmd", "measure_temp"]),
                     ("volts", ["vcgencmd", "measure_volts"])):
        try:
            out[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=5
                                      ).stdout.strip().split("=", 1)[-1]
        except Exception:
            out[key] = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    out["mem_available_mb"] = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass
    try:
        out["load1"] = os.getloadavg()[0]
    except Exception:
        pass
    try:
        out["peak_rss_mb"] = peak_rss_mb()
    except Exception:
        pass
    return out


def decode_throttled(value):
    """vcgencmd's bitfield, spelled out."""
    if not value:
        return None
    bits = {0: "under-voltage now", 1: "arm frequency capped now", 2: "currently throttled",
            3: "soft temp limit now", 16: "under-voltage has occurred",
            17: "arm frequency capping has occurred", 18: "throttling has occurred",
            19: "soft temp limit has occurred"}
    try:
        v = int(value, 16)
    except ValueError:
        return value
    flags = [name for bit, name in bits.items() if v & (1 << bit)]
    return {"raw": value, "flags": flags or ["clean"]}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "model"))
    ap.add_argument("--onnx", default=None, help="ONNX graph (default: the one in this directory)")
    ap.add_argument("--cases", default=None, help="labelled cases JSON")
    ap.add_argument("--task", default=None, help="domain definition JSON")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--include-ambiguous", action="store_true")
    args = ap.parse_args()

    args.onnx = args.onnx or find_graph(HERE)
    before = telemetry()
    print(f"before: {before}", flush=True)
    if (before.get("throttled") or "").strip() not in ("0x0", ""):
        print(f"WARNING: board is not clean at start: {decode_throttled(before.get('throttled'))}",
              flush=True)

    task = Task(args.task)
    cases = load_cases(args.cases, args.include_ambiguous)
    qs = task.flat_questions()
    print(f"{len(cases)} cases, {len(qs['action']['criteria'])} action options, "
          f"{args.threads} threads", flush=True)

    t0 = time.perf_counter()
    rt = LayaRuntime(args.model, args.onnx, args.threads)
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s", flush=True)

    rows, times, samples = [], [], []
    for i, (utterance, want_action, want_vision, want_self, _amb) in enumerate(cases):
        state = task.state_for(utterance)
        t0 = time.perf_counter()
        ans = rt.decide(state, qs)["answers"]
        dt = time.perf_counter() - t0
        times.append(dt)
        rows.append({
            "utterance": utterance,
            "want": want_action, "got": ans["action"]["choice"],
            "confidence": ans["action"]["confidence"],
            "want_vision": want_vision, "got_vision": ans["vision"]["choice"],
            "want_self": want_self, "got_self": ans["about_self"]["noul"],
            "seconds": dt,
        })
        if i % 25 == 0:
            samples.append({"case": i, **telemetry()})
            print(f"  {i}/{len(cases)}  {dt * 1000:.0f} ms", flush=True)

    n = len(rows)
    hits = sum(r["want"] == r["got"] for r in rows)
    vhits = sum(r["want_vision"] == r["got_vision"] for r in rows)
    shits = sum(round(float(r["got_self"])) == r["want_self"] for r in rows)
    base = {f: majority_baseline(cases, i)[1]
            for i, f in ((1, "action"), (2, "vision"), (3, "about_self"))}

    after = telemetry()
    result = {
        "device": open("/proc/device-tree/model").read().strip("\x00") if
        os.path.exists("/proc/device-tree/model") else None,
        "threads": args.threads, "cases": n, "load_seconds": round(load_s, 2),
        "peak_rss_mb": peak_rss_mb(),
        "onnx_mb": round((os.path.getsize(args.onnx) +
                          (os.path.getsize(args.onnx + ".data")
                           if os.path.exists(args.onnx + ".data") else 0)) / 1e6),
        "accuracy": {
            "action": round(hits / n, 4), "vision": round(vhits / n, 4),
            "about_self": round(shits / n, 4),
        },
        "majority_baseline": {k: round(v, 4) for k, v in base.items()},
        "latency_ms": {
            "p50": round(statistics.median(times) * 1000),
            "p95": round(sorted(times)[int(0.95 * len(times))] * 1000),
            "min": round(min(times) * 1000), "max": round(max(times) * 1000),
        },
        "telemetry": {"before": before, "after": after,
                      "throttled_after": decode_throttled(after.get("throttled")),
                      "samples": samples},
        "rows": rows,
    }
    json.dump(result, open(args.out, "w"), indent=1)

    print(f"\naction      {hits}/{n} = {hits / n:.3f}   baseline {base['action']:.3f}")
    print(f"vision      {vhits}/{n} = {vhits / n:.3f}   baseline {base['vision']:.3f}")
    print(f"about_self  {shits}/{n} = {shits / n:.3f}   baseline {base['about_self']:.3f}")
    print(f"\nlatency p50 {result['latency_ms']['p50']} ms  p95 {result['latency_ms']['p95']} ms")
    print(f"throttle    {result['telemetry']['throttled_after']}")
    print(f"peak rss    {result['peak_rss_mb']} MB   (no torch, no transformers)")
    print(f"temp        {after.get('temp')}   mem avail {after.get('mem_available_mb')} MB")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
