#!/usr/bin/env python3
"""Block-wise int8 weight-only quantization of an exported Laya ONNX graph.

One scale per block of 64 weights, rather than one per whole matrix — per-tensor
scales cost this model several points of accuracy.

    python3 quantize_onnx.py laya.onnx --out laya.int8.onnx
"""

import argparse
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act"]


def graph_mb(path):
    ext = f"{path}.data"
    return (os.path.getsize(path) + (os.path.getsize(ext) if os.path.exists(ext) else 0)) / 1e6


def strip_value_info(src, dst):
    """The exporter leaves inferred shapes that onnx's checker later contradicts."""
    m = onnx.load(src)
    del m.graph.value_info[:]
    onnx.save_model(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=os.path.basename(dst) + ".data", convert_attribute=False)
    return dst


def session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=None)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    out = args.out or args.src.replace(".onnx", f".int{args.bits}.onnx")
    stripped = args.src.replace(".onnx", ".stripped.onnx")

    print(f"source      {graph_mb(args.src):.0f} MB")
    strip_value_info(args.src, stripped)

    t0 = time.perf_counter()
    quant = MatMulNBitsQuantizer(
        onnx.load(stripped),
        block_size=args.block_size,
        is_symmetric=True,
        bits=args.bits,
        algo_config=DefaultWeightOnlyQuantConfig(
            block_size=args.block_size, is_symmetric=True, bits=args.bits),
    )
    quant.process()
    quant.model.save_model_to_file(out, use_external_data_format=True)
    print(f"quantized   {graph_mb(out):.0f} MB  "
          f"({graph_mb(args.src) / graph_mb(out):.1f}x smaller, {time.perf_counter() - t0:.0f}s, "
          f"int{args.bits} block={args.block_size})")

    # Same inputs through both graphs: agreement matters more than raw drift.
    sess32 = session(args.src, args.threads)
    shapes = {i.name: i.shape for i in sess32.get_inputs()}
    def dim(shapes, name, axis, default):
        v = shapes[name][axis]
        return v if isinstance(v, int) else default

    n_item = dim(shapes, "input_ids", 0, 3)
    n_seq = dim(shapes, "input_ids", 1, 259)
    n_opt = dim(shapes, "marker_pos", 1, 28)
    feed = {
        "input_ids": np.random.randint(5, 1000, (n_item, n_seq), dtype=np.int64),
        "attention_mask": np.ones((n_item, n_seq), dtype=np.int64),
        "marker_pos": np.stack([np.sort(np.random.choice(np.arange(1, n_seq), n_opt, replace=False))
                                for _ in range(n_item)]).astype(np.int64),
        "marker_mask": np.ones((n_item, n_opt), dtype=bool),
        "qtype": np.zeros((n_item,), dtype=np.int64),
    }
    print(f"  input shapes: { {k: v for k, v in shapes.items()} }")

    ref = sess32.run(OUTPUT_NAMES, feed)
    sess8 = session(out, args.threads)
    got = sess8.run(OUTPUT_NAMES, feed)
    drift = float(np.abs(got[0] - ref[0]).max())
    agree = int(got[0].argmax(-1).flatten()[0]) == int(ref[0].argmax(-1).flatten()[0])
    print(f"  max logit drift {drift:.4f}, argmax {'matches' if agree else 'DIFFERS'}")

    for label, sess in (("fp32", sess32), (f"int{args.bits}", sess8)):
        sess.run(OUTPUT_NAMES, feed)
        t0 = time.perf_counter()
        for _ in range(args.trials):
            sess.run(OUTPUT_NAMES, feed)
        print(f"  {label}: {(time.perf_counter() - t0) / args.trials * 1000:.0f} ms/call "
              f"({args.threads} threads)")


if __name__ == "__main__":
    main()
