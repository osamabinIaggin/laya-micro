#!/usr/bin/env python3
"""Export a Laya checkpoint to ONNX and quantize it to int8.

The Pi already runs onnxruntime for YOLOv8n, so an ONNX graph avoids putting
torch and transformers on the robot. Run this on a workstation and copy the
resulting .onnx to the Pi.

    python3 export_onnx.py --model multilingual --out laya_multilingual
"""

import argparse
import json
import os
import time

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch

import laya
from laya.common import QTYPES, build_sequence, collate_items, render_options

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act"]

SAMPLE_STATE = {
    "robot": "Zeus, a four-legged PiCrawler robot with a camera, mic and speaker",
    "utterance": "come over here and tell me what you see",
    "sees": ["person"],
    "posture": "standing",
}
from bench import flat_questions

# Zeus's real question set: three questions of differing option counts in one batch,
# so the exported graph keeps both the item and option dimensions dynamic.
SAMPLE_QUESTIONS = flat_questions()


def sample_batch(agent, pad_to=None):
    """A real batch built through Laya's own preprocessing, so shapes are honest.

    pad_to fixes the sequence length: a static graph sidesteps the dynamic Range
    ops that defeat onnxruntime's shape inference during quantization.
    """
    items = []
    for spec in SAMPLE_QUESTIONS.values():
        q = agent._to_internal(spec)
        seq, markers = build_sequence(agent.tok, SAMPLE_STATE, q,
                                      agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192))
        assert len(markers) == len(render_options(q))
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    b = collate_items([items], agent.tok.pad_token_id)
    if pad_to:
        n = b["input_ids"].shape[1]
        if n > pad_to:
            raise SystemExit(f"sample needs {n} tokens; raise --pad-to above {pad_to}")
        pad = pad_to - n
        b["input_ids"] = torch.nn.functional.pad(b["input_ids"], (0, pad), value=agent.tok.pad_token_id)
        b["attention_mask"] = torch.nn.functional.pad(b["attention_mask"], (0, pad), value=0)
    return tuple(b[k].to("cpu") for k in INPUT_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="multilingual", help="english | multilingual | typed-decisions")
    ap.add_argument("--out", default=None, help="output basename (default laya_<model>)")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--pad-to", type=int, default=0, help="fixed sequence length (0 = dynamic)")
    args = ap.parse_args()

    base = args.out or f"laya_{os.path.basename(args.model.rstrip('/'))}"
    fp32_path = f"{base}.onnx"

    print(f"loading {args.model} ...")
    if os.path.isdir(args.model):
        agent = laya.load(args.model, device="cpu")
    else:
        router = laya.Router(device="cpu")
        router.preload([args.model])  # the checkpoints ship as subfolders of the base repo
        agent = router._agents[args.model]
    if hasattr(agent.model, "encoder"):
        agent.model.encoder.config.reference_compile = False  # torch.compile path is not exportable
    repo = agent.cfg.get("encoder", args.model)
    model = agent.model.to("cpu").eval()
    inputs = sample_batch(agent, args.pad_to or None)
    for name, t in zip(INPUT_NAMES, inputs):
        print(f"  {name:16s} {tuple(t.shape)} {t.dtype}")

    with torch.no_grad():
        ref_logits, ref_act = model(*inputs)

    # Export under grad: nn.TransformerEncoderLayer only takes its unexportable fused fast
    # path under no_grad, and the legacy fallback freezes reshape dims into the graph.
    if args.pad_to:
        shapes = None  # fully static: more MatMuls keep constant weights, so more quantizes
    else:
        items = torch.export.Dim("items")
        seq = torch.export.Dim("seq", min=8)
        nopt = torch.export.Dim("nopt", min=2)
        shapes = ({0: items, 1: seq}, {0: items, 1: seq},
                  {0: items, 1: nopt}, {0: items, 1: nopt}, {0: items})
    kind = f"static {tuple(inputs[0].shape)}" if args.pad_to else "dynamic"
    print(f"\nexporting fp32 -> {fp32_path}  (dynamo, opset {args.opset}, {kind})")
    torch.onnx.export(
        model, inputs, fp32_path, dynamo=True, optimize=True, opset_version=args.opset,
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES, dynamic_shapes=shapes,
    )
    print("  exported")

    import onnxruntime as ort

    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    print("  graph inputs:", {i.name: i.shape for i in sess.get_inputs()})
    out = sess.run(OUTPUT_NAMES, {n: t.numpy() for n, t in zip(INPUT_NAMES, inputs)})
    print(f"  max logit drift vs torch: {np.abs(out[0] - ref_logits.numpy()).max():.5f}")

    with open(f"{base}.meta.json", "w") as f:
        json.dump({"repo": repo, "opset": args.opset, "input_names": INPUT_NAMES,
                   "output_names": OUTPUT_NAMES, "max_len": agent.cfg.get("max_len", 512),
                   "head_max_len": agent.cfg.get("head_max_len", 192)}, f, indent=2)
    print(f"\nwrote {base}.meta.json — now run: quantize_onnx.py {fp32_path}")


if __name__ == "__main__":
    main()
