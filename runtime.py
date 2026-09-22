#!/usr/bin/env python3
"""Run an exported Laya graph with no PyTorch and no transformers.

A Raspberry Pi needs onnxruntime, numpy and tokenizers — about 30 MB of wheels
instead of two gigabytes. The tokenizing and decoding here are ports of
laya.common and laya.Agent.system_one; outputs match the reference
implementation, which `bench.py --onnx` verifies against labelled cases.

    from runtime import LayaRuntime
    rt = LayaRuntime("laya-multilingual-en", "laya.int8.onnx")
    rt.decide({"utterance": "come here"}, {"go": {"type": "noul",
                                                  "instructions": "a movement was requested"}})
"""

import json
import math
import os

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act"]


def serialize_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_criterion(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q):
    """Option texts in label-index order. Noul is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
                for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    f, tr = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(f) if f not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(tr) if tr not in (None, "") else "yes, the statement holds"),
    ]


def to_internal(qdef):
    crit = qdef.get("criteria")
    if qdef["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    return {"t": qdef["type"], "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": crit}


def confidence_from_probs(p, k):
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype, k):
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


class LayaRuntime:
    def __init__(self, checkpoint_dir, onnx_path, threads=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer", "tokenizer.json"))
        enc = json.load(open(os.path.join(checkpoint_dir, "encoder", "config.json")))
        cfg = json.load(open(os.path.join(checkpoint_dir, "rl_agent_config.json")))

        self.pad_id = enc["pad_token_id"]
        self.cls_id = enc["cls_token_id"]
        self.sep_id = enc["sep_token_id"]
        self.mask_id = enc["mask_token_id"]
        self.mask_tok = self.tok.id_to_token(self.mask_id)
        self.max_len = cfg.get("max_len", 512)
        self.head_max_len = cfg.get("head_max_len", 192)
        self.temperature = cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = cfg.get("temperature_by_options", {})

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    def _encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def build_sequence(self, state, q):
        """[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]"""
        opts = render_options(q)
        ins = str(q["ins"]).replace(self.mask_tok, " ")
        head_ids = self._encode("%s question: %s" % (q["t"], ins))
        opt_ids = [[self.mask_id] + self._encode(" " + o.replace(self.mask_tok, " "))[:48] for o in opts]

        budget = self.head_max_len - sum(len(o) for o in opt_ids)
        if budget < 16:
            per = max(4, (self.head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            budget = self.head_max_len - sum(len(o) for o in opt_ids)
        head_ids = head_ids[: max(8, budget)]

        ids = [self.cls_id] + head_ids + [self.sep_id]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(self.sep_id)

        room = max(0, self.max_len - len(ids) - 1)
        ids += self._encode(serialize_state(state).replace(self.mask_tok, " "))[:room] + [self.sep_id]
        return ids[: self.max_len], [m for m in markers if m < self.max_len]

    def _collate(self, items):
        n = len(items)
        length = max(len(i["ids"]) for i in items)
        kmax = max(len(i["markers"]) for i in items)
        ids = np.full((n, length), self.pad_id, dtype=np.int64)
        att = np.zeros((n, length), dtype=np.int64)
        mpos = np.zeros((n, kmax), dtype=np.int64)
        mmask = np.zeros((n, kmax), dtype=bool)
        for i, it in enumerate(items):
            ids[i, : len(it["ids"])] = it["ids"]
            att[i, : len(it["ids"])] = 1
            k = len(it["markers"])
            mpos[i, :k] = it["markers"]
            mmask[i, :k] = True
        return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos,
                "marker_mask": mmask, "qtype": np.array([i["qtype"] for i in items], dtype=np.int64)}

    def decide(self, state, questions):
        """Same shape as laya.Agent.system_one: {model, answers, usage}."""
        qids = list(questions)
        internal = [to_internal(questions[q]) for q in qids]
        items = [{"ids": (s := self.build_sequence(state, q))[0], "markers": s[1],
                  "qtype": QTYPES[q["t"]]} for q in internal]

        batch = self._collate(items)
        logits, act = self.sess.run(OUTPUT_NAMES, {n: batch[n] for n in INPUT_NAMES})
        act = np.exp(act - act.max(-1, keepdims=True))  # the graph emits act logits
        act = act / act.sum(-1, keepdims=True)

        answers = {}
        for r, (qid, q) in enumerate(zip(qids, internal)):
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = logits[r, :k] / max(1e-3, float(scale))
            p = np.exp(z - z.max())
            p = p / p.sum()
            conf = round(confidence_from_probs(p, k), 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                                "confidence": conf, "action": ext}
            elif q["t"] == "score":
                answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                                "confidence": conf, "action": ext}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4),
                                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                                "action": ext}

        return {"model": "laya-micro", "answers": answers,
                "usage": {"input_tokens": int(batch["attention_mask"].sum()), "output_tokens": 0}}


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("onnx")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--utterance", default="come over here and tell me what you see")
    args = ap.parse_args()

    t0 = time.perf_counter()
    rt = LayaRuntime(args.checkpoint, args.onnx, args.threads)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    questions = {
        "moving": {"type": "noul", "instructions": "The user asked the robot to move."},
        "vision": {"type": "choice", "instructions": "Does answering require the camera?",
                   "criteria": {"yes": "needs to look", "no": "can answer without looking"}},
    }
    t0 = time.perf_counter()
    out = rt.decide({"utterance": args.utterance}, questions)
    print(f"decided in {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(json.dumps(out["answers"], indent=2))
