#!/usr/bin/env python3
"""Time and score a System-One decision backend on Zeus-shaped requests.

Two backends answer the same questions over the same state, so the numbers compare:

  jev   cloud API (TypeSafe direct, or the Vercel AI Gateway)
  laya  local model, Apache 2.0, no network

Laya's card caps `choice` at ~20 options — its options share a 256-token head
budget — and Zeus has 27 actions. `--shape` picks how that is handled:

  flat  all 27 actions in one choice, to measure the documented falloff
  hier  category first, then the action inside it (never more than 7 options)

    python3 bench.py --backend laya --shape hier --model english
    python3 bench.py --backend jev  --shape flat --compare-ollama
"""

import argparse
import json
import os
import resource
import statistics
from collections import Counter
import sys
import time

DIRECT = ("https://api.typesafe.ai/v1/systemone", "jev-latest", "TYPESAFE_API_KEY")
GATEWAY = ("https://ai-gateway.vercel.sh/typesafe/v1/systemone", "typesafe-ai/jev", "AI_GATEWAY_API_KEY")

from task import Task, load_cases, majority_baseline

TASK = Task()
ACTION_GROUPS, CATEGORIES = TASK.action_groups, TASK.categories
GROUP_OF, ALL_ACTIONS = TASK.group_of, TASK.actions
state_for = TASK.state_for
flat_questions = TASK.flat_questions
first_questions = TASK.first_questions
second_questions = TASK.second_questions


def module_mb(module):
    import io

    import torch

    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 * 1024)


def rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def unwrap(answers, key):
    a = answers.get(key, {})
    if not isinstance(a, dict):
        return a, None
    return a.get("choice", a.get("noul", a.get("score"))), a.get("confidence")


class JevBackend:
    name = "jev"

    def __init__(self, force_gateway):
        import requests

        url, model, env = GATEWAY if (force_gateway or os.getenv("AI_GATEWAY_API_KEY")) else DIRECT
        key = os.getenv(env)
        if not key:
            sys.exit(f"Set {env} (or AI_GATEWAY_API_KEY to use the Vercel gateway).")
        self.url, self.model = url, model
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        self.tokens = 0
        print(f"endpoint : {url}\nmodel    : {model}")

    def ask(self, utterance, questions):
        r = self.session.post(
            self.url,
            json={"model": self.model, "state": state_for(utterance), "questions": questions},
            timeout=15,
        )
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        self.tokens += (body.get("usage") or {}).get("input_tokens", 0)
        return body["answers"]

    def cost(self):
        return f"{self.tokens} input tokens -> ${self.tokens / 1_000_000 * 0.042:.6f}"


class OnnxModel:
    """Stands in for Agent.model so Laya's own tokenizing and decoding are reused."""

    def __init__(self, path, threads):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.sess.get_inputs()]
        self.shapes = {i.name: [d if isinstance(d, int) else None for d in i.shape]
                       for i in self.sess.get_inputs()}

    # Pad value per input. A padded token is masked out by attention_mask, and a padded
    # option slot points at position 0 with marker_mask False, so neither reaches the logits.
    PADS = {"input_ids": 0, "attention_mask": 0, "marker_pos": 0, "marker_mask": False}

    def __call__(self, *tensors):
        import torch

        feed = {}
        for name, x in zip(self.names, tensors):
            want = self.shapes[name]
            for axis, size in enumerate(want):
                if size is None or x.shape[axis] == size:
                    continue
                if axis == 0:
                    raise ValueError(
                        f"graph is fixed at {size} questions per call but got {x.shape[0]}; "
                        "a static export answers one question set, so re-export with --pad-to "
                        "for this set or use the dynamic graph")
                if x.shape[axis] > size:
                    raise ValueError(f"{name} axis {axis} is {x.shape[axis]}, above the "
                                     f"graph's fixed {size}; re-export with a larger --pad-to")
                pad = [0, 0] * (x.dim() - axis - 1)
                pad += [0, size - x.shape[axis]]
                x = torch.nn.functional.pad(x, pad, value=self.PADS[name])
            feed[name] = x.cpu().numpy()

        logits, act = self.sess.run(["logits", "act"], feed)
        n_opt = tensors[self.names.index("marker_pos")].shape[1]
        return torch.from_numpy(logits)[:, :n_opt], torch.from_numpy(act)

    def to(self, *a, **k):
        return self

    def eval(self):
        return self


class LayaBackend:
    name = "laya"

    def __init__(self, model, device=None, quantize=None, per_channel=False, onnx=None, threads=4):
        os.environ.setdefault("USE_TF", "0")  # transformers' TF probe can deadlock model construction
        from laya import Router

        self.model = model
        t0 = time.perf_counter()
        if os.path.isdir(model):
            import laya

            agent = laya.load(model, device=device)
            self.agent, self.router = agent, None
        else:
            self.router = Router(max_loaded=2, device=device)
            self.router.preload([model])  # unpreloaded checkpoints reload at ~7.4s each on CPU
            agent = self.router._agents.get(model)
            self.agent = agent
        load = time.perf_counter() - t0

        import torch

        if onnx:
            agent.model = OnnxModel(onnx, threads)
            agent.device, agent.dtype = torch.device("cpu"), torch.float32
            size = os.path.getsize(onnx) + (os.path.getsize(onnx + ".data")
                                            if os.path.exists(onnx + ".data") else 0)
            print(f"onnx     : {onnx}  ({size / 1e6:.0f} MB, {threads} threads)")

        self.quantized = quantize
        if quantize and not onnx:
            engines = torch.backends.quantized.supported_engines
            engine = "qnnpack" if "qnnpack" in engines else next(e for e in engines if e != "none")
            torch.backends.quantized.engine = engine  # defaults to "none", which fails at prepack
            # The head's TransformerEncoderLayer fast path reads tensor attrs that quantization removes.
            torch.backends.mha.set_fastpath_enabled(False)
            before = module_mb(agent.model)
            agent.model = agent.model.to("cpu").eval()
            # Quantizing the decision head destroys this model; the encoder tolerates it.
            target = agent.model if quantize.startswith("all") else agent.model.encoder
            # Per-tensor gives one scale per whole weight matrix and costs real accuracy;
            # per-channel gives one per output row.
            spec = ({torch.nn.Linear: torch.ao.quantization.per_channel_dynamic_qconfig}
                    if per_channel else {torch.nn.Linear})
            converted = torch.ao.quantization.quantize_dynamic(target, spec, dtype=torch.qint8)
            if quantize.startswith("all"):
                agent.model = converted
            else:
                agent.model.encoder = converted
            agent.device = torch.device("cpu")
            agent.dtype = torch.float32
            print(f"quantize : int8 {quantize} {'per-channel' if per_channel else 'per-tensor'} ({engine})  {before:.0f} MB -> {module_mb(agent.model):.0f} MB")
        device = next((str(p.device) for m in vars(agent).values()
                       if isinstance(m, torch.nn.Module) for p in m.parameters()), "unknown")
        print(f"model    : {model}\nload     : {load:.1f}s   (rss {rss_mb():.0f} MB, device {device}, "
              f"threads {torch.get_num_threads()})")

    def ask(self, utterance, questions):
        state = state_for(utterance)
        if self.router is None:
            return self.agent.system_one(state, questions)["answers"]
        return self.router.predict(state, questions, model=self.model)["answers"]

    def cost(self):
        return f"local — $0.00, peak rss {rss_mb():.0f} MB"


def run(backend, shape, trials, cases):
    times, hits, vision_hits, self_hits, rows = [], 0, 0, 0, []
    cat_hits = 0

    for utterance, want_action, want_vision, want_self, _amb in cases:
        durations = []
        got_action = got_vision = got_self = category = None
        conf = cat_conf = None
        want_category = GROUP_OF.get(want_action, "none")
        for _ in range(trials):
            t0 = time.perf_counter()
            if shape == "flat":
                answers = backend.ask(utterance, flat_questions())
                got_action, conf = unwrap(answers, "action")
            else:
                answers = backend.ask(utterance, first_questions())
                category, cat_conf = unwrap(answers, "category")
                if category == "none":
                    got_action, conf = "none", cat_conf
                else:
                    inner = backend.ask(utterance, second_questions(category))
                    got_action, conf = unwrap(inner, "action")
            durations.append(time.perf_counter() - t0)
            got_vision, _ = unwrap(answers, "vision")
            got_self, _ = unwrap(answers, "about_self")

        med = statistics.median(durations)
        times.append(med)
        ok = got_action == want_action
        hits += ok
        cat_hits += (category or "none") == want_category
        vision_hits += got_vision == want_vision
        self_hits += round(float(got_self or 0)) == want_self
        rows.append((utterance, want_action, got_action, conf, med, ok, category, want_category,
                     got_vision, float(got_self or 0)))
        mark = "ok  " if ok else "MISS"
        c = f"{conf:.2f}" if isinstance(conf, float) else "  - "
        stage = ""
        if shape == "hier":
            cc = f"{cat_conf:.2f}" if isinstance(cat_conf, float) else " -  "
            bad = " <-CAT" if (category or "none") != want_category else ""
            stage = f"  cat={category or 'none'}/{want_category}@{cc}{bad}"
        if not ok or len(cases) <= 30:
            print(f"  {mark} {med * 1000:6.0f}ms  conf {c}  {utterance[:34]:34s} want={want_action:11s} got={got_action:11s}{stage}")

    n = len(cases)
    if shape == "hier":
        print(f"\ncategory   {cat_hits}/{n} = {cat_hits / n:.3f}")
    for label, hit, idx in (("action", hits, 1), ("vision", vision_hits, 2), ("about_self", self_hits, 3)):
        _, base = majority_baseline(cases, idx)
        verdict = "beats" if hit / n > base else "LOSES TO"
        print(f"{label:11s} {hit}/{n} = {hit / n:.3f}   {verdict} majority baseline {base:.3f}")

    # Rare positives: accuracy hides false alarms, so score the minority class directly.
    for label, col, positive in (("vision", 2, lambda v: v != "none"), ("about_self", 3, lambda v: v >= 0.5)):
        tp = sum(1 for r, c in zip(rows, cases) if positive(c[col]) and positive(r[8 if col == 2 else 9]))
        fp = sum(1 for r, c in zip(rows, cases) if not positive(c[col]) and positive(r[8 if col == 2 else 9]))
        fn = sum(1 for r, c in zip(rows, cases) if positive(c[col]) and not positive(r[8 if col == 2 else 9]))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {label} positives: precision {prec:.3f} recall {rec:.3f}  (tp {tp} fp {fp} fn {fn})")
    print(f"\nlatency p50 {statistics.median(times) * 1000:.0f} ms   min {min(times) * 1000:.0f}   max {max(times) * 1000:.0f}")
    print(f"cost        {backend.cost()}")
    return rows


def time_ollama(utterance, model):
    """The path a decision model would replace: qwen picking actions today."""
    import requests

    prompt = (
        "Reply with JSON only: {\"actions\":[...],\"answer\":\"...\"}. "
        f"Valid actions: {', '.join(ALL_ACTIONS)}. User said: {utterance}"
    )
    t0 = time.perf_counter()
    r = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"num_ctx": 2048, "num_predict": 120, "temperature": 0}},
        timeout=120,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return dt, r.json().get("response", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("jev", "laya"), default="laya")
    ap.add_argument("--shape", choices=("flat", "hier"), default="hier")
    ap.add_argument("--model", default="english",
                    help="laya checkpoint name (english|multilingual|typed-decisions) or a local directory")
    ap.add_argument("--device", default=None, help="torch device for laya: cpu | mps | cuda")
    ap.add_argument("--onnx", default=None, help="run this ONNX graph instead of the torch model")
    ap.add_argument("--threads", type=int, default=4, help="onnxruntime intra-op threads")
    ap.add_argument("--per-channel", action="store_true", help="per-channel weight scales")
    ap.add_argument("--quantize", choices=("encoder", "all"), default=None,
                    help="int8 dynamic quantization, CPU only: encoder-only (safe) or all (includes the head)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--cases", default=None, help="labelled cases JSON")
    ap.add_argument("--include-ambiguous", action="store_true")
    ap.add_argument("--gateway", action="store_true", help="force the Vercel AI Gateway endpoint")
    ap.add_argument("--compare-ollama", action="store_true")
    ap.add_argument("--ollama-model", default="qwen2.5:0.5b")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    cases = load_cases(args.cases, args.include_ambiguous)
    print(f"backend  : {args.backend}\nshape    : {args.shape}\ncases    : {len(cases)} ({'with' if args.include_ambiguous else 'without'} ambiguous)")
    backend = JevBackend(args.gateway) if args.backend == "jev" else LayaBackend(args.model, args.device, args.quantize, args.per_channel, args.onnx, args.threads)
    print()
    rows = run(backend, args.shape, args.trials, cases)

    if args.compare_ollama:
        print(f"\ncomparing against {args.ollama_model} (the current router)...")
        try:
            durations = [time_ollama(u, args.ollama_model)[0] for u, *_ in cases[:5]]
            med = statistics.median(durations)
            ours = statistics.median([r[4] for r in rows])
            print(f"  ollama p50  {med * 1000:.0f} ms   ->  {med / ours:.1f}x slower than {args.backend}")
        except Exception as e:
            print(f"  ollama unavailable here: {e}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"backend": args.backend, "shape": args.shape, "model": args.model,
                       "rows": [{"utterance": r[0], "want": r[1], "got": r[2], "confidence": r[3],
                                 "seconds": r[4], "ok": r[5], "category": r[6], "want_category": r[7],
                                 "got_vision": r[8], "got_self": r[9],
                                 "want_vision": c[2], "want_self": c[3]}
                                for r, c in zip(rows, cases)]}, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
