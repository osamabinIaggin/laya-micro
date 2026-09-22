# Reproducing laya-micro

Every number this repository reports comes from the four scripts in it, run over the
165 labelled cases in `cases/zeus_cases.json`. This document walks the whole pipeline
from a clean machine, states what each step should print, and ends with how to point
the pipeline at a domain that is not robot commands.

Where a number has not been measured, it says PENDING or TODO rather than guessing.

## What you are reproducing

Apple M3 Pro, CPU only, 165 labelled commands, `--shape flat`, multilingual checkpoint:

| build | disk | peak RSS | action accuracy |
| --- | --- | --- | --- |
| stock multilingual fp32 (torch) | 1228 MB | 3230 MB | 0.661 |
| vocab-pruned fp32 (torch) | 275 MB | 1061 MB | 0.661 |
| pruned + ONNX fp32 | 551 MB | 1516 MB | 0.661 |
| pruned + ONNX int8 block-64 | 371 MB | 1312 MB | 0.667 |

Peak RSS is the whole harness process measured with `ru_maxrss`. The ONNX rows are
higher than the pruned-torch row because the harness loads the torch checkpoint for
tokenizing and decoding before swapping the ONNX session in for the forward pass.

These are Apple M3 Pro, CPU numbers. Latency does not transfer between CPU architectures
— see [Not measured yet](#not-measured-yet).

## 1. Prerequisites

- **Python 3.10 or newer.** Measured on 3.12.13.
- **About 4 GB of free disk** for the Hugging Face checkpoint cache, plus roughly 1 GB
  more for the pruned checkpoint and the ONNX graphs you will write.
- **A CPU machine.** Everything here is CPU-only; `--device cpu` is passed explicitly so
  a Mac does not silently run on MPS and change the numbers.
- **An isolated virtualenv.** This is not boilerplate advice. `laya` pins `torch>=2.0.0`
  and `transformers>=4.45.0`, and torch alone is over a gigabyte; installing it into a
  system Python will fight with whatever else that Python serves. On a robot this matters
  more, because the Pi's system `numpy` and `onnxruntime` are already spoken for by the
  vision stack, and a pip upgrade of either one breaks it.

## 2. Install

```bash
git clone https://github.com/osamabinIaggin/laya-micro.git
cd laya-micro
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install laya onnx onnxruntime onnxscript
```

`laya` pulls `torch`, `transformers`, `safetensors`, `huggingface_hub` and `numpy`.
`onnx`, `onnxruntime` and `onnxscript` are only needed for the export and quantize steps;
`onnxscript` in particular is what the dynamo exporter uses, and the export fails without it.

The versions this pipeline was measured with:

```text
Python 3.12.13
laya 0.3.4              torch 2.14.0           transformers 5.17.0
tokenizers 0.23.2       safetensors 0.8.0      huggingface_hub 1.32.0
numpy 2.5.3             onnx 1.23.0            onnx-ir 1.0.0
onnxscript 0.7.2        onnxruntime 1.30.0
```

Export `USE_TF=0` for every command below:

```bash
export USE_TF=0
```

`transformers` probes for TensorFlow at import time, and its abseil runtime can deadlock
model construction. The scripts set this themselves with `os.environ.setdefault`, but if
you import them from your own code, set it before the first `transformers` import.

## 3. Get the checkpoints

**The gotcha:** the three Laya checkpoints are not separate Hugging Face repositories.
They are *subfolders* of `convaiinnovations/laya`:

```text
convaiinnovations/laya
├── (repo root)     laya                  ModernBERT-large, 421M, 512 ctx
├── multilingual/   laya-multilingual     mmBERT-base, 322M, 1024 ctx, 100+ languages
└── typed-decisions/laya-typed-decisions  421M, specialist
```

So `laya.load("convaiinnovations/laya-multilingual")` goes to the network and fails.
`Router.preload(["multilingual"])` resolves the cached subfolder, which is what `bench.py`
and `export_onnx.py` do. To pre-download just the multilingual checkpoint:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("convaiinnovations/laya", allow_patterns=["multilingual/*"]))
PY
```

`prune_vocab.py` calls the same `snapshot_download`, so you can skip this and let the
prune step fetch it.

**Note on paths:** `--cases` defaults to `cases/zeus_cases.json`. Every command below passes
it explicitly anyway, so the commands keep working if you move the file.

## 4. Baseline: the stock multilingual checkpoint

```bash
python bench.py \
  --backend laya --model multilingual --shape flat --device cpu \
  --cases cases/zeus_cases.json
```

Expected (`...` marks values that vary by machine and are not part of the reported results):

```text
backend  : laya
shape    : flat
cases    : 165 (without ambiguous)
model    : multilingual
load     : 24.5s   (rss ... MB, device cpu, threads ...)

action      109/165 = 0.661   beats majority baseline 0.291
vision      139/165 = 0.842   LOSES TO majority baseline 0.885
about_self  80/165 = 0.485   LOSES TO majority baseline 0.903
  vision positives: precision ... recall ...  (tp ... fp ... fn ...)
  about_self positives: precision 0.137 recall 0.812  (tp ... fp 82 fn ...)

latency p50 ... ms   min ...   max ...
cost        local — $0.00, peak rss 3230 MB
```

Read that output carefully, because it is the point of the harness:

- **Action selection is genuinely useful.** 0.661 against a majority baseline of 0.291 is
  2.3x better than always guessing the most common action.
- **Vision and about_self lose to a constant "no".** 0.842 against a 0.885 baseline, and
  0.485 against a 0.903 baseline. A classifier that answered "no vision needed, not about
  the robot" every single time would score higher on both.
- **`about_self` is a threshold problem, not a broken model.** Recall 0.812 with precision
  0.137 means it finds the true positives and then fires on 82 cases that are not.

The file carries 173 cases; 8 are flagged `ambiguous` and excluded, leaving the 165 used
everywhere in this document. `--include-ambiguous` puts them back and changes every
denominator, so do not compare those runs against the table above.

## 5. Prune the vocabulary

mmBERT-base spends 196,608,000 of its 321,908,998 parameters — 61.1% — on a
256,000 x 768 embedding table covering 100+ languages. An English-only robot touches a few
thousand of those rows. This step slices the table to the tokens the domain uses and
rewrites the tokenizer to match. **No transformer weight changes**, so in-domain answers
are bit-identical.

```bash
python prune_vocab.py \
  --subfolder multilingual \
  --out laya-multilingual-en \
  --seed-top 4096 \
  --cases cases/zeus_cases.json
```

Expected:

```text
vocabulary 256000 -> 15188 (5.9%)
tokenizer parity: 2805/2805 strings identical
encoder.embeddings.tok_embeddings.weight (256000, 768) -> (15188, 768)
wrote laya-multilingual-en  (275 MB)
```

The parity line is the load-bearing one. The tokenizer is the Gemma-2 BPE with 580,604
merge rules, and a kept token is useless unless every merge that produces it also survives:
for each kept token you must keep both parents of every merge producing it, transitively,
or BPE cannot reassemble the token and words tokenize differently than they did during
training. The script closes the keep-set over those merges, then re-encodes every corpus
string through both tokenizers and compares, modulo the id remapping. It refuses to touch
the weights if a single string disagrees:

```text
N of 2805 strings tokenize differently after pruning
```

If you see that, widen the corpus or raise `--seed-top` (see
[Adapting to another domain](#10-adapting-to-another-domain)).

### Verify the pruned checkpoint

```bash
python bench.py \
  --backend laya --model ./laya-multilingual-en --shape flat --device cpu \
  --cases cases/zeus_cases.json
```

Expected — identical accuracy on all three tasks, a quarter of the disk, a third of the RAM:

```text
model    : ./laya-multilingual-en
load     : 9.6s   (rss ... MB, device cpu, threads ...)

action      109/165 = 0.661   beats majority baseline 0.291
vision      139/165 = 0.842   LOSES TO majority baseline 0.885
about_self  80/165 = 0.485   LOSES TO majority baseline 0.903
...
cost        local — $0.00, peak rss 1061 MB
```

Checkpoint 1228 MB -> 275 MB, peak RSS 3230 MB -> 1061 MB, load 24.5 s -> 9.6 s, and the
same 109/165, 139/165, 80/165.

### Scope note

**This only pays off on the multilingual checkpoint.** The English checkpoint's embedding
table is 50,368 x 1024 = 51.6M parameters, only 12.2% of its 421M, so slicing it buys
comparatively little. Prune the multilingual checkpoint; do not expect the same win from
`--subfolder ''`.

## 6. Export to ONNX

```bash
python export_onnx.py \
  --model ./laya-multilingual-en \
  --out laya-micro \
  --opset 18
```

Expected:

```text
loading ./laya-multilingual-en ...
  input_ids        (3, ...) torch.int64
  attention_mask   (3, ...) torch.int64
  marker_pos       (3, ...) torch.int64
  marker_mask      (3, ...) torch.bool
  qtype            (3,) torch.int64

exporting fp32 -> laya-micro.onnx  (dynamo, opset 18, dynamic)
  exported
  graph inputs: {...}
  max logit drift vs torch: ...

wrote laya-micro.meta.json — now run: quantize_onnx.py laya-micro.onnx
```

This writes `laya-micro.onnx` (plus a `laya-micro.onnx.data` sidecar when the exporter
puts weights in external data) and `laya-micro.meta.json`. The pruned fp32 ONNX graph is
551 MB on disk.

Three things about this export are not negotiable, and all three are already in the script:

1. **`dynamo=True`.** The legacy TorchScript exporter takes an `nn.MultiheadAttention`
   fallback that freezes reshape dimensions into the graph. You will not notice at export
   time; it surfaces later, during quantization, as a shape-inference error. Static shapes
   do not work around it. Only the dynamo exporter does.
2. **`model.encoder.config.reference_compile = False`.** ModernBERT's `torch.compile`
   path is not exportable.
3. **Export under grad.** `nn.TransformerEncoderLayer` only takes its unexportable fused
   fast path under `no_grad`, so the export deliberately does not wrap the call.

And one about the sample batch: it contains **three** questions, not one. With a single
item `torch.export` specialises the batch dimension to 1, and the resulting graph can then
only ever answer one question per call. Keep at least two.

### Static-shape export

`--pad-to` fixes the sequence length instead of leaving it dynamic. It was measured and
rejected, and is documented here so nobody spends an afternoon rediscovering it.

At `--pad-to 384`, the quantized static graph came out at 373 MB against the dynamic
graph's 371 MB — no smaller — and roughly twice as slow, 989 ms against 580 ms per
three-question call on an M3 Pro, because every call pads to the full length whatever the
input actually needs. Its accuracy was never measured, because there was no reason to
deploy it.

A static export also fixes the **item** dimension, so the graph only answers the exact
question set it was exported for, and callers must pad their own inputs. `bench.py` does
not do that padding, so a static graph needs its own runner.

If you want to try it anyway, pick a length at least as long as your longest rendered
prompt and no longer than the checkpoint's context. If the sample batch does not fit, the
script refuses:

```text
sample needs 259 tokens; raise --pad-to above 256
```

Measure one process at a time. Two benchmarks running concurrently compete for memory and
report a lower peak RSS than either would alone.

## 7. Quantize

```bash
python quantize_onnx.py laya-micro.onnx \
  --out laya-micro.int8.onnx \
  --block-size 64 --bits 8 --threads 4
```

Expected:

```text
source      551 MB
quantized   371 MB  (...x smaller, ...s, int8 block=64)
  input shapes: {...}
  max logit drift ..., argmax matches
  fp32: ... ms/call (4 threads)
  int8: ... ms/call (4 threads)
```

The script deletes `graph.value_info` before quantizing, writes the stripped graph beside
the source as `laya-micro.stripped.onnx`, and then runs `MatMulNBitsQuantizer` with
`block_size=64, is_symmetric=True, bits=8`.

**Granularity is the whole story here.** On the same 165 cases, action accuracy:

| quantization | action accuracy | vs fp32 |
| --- | --- | --- |
| fp32 baseline | 0.661 | — |
| torch dynamic int8, per-tensor, whole model | 0.594 | -6.7 points |
| torch dynamic int8, per-tensor, encoder only | 0.594 | -6.7 points |
| torch dynamic int8, per-channel | 0.624 | -3.7 points |
| ONNX MatMulNBits, block-wise, `block_size=64` | 0.667 | **+0.6 points** |

One scale per whole weight matrix costs about seven points. One scale per 64 weights costs
nothing — it comes out one case *better* than fp32. Note also that excluding the decision
head from per-tensor quantization changed nothing at all: 0.594 either way. The common
explanation that "the head is sensitive" is not what is happening for per-tensor
quantization of this model; what matters is how finely the scales are allocated, not which
modules you quantize.

Reproduce those torch-side rows with:

```bash
python bench.py --model ./laya-multilingual-en --shape flat --device cpu \
  --cases cases/zeus_cases.json --quantize all                    # 0.594
python bench.py --model ./laya-multilingual-en --shape flat --device cpu \
  --cases cases/zeus_cases.json --quantize encoder                # 0.594
python bench.py --model ./laya-multilingual-en --shape flat --device cpu \
  --cases cases/zeus_cases.json --quantize encoder --per-channel  # 0.624
```

TODO: the recorded per-channel run does not note whether it used `--quantize encoder` or
`--quantize all`. The two per-tensor runs were identical, so the scope is unlikely to
matter, but confirm it before citing the scope.

## 8. Benchmark the ONNX builds

```bash
# pruned + ONNX fp32 — 551 MB on disk
python bench.py --backend laya --model ./laya-multilingual-en \
  --onnx laya-micro.onnx --shape flat --threads 4 \
  --cases cases/zeus_cases.json

# pruned + ONNX int8 block-64 — 371 MB on disk
python bench.py --backend laya --model ./laya-multilingual-en \
  --onnx laya-micro.int8.onnx --shape flat --threads 4 \
  --cases cases/zeus_cases.json
```

Expected, respectively:

```text
onnx     : laya-micro.onnx  (551 MB, 4 threads)
action      109/165 = 0.661   beats majority baseline 0.291
cost        local — $0.00, peak rss 1516 MB
```

```text
onnx     : laya-micro.int8.onnx  (371 MB, 4 threads)
action      110/165 = 0.667   beats majority baseline 0.291
cost        local — $0.00, peak rss 1312 MB
```

Against the stock checkpoint that is 3.3x smaller on disk, 2.5x less RAM, and accuracy
marginally better than where it started.

`--threads` sets onnxruntime's intra-op thread count; leave it at 4 to compare against
these numbers.

## 9. The comparisons behind the caveats

### Flat vs hierarchical questions

Laya's model card says keep `choice` questions under about 20 options, because the options
share a fixed token budget (`head_max_len`). Zeus has 27 actions plus `none` = 28 options,
which renders to 259 tokens against a `head_max_len` of 256 — saturated before any state
is added. The documented fix is to split into a hierarchy. We measured it:

```bash
python bench.py --model ./laya-multilingual-en --shape flat --device cpu --cases cases/zeus_cases.json
python bench.py --model ./laya-multilingual-en --shape hier --device cpu --cases cases/zeus_cases.json
python bench.py --model english --shape flat --device cpu --cases cases/zeus_cases.json
python bench.py --model english --shape hier --device cpu --cases cases/zeus_cases.json
```

| checkpoint | flat (28 options) | hierarchical (category, then max 7) |
| --- | --- | --- |
| multilingual | 0.661 | 0.182 |
| english | 0.406 | 0.364 |

**The advice inverts by checkpoint.** For multilingual, the hierarchy's category stage
collapses and takes the whole pipeline with it. For English, flat is still ahead, by less.
Measure it on your own checkpoint and your own option set before splitting anything.

`--shape hier` prints an extra `category  X/165 = ...` line so you can see which stage
failed. TODO: category-stage accuracy is not recorded in this document.

### Cloud Jev, for comparison

`bench.py` can score the closed TypeSafe Jev API through the same harness:

```bash
export TYPESAFE_API_KEY=...        # or AI_GATEWAY_API_KEY for the Vercel gateway
python bench.py --backend jev --shape flat --cases cases/zeus_cases.json
```

Without a key it exits with:

```text
Set TYPESAFE_API_KEY (or AI_GATEWAY_API_KEY to use the Vercel gateway).
```

This path bills per request and needs the network. It is entirely optional; nothing in the
tables above depends on it.

### Against the LLM it would replace

`--compare-ollama` times the current qwen2.5:0.5b router on the first five cases and prints
the ratio. It needs Ollama listening on `127.0.0.1:11434`; without it the harness prints
`ollama unavailable here: ...` and carries on.

## 10. Adapting to another domain

Nothing above is specific to robots except the labels. To point the pipeline at your own
decisions, you change three things.

### 10.1 Your labelled cases

`cases/zeus_cases.json` is a JSON array. One object per case:

```json
{
  "utterance": "come over here",
  "action": "forward",
  "vision": "none",
  "about_self": 0,
  "ambiguous": false
}
```

- `utterance` — the input string.
- `action` — the correct `choice` label, which must be a key of your action set.
- `vision` — the correct label for the side `choice` question.
- `about_self` — 0 or 1, the correct answer to the `noul` question.
- `ambiguous` — `true` for cases where honest annotators disagree. They are excluded from
  headline numbers unless you pass `--include-ambiguous`. Flag them rather than deleting
  them; it keeps the exclusion visible.

Pass yours with `--cases path/to/your_cases.json` to both `bench.py` and `prune_vocab.py`.
**Use the same file for both** — the keep-set is built from the cases, so pruning against
one set and scoring against another quietly guarantees out-of-domain tokens.

**Label enough cases, and label them in their real proportions.** An early 14-case balanced
sample of this exact task scored vision 0.929 and about_self 0.857 and looked excellent.
On the realistic 165-case distribution both fall below a constant "no". Small balanced
samples flatter rare-class performance badly. That is why the harness prints a majority-class
baseline next to every accuracy, and precision/recall on the rare positive class — read them
every run, not just the accuracy.

### 10.2 Your decision surface

The questions live in `cases/zeus_task.json`, loaded at `bench.py` import (override the path
with `LAYA_MICRO_TASK`):

- `ACTION_GROUPS` — your labels, grouped, as `{group: {label: description}}`. The
  descriptions are the criteria the model actually reads, so write them as descriptions of
  the case, not as restatements of the label.
- `CATEGORIES` — the group descriptions, used only by `--shape hier`.
- `SIDE_QUESTIONS` — the questions asked alongside the main one. The `vision` entry is a
  `choice`; the `about_self` entry is a `noul`, a calibrated probability that a statement is
  true. Replace, drop, or add to these — `run()` scores whatever `vision` and `about_self`
  resolve to, so if you rename them, rename them in `run()` too.
- `BASE_STATE` and `state_for()` — the state handed to the model with every question. Keep
  it small; it shares the context with the rendered options.

Watch the option budget. Count your options, render them, and compare against the
checkpoint's `head_max_len` before you trust a flat question — 28 options came to 259
tokens against a budget of 256 here.

### 10.3 Your keep-set corpus

`prune_vocab.corpus()` imports `bench` and renders **every string the model will ever see**:
each case's utterance, the JSON state, the vision label, and every question's instructions
and criteria — plus all printable ASCII and the integers 0 to 1000 as headroom.

This is the part to get right, because the parity check only proves that *the corpus you
supplied* tokenizes identically. Text outside it still encodes — byte-fallback tokens are
always kept, so nothing becomes `<unk>` — but it may tokenize differently than it did in
training, and that is exactly the failure this technique is supposed to avoid.

So:

- Add every source of domain text to `corpus()`: product names, place names, user fields,
  anything a template can interpolate.
- Raise `--seed-top` if your inputs range wider than your cases. Ids are frequency-ordered,
  so `--seed-top N` keeps the N most frequent tokens regardless of your corpus. It costs
  768 floats per row and buys real headroom.
- Re-read the parity line every run. It is the guarantee.

Then run the same pipeline: prune, bench the pruned torch checkpoint against the unpruned
one (accuracy must be identical, not merely close), export, quantize, bench again.

## 11. Troubleshooting

**`[ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (772) vs (256)`**
Raised during quantization, caused during export. The legacy TorchScript exporter's
`MultiheadAttention` fallback froze reshape dimensions into the graph. Re-export with
`dynamo=True`. Switching to static shapes does **not** work around this; only the dynamo
exporter does.

**`quantized::linear_prepack NoQEngine`**
`torch.backends.quantized.engine` defaults to `"none"`. Set it to `qnnpack` before calling
`quantize_dynamic`. `bench.py` does this for you.

**`function object has no attribute device`** (every quantized inference dies)
The `TransformerEncoderLayer` fast path reads tensor attributes that quantization removes.
Call `torch.backends.mha.set_fastpath_enabled(False)`. `bench.py` does this for you.

**Export succeeds but the graph only answers one question per call**
The export sample had a single item, so `torch.export` specialised the batch dimension to 1.
Give it at least two items.

**`onnxruntime.quantization.quant_pre_process` asserts inside `_infer_Range`**
On a dynamic export it trips its own shape inference. On a static export it instead fails to
reload `optimized.onnx`, because the external-data file is not carried into its temp
directory. It is a dead end here either way — skip it, delete `graph.value_info`, and
quantize directly, which is what `quantize_onnx.py` does.

**Model construction hangs at import**
`transformers` probes for TensorFlow at import and its abseil runtime can deadlock. Run with
`USE_TF=0`.

**`laya.load` goes to the network for a checkpoint you already downloaded**
The checkpoints are subfolders of `convaiinnovations/laya`, not standalone repo ids. Use
`Router.preload(["multilingual"])`, or pass a local directory path.

**ModernBERT export fails somewhere inside `torch.compile`**
Set `model.encoder.config.reference_compile = False` before exporting.

**`N of 2805 strings tokenize differently after pruning`**
The keep-set is not closed over the merges that build some token, or your corpus grew past
what you pruned for. Re-run with a wider corpus or a larger `--seed-top`. The script stops
before writing weights, so nothing is corrupted.

**`sample needs N tokens; raise --pad-to above 512`**
Your rendered sample is longer than the static length you asked for. Raise `--pad-to`, or
shorten the state and options.

**`Set TYPESAFE_API_KEY (or AI_GATEWAY_API_KEY to use the Vercel gateway).`**
Only the `--backend jev` path needs a key. Everything in this document runs locally.

## Not measured yet

- **Other CPU architectures.** Every number in this document is
  Apple M3 Pro, CPU. Apple Silicon latency does not predict ARM latency, and nothing here
  should be read as a Pi figure.
  The only published Pi 5 data point for a Laya ONNX build is navopw's, from 2026-09-20:
  English checkpoint, 234 ms for a single question, 1015 ms for four questions, RSS
  1076.7 MiB, 4 threads, ONNX Runtime 1.30.0. That is a different build — fp32 compute with
  fp16 vocabulary storage, unpruned — so it is context, not a comparison.
- **Latency for any laya-micro build.** The harness prints `latency p50 / min / max` and
  `ms/call`, but only accuracy, disk and peak RSS were recorded. TODO.
- **Category-stage accuracy for `--shape hier`.** TODO; only the end-to-end action
  accuracies (0.182 multilingual, 0.364 English) are recorded.
- **`tp`/`fn` counts for the precision/recall lines.** Only `about_self` precision 0.137,
  recall 0.812 and its 82 false positives were written down. TODO.
