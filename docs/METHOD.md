# Method

laya-micro shrinks a [Laya](https://github.com/NandhaKishorM/laya) decision model until it fits
alongside a robot's speech, vision and LLM stack on a single-board computer. It does this with two
independent reductions and one evaluation harness:

1. **Vocabulary pruning** — slice the multilingual embedding table down to the tokens a domain
   actually uses. Exactly lossless in-domain, and by far the larger win.
2. **Block-wise int8** — quantize the ONNX graph's matmul weights with one scale per 64 weights.
   Measured lossless here; coarser scale granularities are not.
3. **A benchmark that scores correctness, not agreement** — 165 human-labelled robot commands with
   majority-class baselines reported on every run.

This document explains how each works and what it costs. For the numbers alone, see the README.

---

## 0. What the numbers mean

Every measurement below was taken on an **Apple M3 Pro, CPU only**, against **165 labelled commands**
in `cases/zeus_cases.json` (8 further cases are flagged `ambiguous` and excluded). "Accuracy" without
qualification means action-selection accuracy on the flat 28-option question.

Three things are worth stating before any of it:

- **Apple Silicon latency does not predict ARM latency.** Raspberry Pi 5 numbers for laya-micro are
  *unmeasured on other architectures* — not slow, not fast.
- **Accuracy is only meaningful against a baseline.** The classes are imbalanced, so `bench.py`
  prints the majority-class baseline beside every score:

  | task | majority baseline | Laya (multilingual, pruned, flat) | verdict |
  | --- | --- | --- | --- |
  | action | 0.291 | 0.661 | 2.3x the baseline — genuinely useful |
  | vision | 0.885 | 0.842 | **loses to always answering "no"** |
  | about_self | 0.903 | 0.485 | **far worse than always answering "no"** |

- **Small balanced samples flatter rare-class performance badly.** An early 14-case balanced sample
  scored vision 0.929 and about_self 0.857 and looked excellent. On the realistic 165-case
  distribution both fall below a constant "no". The `about_self` failure is a decision-threshold
  problem rather than a broken model — recall 0.812, precision 0.137, 82 false positives — but it is
  a failure either way, and the repository reports it on every run.

---

## 1. Vocabulary pruning

### 1.1 Where the parameters are

The multilingual checkpoint is mmBERT-base: hidden size 768, 22 layers, a 256,000-token vocabulary
covering 100+ languages.

```text
embedding table   256,000 x 768  =  196,608,000 parameters
whole model                      =  321,908,998 parameters
                                    ---------------------
embedding share   196,608,000 / 321,908,998 = 0.6108  ->  61.1%
```

Six out of every ten parameters exist to spell languages an English-only robot will never hear. The
rest of the model — the part that actually decides anything — is:

```text
321,908,998 - 196,608,000 = 125,300,998 parameters
```

**This only pays off on the multilingual checkpoint.** The English checkpoint is ModernBERT-large
with a 50,368-token vocabulary at hidden size 1024:

```text
50,368 x 1024 = 51,576,832 parameters
51,576,832 / 421,000,000 = 0.1225  ->  12.2%
```

Pruning a table that is 12.2% of the model is not worth the machinery. Pruning one that is 61.1% is.

### 1.2 What a domain actually touches

The keep set is built from a corpus of every string the robot can put in front of the model:
utterances, the rendered JSON state, the vision labels, and every question — instructions plus each
option's label and description, in the exact form Laya renders them. Printable ASCII and the integers
0–1000 are added as headroom for whatever the speech recogniser spells.

Measured on Zeus's corpus (2,805 strings):

```text
distinct tokens the corpus touches        491
```

491 rows out of 256,000. Keeping only those would be reckless — a single unseen word would fall back
to bytes — so the keep set is widened deliberately:

| source | count | why |
| --- | --- | --- |
| corpus tokens | 491 | what the domain provably uses |
| `--seed-top N` lowest ids | 8,192 | BPE ids are frequency-ordered, so the low ids are the common English subwords |
| added/special tokens | 249 | `<pad>`, `<bos>`, `<eos>`, `<mask>`, and the checkpoint's own additions |
| byte-fallback tokens `<0x..>` | 255 | any UTF-8 input stays encodable without `<unk>` |

Union: **8,395 tokens**. And that is where a naive implementation quietly breaks.

### 1.3 Why a naive slice corrupts tokenization

BPE does not look tokens up. It *builds* them: start from bytes, then repeatedly apply the
highest-priority merge rule whose two halves are adjacent. The Gemma-2 BPE that Laya's multilingual
checkpoint uses has **580,604 merge rules**.

So a token being present in the vocabulary is not sufficient for it to ever be produced. If any
merge on the path that assembles it has lost a parent, the merge is gone, and the tokenizer
decomposes the word some other way.

This is not theoretical. Slicing to the 8,395-token union above and filtering the merges to those
whose parts survive gives:

```text
naive slice: 751 of 2805 corpus strings tokenize differently
```

Over a quarter of the corpus, including the plainest command in it:

```text
string   : 'come over here'
original : ['▁come', '▁over', '▁here']
naive    : ['▁c', '<0x6F>', 'me', '▁', '<0x6F>', '<0x76>', 'e', '<0x72>', '▁', '<0x68>', 'e', '<0x72>', 'e']
```

`▁come` is *in* the sliced vocabulary. It is simply unreachable: the intermediate tokens that merge
into it were dropped, so BPE shreds the word into byte fallbacks. The model still runs. It still
returns an answer with a confidence attached. It is now reading input that looks nothing like its
training distribution, and nothing in the pipeline says so.

### 1.4 Merge closure

The fix is to close the keep set over the merges that produce its members. For every kept token,
keep **both parents of every merge that produces it**, and recurse:

```python
child2parents = {}
for a, b in merges:
    child2parents.setdefault(a + b, []).append((a, b))

keep = {inv[i] for i in ids}
stack = list(keep)
while stack:
    for a, b in child2parents.get(stack.pop(), ()):
        for q in (a, b):
            if q not in keep:
                keep.add(q)
                stack.append(q)
```

A token can be produced by several different merges; all of them are followed, because the tokenizer
picks by merge priority, not by convenience. The closure is transitive — parents drag in their own
parents — and terminates because every merge strictly shortens.

Measured cost of closure on Zeus's corpus:

```text
union before closure      8,395
after merge closure      15,188   (+6,793)
kept merge rules         42,354   of 580,604
final vocabulary         15,188 / 256,000 = 5.93%
```

The closure nearly doubles the vocabulary. That is the price of correctness, and it is cheap: 15,188
rows is still 5.9% of the original table.

`--seed-top` is the one real knob, and it moves the result a lot:

```text
seed_top  4,096  ->   7,779 tokens (3.04%)
seed_top  8,192  ->  15,188 tokens (5.93%)   <- the shipped checkpoint
seed_top 16,384  ->  28,312 tokens (11.06%)
```

Raise it if the robot must survive vocabulary the corpus never showed it; every extra token costs
768 parameters (3 KB at fp32) and nothing else.

### 1.5 The parity check gates everything

Before a single weight is touched, the new tokenizer is written and every corpus string is encoded
through both tokenizers. Modulo the id remapping, the sequences must be identical:

```python
old2new = {vocab[t]: i for t, i in new_id.items()}
if [old2new.get(x, -1) for x in old.encode(s).ids] != new.encode(s).ids:
    bad += 1
```

If `bad` is non-zero the script exits without writing weights:

```text
tokenizer parity: 2805/2805 strings identical
```

This check is the entire safety argument for calling the result lossless. It is also what makes the
domain explicit: the corpus *is* the guarantee. Anything the corpus did not cover is not covered.

### 1.6 Exactly what changes

`model.safetensors` — of 170 tensors, **one** carries a vocabulary dimension:

```text
encoder.embeddings.tok_embeddings.weight   (256000, 768) -> (15188, 768)
```

```python
idx = torch.tensor([vocab[t] for t in order], dtype=torch.long)
sd[k] = sd[k].index_select(0, idx).contiguous()
```

`order` is the kept tokens sorted by their *old* id, so new ids stay monotone in old ids and the
frequency ordering that `--seed-top` relies on survives.

No attention weight, no MLP weight, no layer norm, no positional parameter, and no decision-head
tensor is read or written. Total parameters, verified from the written file:

```text
321,908,998 -> 136,965,382     (125,300,998 unchanged + 15,188 x 768 = 11,664,384 embedding)
```

The embedding table goes from 61.1% of the model to 8.5% of it.

`encoder/config.json`:

- `vocab_size` -> 15188
- `pad_token_id`, `cls_token_id`, `bos_token_id`, `eos_token_id`, `sep_token_id`, `mask_token_id`
  remapped to the new ids

`tokenizer/tokenizer.json`:

- `model.vocab` rewritten to contiguous new ids
- `model.merges` filtered to rules whose both parents *and* child survive
- `added_tokens[].id` remapped
- `post_processor.special_tokens[*].ids` remapped

`rl_agent_config.json` and `tokenizer/tokenizer_config.json` are copied byte-for-byte. The output is
a drop-in checkpoint directory: `laya.load("laya-multilingual-en")` loads it unchanged.

### 1.7 Why this is lossless rather than approximately lossless

The embedding layer is a lookup: token id `i` selects row `E[i]`. Pruning applies a bijection between
kept old ids and new ids and permutes the rows to match. For any string whose tokenization is
unchanged — which the parity check has already established for the whole corpus — the sequence of
*row vectors* entering layer 0 is identical, element for element. Every downstream weight is the
original weight. The logits are therefore bit-identical, not close.

The measured accuracy confirms it, with all three tasks scoring exactly as the stock checkpoint on
every individual case:

```text
action 109/165      vision 139/165      about_self 80/165
```

**The boundary is real, though.** A string containing a dropped token now tokenizes differently — it
degrades to byte fallbacks rather than failing loudly. The model will answer, with a confidence, and
the confidence will not know. If the deployment's language changes, re-run the pruner against a
corpus that reflects it.

---

## 2. Block-wise int8

### 2.1 What a scale spans

Quantizing a float weight matrix to int8 means storing integers plus a scale that maps them back:
`w ≈ s * q`, with `q` an 8-bit integer. One scale has to cover the entire dynamic range of whatever
it spans, and the largest magnitude in that span sets the step size for everything else in it. A
single outlier coarsens every weight sharing its scale.

The only question is how finely scales are allocated. For a `[1152, 768]` matmul weight (Laya's MLP
shape at hidden size 768, intermediate size 1152):

| granularity | scales | each scale covers |
| --- | --- | --- |
| per-tensor | 1 | 884,736 weights |
| per-channel | 1,152 | 768 weights (one output row) |
| block-wise, `block_size=64` | 13,824 | 64 weights |

### 2.2 The measured ladder

Same pruned checkpoint, same 165 cases, action-selection accuracy:

| build | accuracy | vs fp32 |
| --- | --- | --- |
| fp32 baseline | 0.661 | — |
| torch dynamic int8, per-tensor, whole model | 0.594 | -6.7 points |
| torch dynamic int8, per-tensor, **encoder only** | 0.594 | -6.7 points |
| torch dynamic int8, per-channel | 0.624 | -3.7 points |
| ONNX MatMulNBits, block-wise, `block_size=64` | **0.667** | +0.6 points (one case better) |

Read down that column: granularity is the whole story. One scale per weight matrix costs about seven
points. One scale per 64 weights costs nothing measurable, and on this set lands one case ahead of
fp32 — noise in laya-micro's favour, not an improvement to claim.

### 2.3 The head-exclusion folk wisdom did not reproduce

The common advice is that the decision head is precision-sensitive and should be excluded from
quantization. `bench.py --quantize encoder` does exactly that, leaving the head in fp32.

It scores **0.594** — identical, case for case, to quantizing the whole model. Whatever per-tensor
int8 destroys, it destroys inside the encoder, and protecting the head buys nothing. The popular
explanation is wrong for per-tensor quantization on this model. Choosing scale granularity, not
choosing which modules to exempt, is what recovers the accuracy.

### 2.4 The recipe

`quantize_onnx.py` uses onnxruntime's `MatMulNBitsQuantizer`, which rewrites each `MatMul` against a
constant initializer into a `MatMulNBits` node holding int8 weights plus one scale per block of 64
along the reduction axis, symmetric (no zero point):

```python
quant = MatMulNBitsQuantizer(
    onnx.load(stripped),
    block_size=64,
    is_symmetric=True,
    bits=8,
    algo_config=DefaultWeightOnlyQuantConfig(block_size=64, is_symmetric=True, bits=8),
)
quant.process()
```

This is weight-only quantization: activations stay float, so there is no calibration set and no
activation-range estimation to get wrong. The embedding table is a `Gather`, not a `MatMul`, and is
left untouched — which is precisely why the two techniques compose.

The recipe comes from [nvkudva/laya-web-q8](https://huggingface.co/nvkudva/laya-web-q8), which
applied block-wise MatMulNBits int8 to Laya while keeping embeddings and the decision head at higher
precision, reporting 1688 MB -> 524 MB with 100% argmax agreement on a 26-question set. laya-micro
uses that recipe as published and scores it on labelled cases.

The accuracy ladder also corroborates a call made elsewhere:
[navopw/laya-onnx](https://github.com/navopw/laya-onnx) ships fp32 compute with fp16 vocabulary
storage and rejected int8 over accuracy loss. Per-tensor int8 costs 6.7 points here, so that was the
right call at that granularity.

The quantizer's own check is agreement, which is necessary but not sufficient:

```text
  max logit drift <value>, argmax matches
```

Agreement with fp32 says the conversion was faithful. It says nothing about whether the model is
right. That is what `bench.py` is for, and why the ladder above is scored on labelled cases rather
than on drift.

---

## 3. Why the two compose

The two techniques touch disjoint parameters:

| | what it is in the graph | what it holds | reduced by |
| --- | --- | --- | --- |
| embedding table | `Gather` | 196,608,000 params (61.1%) | pruning |
| attention + MLP + head | `MatMul` | 125,300,998 params (38.9%) | block-wise int8 |

`MatMulNBitsQuantizer` only rewrites matmuls against constant initializers, so no amount of
quantization compresses the embedding table. Pruning, conversely, never touches a matmul weight.
Their savings are therefore close to independent, which the measured builds bear out:

| build | disk | peak RSS | action acc |
| --- | --- | --- | --- |
| stock multilingual fp32 (torch) | 1228 MB | 3230 MB | 0.661 |
| vocab-pruned fp32 (torch) | 275 MB | 1061 MB | 0.661 |
| pruned + ONNX fp32 | 551 MB | 1516 MB | 0.661 |
| pruned + ONNX int8 block-64 | 371 MB | 1312 MB | **0.667** |

Net: 3.3x smaller on disk, 2.5x less RAM, accuracy marginally better than stock. Load time drops
24.5 s -> 9.6 s for the pruned torch checkpoint.

Two things in that table are worth not glossing over. The ONNX fp32 export is *larger* than the
pruned torch checkpoint, because ONNX writes fp32 initializers; int8 quantization is what brings it
back down. And RSS does not fall as far as disk does — runtime arenas, the session's optimized graph
copy and the tokenizer all cost memory that neither technique addresses.

A fully static-shape export was measured and **dropped**. At `--pad-to 384` the quantized graph
came out at 373 MB against the dynamic build's 371 MB — no smaller — and roughly twice as slow,
989 ms against 580 ms per three-question call, because every call pads to the full 384 tokens
whatever the input actually needs. Its accuracy was never measured, because there was no reason to
deploy it. The dynamic graph wins on both axes.

---

## 4. The ONNX export path

The Pi already runs onnxruntime for YOLOv8n, so exporting to ONNX keeps torch and transformers off
the robot entirely. Export on a workstation, copy the `.onnx` (and its `.data` sidecar) across.

The exported graph takes Laya's five preprocessed tensors and returns its two outputs:

```text
input_ids       [items, seq]   int64
attention_mask  [items, seq]   int64
marker_pos      [items, nopt]  int64     one position per rendered option
marker_mask     [items, nopt]  bool
qtype           [items]        int64     choice | score | noul
->
logits, act
```

`bench.py --onnx` swaps this session in for `agent.model`, so Laya's own tokenizing, option rendering
and answer decoding are reused unchanged and the comparison stays honest.

### Gotcha 1: the legacy exporter freezes reshape dimensions

`torch.onnx.export` **must** use `dynamo=True`. The TorchScript exporter's
`nn.MultiheadAttention` fallback bakes reshape dimensions into the graph. It exports without
complaint; the failure surfaces later, during quantization:

```text
[ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (772) vs (256)
```

Exporting with static shapes does **not** work around this. Only the dynamo exporter does.

### Gotcha 2: ModernBERT's compile path is not exportable

```python
agent.model.encoder.config.reference_compile = False
```

Set this before export or the `torch.compile` path is taken and the export fails.

### Gotcha 3: export under grad

`nn.TransformerEncoderLayer` takes an unexportable fused fast path under `no_grad`. Compute the
reference outputs inside `torch.no_grad()` if you want them, but run `torch.onnx.export` outside it.

### Gotcha 4: `quant_pre_process` is a dead end here

onnxruntime's shape-inference preprocessor fails both ways on these graphs. On a dynamic export it
asserts inside `_infer_Range`. On a static export it cannot reload its own `optimized.onnx`, because
the external-data file is not carried into its temp dir. Skip it entirely — delete the stale
`value_info` the exporter left behind and quantize directly:

```python
m = onnx.load(src)
del m.graph.value_info[:]
```

### Gotcha 5: the export sample needs at least two items

With a single-item sample batch, `torch.export` specialises the batch dimension to 1 and the
resulting graph can only ever answer one question per call. `export_onnx.py` builds its sample from
Zeus's real three-question set, which also keeps the option dimension honestly variable.

### Gotcha 6: torch-side quantization needs two flags set

Only relevant if you compare against `--quantize` in `bench.py` rather than the ONNX path:

```python
torch.backends.quantized.engine = "qnnpack"      # defaults to "none"
torch.backends.mha.set_fastpath_enabled(False)
```

Without the first:

```text
quantized::linear_prepack NoQEngine
```

Without the second, every quantized inference dies with:

```text
function object has no attribute device
```

### Gotcha 7: run with `USE_TF=0`

`transformers` probes for TensorFlow at import, and its abseil runtime can deadlock model
construction. Every script sets `os.environ.setdefault("USE_TF", "0")` before importing.

### Gotcha 8: the checkpoints are subfolders

`laya`, `laya-multilingual` and `laya-typed-decisions` ship as **subfolders** of
`convaiinnovations/laya`, not as standalone repositories. `laya.load` on a standalone repo id hits
the network; `Router.preload(["multilingual"])` resolves the cached subfolder.

### Static shapes

`--pad-to N` fixes the sequence length at `N` instead of leaving it dynamic. More matmuls then keep
constant-shaped weights, so more of the graph is quantizable, and the dynamic `Range` ops that defeat
shape inference disappear. `N` must be at least as long as the longest batch the deployment builds,
and callers must pad their own inputs to exactly `N`.

A static export fixes the item dimension too, so the graph answers exactly the question set it was
exported for. That also means callers must pad their own inputs to `N`, which `bench.py` does not do
— a static graph needs its own runner.

It was measured and rejected. At `--pad-to 384` on the pruned multilingual checkpoint the quantized
graph came out at 373 MB against the dynamic graph's 371 MB, so no smaller, and about twice as slow:
989 ms against 580 ms per three-question call on an M3 Pro, because every call pads to the full
length whatever the input needs. Its accuracy was never measured, because there was no reason to
deploy it.

---

## 5. A constraint worth designing around: the option token budget

Laya's model card advises keeping `choice` questions under roughly 20 options, because a question's
options share a fixed token budget — `head_max_len`, which is **256** for the multilingual
checkpoint.

Zeus has 27 actions plus `none` = 28 options, which render to **259 tokens**. The budget is saturated
before any state is added.

The documented remedy is a hierarchy: ask for a category first (7 options), then the action within
it (never more than 7). `bench.py --shape hier` implements exactly that. Measured:

| checkpoint | flat (28 options) | hierarchical (2 stages) |
| --- | --- | --- |
| multilingual | **0.661** | 0.182 |
| english | 0.406 | 0.364 |

It inverts by checkpoint. On multilingual the category stage collapses and takes the whole pipeline
with it; the oversubscribed flat question is more than three times better. On English the two are
closer, and flat is still ahead. The advice is sound in principle and destroyed one of these two
checkpoints in practice — measure it on the checkpoint you are shipping.

---

## 6. Reproducing

```bash
# 1. Prune the multilingual checkpoint against your domain corpus.
#    Refuses to write weights unless tokenizer parity is exact.
python3 prune_vocab.py --subfolder multilingual --out laya-multilingual-en --seed-top 8192

# 2. Export to ONNX (dynamo exporter, opset 18, dynamic item/seq/option dims).
python3 export_onnx.py --model laya-multilingual-en --out laya_multilingual_en

# 3. Block-wise int8: one scale per 64 weights, symmetric.
python3 quantize_onnx.py laya_multilingual_en.onnx --block-size 64 --bits 8

# 4. Score it on the labelled cases, with baselines and precision/recall.
python3 bench.py --backend laya --shape flat --model laya-multilingual-en \
    --onnx laya_multilingual_en.int8.onnx --threads 4
```

Step 4 is not optional. Steps 1–3 all report internal consistency checks — parity, logit drift,
argmax agreement — and every one of them can pass on a model that answers wrongly.

To re-derive the vocabulary statistics in section 1 without writing a checkpoint, call the pruner's
own keep-set builder directly:

```python
import json, prune_vocab as pv
texts = pv.corpus("cases/zeus_cases.json")
d, vocab, keep = pv.keep_set(SRC_TOKENIZER_JSON, texts, 8192)
print(len(texts), len(vocab), len(keep))
```

---

## 7. Changes to the original model

Laya's weights are Apache-2.0, licensed by Convai Innovations. Apache-2.0 section 4 requires that
modified files carry prominent notices stating the changes. Any Laya checkpoint redistributed from
this repository has been changed in the following ways, and `NOTICE` records the same:

- **Vocabulary pruned.** `encoder.embeddings.tok_embeddings.weight` sliced from 256,000 rows to the
  merge-closed keep set of a domain corpus; `encoder/config.json` `vocab_size` and special-token ids
  rewritten; `tokenizer/tokenizer.json` vocabulary, merge rules, added tokens and post-processor ids
  remapped. No transformer or decision-head weight is altered.
- **Exported to ONNX** via `torch.onnx.export` with the dynamo exporter at opset 18.
- **Quantized** with `MatMulNBitsQuantizer` to 8-bit block-wise weights, `block_size=64`, symmetric,
  applied to matmul weights only.
