# Benchmarks

Every number laya-micro reports, with the conditions it was taken under. Where something has not
been measured it says PENDING or TODO rather than estimating.

## Conditions

| | |
| --- | --- |
| Machine | Apple M3 Pro, CPU only |
| Cases | 165 labelled commands from [`cases/zeus_cases.json`](../cases/zeus_cases.json) (173 total, 8 flagged `ambiguous` and excluded) |
| Checkpoint | `laya-multilingual` (mmBERT-base), vocabulary-pruned to the Zeus corpus unless the row says "stock" |
| Question shape | `--shape flat` — all 27 actions plus `none` in one `choice` |
| ONNX threads | 4 |

"Action accuracy" means top-1 accuracy at picking one of 27 robot actions, or `none`. Peak RSS is the
whole harness process via `ru_maxrss`; the ONNX rows include the torch checkpoint the harness keeps
loaded for tokenizing and decoding.

**These are Apple Silicon numbers.** Apple Silicon latency does not predict ARM latency. Raspberry
All figures are Apple M3 Pro, CPU, 4 threads.

## Size, memory and accuracy

| build | disk | peak RSS | action acc |
| --- | --- | --- | --- |
| stock multilingual fp32 (torch) | 1228 MB | 3230 MB | 0.661 |
| vocab-pruned fp32 (torch) | 275 MB | 1061 MB | 0.661 |
| pruned + ONNX fp32, dynamic shapes | 551 MB | 1516 MB | 0.661 |
| pruned + ONNX int8 block-64, dynamic shapes | 371 MB | 1312 MB | **0.667** |
| **pruned + ONNX int8 block-64, via `runtime.py`** | **371 MB** | **358 MB** | **0.667** |

Side-question accuracy is unchanged by quantization except for one vision case: the dynamic fp32 build
scores vision 139/165 and about_self 80/165, the int8 build 140/165 and 80/165.

Stock to pruned int8: 3.3x smaller on disk, 2.5x less RAM, accuracy marginally better than stock.

A fully static-shape export was measured and **dropped**. At `--pad-to 384` the quantized graph
came out at 373 MB against the dynamic build's 371 MB — no smaller — and roughly twice as slow,
989 ms against 580 ms per three-question call, because every call pads to the full 384 tokens
whatever the input actually needs. Its accuracy was never measured, because there was no reason to
deploy it. The dynamic graph wins on both axes.

Each row was measured one process at a time; concurrent runs contend for memory and their RSS
readings are not comparable.

The static rows and the rows above them were recorded in different sessions on the same machine and
harness. The accuracies are exact matches, so the builds are comparable; treat small RSS differences
between the static and dynamic rows with more caution than the disk figures.

Two things in that table are worth not glossing over. The fp32 ONNX export is *larger* than the
pruned torch checkpoint, because ONNX writes fp32 initializers — quantization is what brings it back
down. And RSS does not fall as far as disk does; runtime arenas, the session's optimized graph copy
and the tokenizer cost memory that neither technique touches.

## Vocabulary pruning

Applied to the multilingual checkpoint. No transformer weight changes, so in-domain answers are
bit-identical.

| | before | after |
| --- | --- | --- |
| Vocabulary | 256,000 | 15,188 (5.9%) |
| Checkpoint on disk | 1228 MB | 275 MB |
| Peak RSS | 3230 MB | 1061 MB |
| Load time | 24.5 s | 9.6 s |
| Tokenizer parity | — | 2805/2805 strings identical |
| action / vision / about_self | 109/165, 139/165, 80/165 | 109/165, 139/165, 80/165 |

**This only pays off on the multilingual checkpoint.** mmBERT-base spends 196,608,000 of its
321,908,998 parameters — 61.1% — on a 256,000 x 768 embedding table. The English checkpoint's table
is 50,368 x 1024 = 51.6M parameters, only 12.2% of its 421M.

## Quantization granularity

Same pruned checkpoint, same 165 cases.

| quantization | action acc | vs fp32 |
| --- | --- | --- |
| fp32 baseline | 0.661 | — |
| torch dynamic int8, per-tensor, whole model | 0.594 | -6.7 points |
| torch dynamic int8, per-tensor, encoder only | 0.594 | -6.7 points |
| torch dynamic int8, per-channel | 0.624 | -3.7 points |
| ONNX MatMulNBits, block-wise, `block_size=64` | 0.667 | +0.6 points |

What matters is how finely scales are allocated, not which modules you quantize. One scale per whole
weight matrix costs about seven points; one scale per 64 weights costs nothing. Excluding the
decision head from per-tensor quantization changed the result not at all — 0.594 either way — so the
common "the head is sensitive" explanation does not hold for per-tensor quantization of this model.

TODO: the recorded per-channel run does not note whether it used `--quantize encoder` or
`--quantize all`. The two per-tensor runs were identical, so the scope is unlikely to matter, but
confirm it before citing the scope.

## Accuracy against baselines

The classes are imbalanced, so `bench.py` prints a majority-class baseline beside every score.

| task | majority baseline | Laya | verdict |
| --- | --- | --- | --- |
| action | 0.291 | 0.661 | 2.3x the baseline — genuinely useful |
| vision | 0.885 | 0.842 | **loses to always answering "no"** |
| about_self | 0.903 | 0.485 | **far worse than always answering "no"** |

On `about_self` the harness reports recall 0.812 against precision 0.137 — 82 false positives. It
finds most of the true positives and then fires on 82 cases that are not.

That looks like a decision threshold sitting in the wrong place, so we swept it. It is not:

| threshold | accuracy | precision | recall | F1 | tp | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 0.261 | 0.116 | 1.000 | 0.208 | 16 | 122 | 0 |
| 0.50 (default) | 0.485 | 0.137 | 0.812 | 0.234 | 13 | 82 | 3 |
| 0.70 | 0.745 | 0.158 | 0.375 | 0.222 | 6 | 32 | 10 |
| 0.85 (best F1) | 0.891 | 0.400 | 0.250 | 0.308 | 4 | 6 | 12 |
| 0.95 | 0.909 | 1.000 | 0.062 | 0.118 | 1 | 0 | 15 |

**No threshold beats the 0.903 majority baseline while retaining useful recall.** The only settings
that reach baseline accuracy get there by answering "no" to nearly everything — at 0.95 it catches
one of sixteen positives. Ranking quality is the real limit: **AUC 0.672**, with a mean score of
0.651 on positives against 0.513 on negatives. The scores overlap too much for any cut point to
separate them.

So this is a capability limit on this task, not a tuning oversight. Zeus keeps its existing regex
for self-queries.

`vision` fails differently, and more usefully. Its errors are almost entirely one-directional:

| want → got | count |
| --- | --- |
| none → none | 123 |
| **none → objects** | **22** |
| objects → objects | 11 |
| describe → describe | 6 |
| objects → none | 1 |
| describe → objects | 1 |
| none → describe | 1 |

It catches 17 of the 19 genuine camera queries and misclassifies 22 that need no camera. For a robot
that means unnecessary camera work rather than missed requests — the cheaper direction to fail in,
though still below the baseline overall.

**Small balanced samples flatter rare-class performance badly.** An early 14-case balanced sample of
this task scored vision 0.929 and about_self 0.857 and looked excellent. On the realistic 165-case
distribution both fall below a constant "no". That is why baselines are printed on every run.

## Flat against hierarchical questions

Laya's model card advises keeping `choice` under roughly 20 options, because options share a fixed
token budget (`head_max_len`). Zeus has 27 actions plus `none` = 28 options, which renders to 259
tokens against a `head_max_len` of 256 — saturated before any state is added. The documented fix is
to split into a hierarchy.

| checkpoint | flat (28 options) | hierarchical (category, then max 7) |
| --- | --- | --- |
| multilingual | 0.661 | 0.182 |
| english | 0.406 | 0.364 |

**The advice inverts by checkpoint.** For multilingual the category stage collapses and takes the
pipeline with it. For English flat is still ahead, by less. Measure it on your own checkpoint and
option set before splitting anything.

## Prior published figures, for context

These belong to other people's builds and are not laya-micro measurements.

- **navopw/laya-onnx**, Raspberry Pi 5, 2026-09-20: English checkpoint, 234 ms for a single question,
  1015 ms for four questions, RSS 1076.7 MiB, 4 threads, ONNX Runtime 1.30.0. fp32 compute with fp16
  vocabulary storage, unpruned — a different build, so context rather than comparison.
- **nvkudva/laya-web-q8**: 1688 MB to 524 MB with 100% argmax agreement on their 26-question set,
  using the block-wise MatMulNBits recipe [`quantize_onnx.py`](../quantize_onnx.py) follows.

## Not measured

- **Other CPU architectures.** Every number above is Apple M3 Pro, CPU. Latency in particular does not transfer between architectures.
- **Latency for any laya-micro build.** The harness prints `latency p50 / min / max` and `ms/call`,
  but only accuracy, disk and peak RSS were recorded. TODO.
- **Category-stage accuracy for `--shape hier`.** TODO; only the end-to-end action accuracies
  (0.182 multilingual, 0.364 English) are recorded.
- **`tp`/`fn` counts for every run.** Recorded for the ONNX builds — `about_self` tp 13, fp 82,
  fn 3; `vision` tp 18, fp 24, fn 1 at fp32 and fp 23 at int8 — but not written down for the stock
  or pruned-torch runs. TODO.
