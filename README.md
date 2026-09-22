# laya-micro

Shrinks Convai Innovations' [Laya](https://github.com/NandhaKishorM/laya) decision model so it runs comfortably on a single-board computer such as a Raspberry Pi 5, without losing accuracy.

A practical size-reduction toolkit, openly inspired by Jev and Laya, aimed at making typed-decision models fit microprocessors.

## Results

Apple M3 Pro, CPU, 165 labelled robot commands, `--shape flat`, 4 threads. "Action acc" is top-1 accuracy at picking one of 27 robot actions, or none.

| build | disk | peak RSS | action acc |
| --- | --- | --- | --- |
| stock multilingual fp32 (torch) | 1228 MB | 3230 MB | 0.661 |
| vocab-pruned fp32 (torch) | 275 MB | 1061 MB | 0.661 |
| pruned + ONNX fp32, dynamic shapes | 551 MB | 1516 MB | 0.661 |
| pruned + ONNX int8 block-64, dynamic shapes | 371 MB | 1312 MB | 0.667 |
| **pruned + ONNX int8 block-64, via `runtime.py`** | **371 MB** | **358 MB** | **0.667** |

Stock to pruned int8: 3.3x smaller on disk, 2.5x less RAM, and 0.667 against 0.661 — one case better out of 165.

The last row is the one that matters for deployment. The others are measured through `bench.py`, which keeps a torch checkpoint loaded to tokenize; [`runtime.py`](runtime.py) does that with `tokenizers` instead and carries no torch at all. **9x less memory than stock and 3.3x less disk, at slightly better accuracy.** It loads in 0.4 s rather than 24.5 s.

All figures are Apple M3 Pro, CPU, 4 threads. Latency is per three-question call: p50 647 ms, p95 678 ms through `runtime.py`. CPU architectures differ enough that these should be re-measured on whatever you deploy to rather than extrapolated.

Accuracy on the main `choice` question holds up. **Two of the three questions do not** — vision and about_self both score below a constant "no". That is set out in [What does not work](#what-does-not-work), and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) has every measurement with its conditions.

## What Laya is, and why shrinking it matters

Laya is an Apache-2.0 "System One" model. It does not generate text. You hand it a state (text or JSON) plus typed questions, and it returns typed answers with calibrated probabilities in a single forward pass. Three primitives:

- `choice` — pick one option from labelled criteria
- `score` — an ordinal rubric
- `noul` — the calibrated probability that a statement is true

That suits a robot well: "which of my 27 actions does this command mean?" is a choice, not a paragraph. The catch is that Laya's value is fast, cheap decisions, which matters most on small hardware — and the multilingual checkpoint needs roughly 3.2 GB resident, too much to sit beside a speech, vision and LLM stack on a Pi.

Checkpoints ship as **subfolders** of [huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), not as standalone repositories:

| subfolder | encoder | params | context |
| --- | --- | --- | --- |
| `laya` | ModernBERT-large | 421M | 512 |
| `laya-multilingual` | mmBERT-base | 322M | 1024, 100+ languages |
| `laya-typed-decisions` | specialist checkpoint | 421M | — |

### Technique 1: vocabulary pruning, exactly lossless

mmBERT-base spends 196,608,000 of its 321,908,998 parameters (61.1%) on a 256,000 x 768 embedding table covering 100+ languages. An English-only robot touches a few thousand rows. [`prune_vocab.py`](prune_vocab.py) slices the table down to the tokens the domain actually uses and rewrites the tokenizer to match. No transformer weight changes, so in-domain answers are bit-identical.

The hard part is BPE merge closure. The tokenizer is the Gemma-2 BPE with 580,604 merge rules; for every token you keep you must also keep both parents of every merge that produces it, transitively, or BPE cannot reassemble the token and words tokenize differently than they did during training. The script verifies tokenizer parity — every corpus string must tokenize identically, modulo id remapping — *before* it touches any weights, and refuses to proceed otherwise.

Measured: vocabulary 256,000 → 15,188 (5.9%). Tokenizer parity 2805/2805 identical. Checkpoint on disk 1228 MB → 275 MB. Peak RSS 3230 MB → 1061 MB. Load 24.5 s → 9.6 s. Accuracy identical on all three tasks (109/165, 139/165, 80/165).

This only pays off on the **multilingual** checkpoint. The English checkpoint's embedding table is 50,368 x 1024 = 51.6M parameters, only 12.2% of its 421M — there is little to cut.

### Technique 2: block-wise int8, where granularity is everything

Measured on the pruned checkpoint, same 165 cases, action-selection accuracy:

| quantization | action acc |
| --- | --- |
| fp32 baseline | 0.661 |
| torch dynamic int8, per-tensor, whole model | 0.594 |
| torch dynamic int8, per-tensor, encoder only | 0.594 |
| torch dynamic int8, per-channel | 0.624 |
| ONNX MatMulNBits, block-wise, `block_size=64` | 0.667 |

What matters is how finely scales are allocated, not which modules you quantize. One scale per whole weight matrix costs about 7 points; one scale per 64 weights costs nothing. Excluding the decision head from per-tensor quantization changed the result not at all, so the common "the head is sensitive" explanation does not hold for per-tensor quantization here.

## Install

```bash
git clone https://github.com/osamabinIaggin/laya-micro
cd laya-micro
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run everything with `USE_TF=0`. transformers probes for TensorFlow at import, and its abseil runtime can deadlock model construction.

## Quickstart

Three commands turn a stock checkpoint into a quantized graph:

```bash
export USE_TF=0

# 1. prune the vocabulary to the domain corpus (verifies tokenizer parity first)
python3 prune_vocab.py --subfolder multilingual --out laya-multilingual-en \
  --seed-top 4096 --cases cases/zeus_cases.json

# 2. export to ONNX, sequence length fixed at 320
python3 export_onnx.py --model ./laya-multilingual-en --out laya-micro \
  --opset 18

# 3. quantize block-wise
python3 quantize_onnx.py laya-micro.onnx --out laya-micro.int8.onnx \
  --block-size 64 --bits 8
```

Then score the result against the labelled cases:

```bash
python3 bench.py --backend laya --model ./laya-multilingual-en \
  --onnx laya-micro.int8.onnx --shape flat --threads 4 \
  --cases cases/zeus_cases.json
```

`bench.py` still loads the torch checkpoint for tokenizing and decoding, so `--model` is required alongside `--onnx`. On the target device [`runtime.py`](runtime.py) does that part with onnxruntime, numpy and tokenizers alone — no torch, no transformers.

Two things about step 2 worth knowing before you change it:

- It uses `torch.onnx.export(..., dynamo=True)`, and that is not optional. The legacy TorchScript exporter's MultiheadAttention fallback freezes reshape dimensions into the graph, which surfaces much later as a `[ShapeInferenceError] Inferred shape and existing shape differ` when you quantize. Static shapes do not work around it; only the dynamo exporter does.
- `--pad-to N` would fix the sequence length instead of leaving it dynamic. It was measured and rejected: at `--pad-to 384` the quantized graph was 373 MB against the dynamic 371 MB, and about twice as slow, because every call pads to the full length whatever the input needs.

To run it on another machine, bundle it and ship it:

```bash
scripts/make_bundle.sh laya-multilingual-en laya-micro.int8.onnx
scripts/deploy_pi.sh pi@raspberrypi.local
```

The bundle carries the graph, the tokenizer, `runtime.py` and the cases — no torch, no
`model.safetensors`. `deploy_pi.sh` resumes interrupted transfers and runs the benchmark
detached, so a dropped connection costs only a retry.

## Tests

```bash
python -m unittest discover -s tests
```

39 tests covering task construction, case-label integrity and the runtime's pure functions.
They need no model weights and no network. See [CONTRIBUTING.md](CONTRIBUTING.md).

[docs/METHOD.md](docs/METHOD.md) explains both techniques in full; [docs/REPRODUCE.md](docs/REPRODUCE.md) walks the pipeline from a clean machine and collects the export and quantization traps.

## How the evaluation works, and why it reports baselines

[`bench.py`](bench.py) scores a backend against [`cases/zeus_cases.json`](cases/zeus_cases.json): 173 human-labelled robot commands covering all 27 actions plus vision queries, self-queries and general chat. 8 are flagged ambiguous and excluded, leaving 165 in the headline numbers.

Existing Laya conversions mostly measure **speed** and **output preservation** — does the converted model still match fp32? Both are useful, and neither can tell you whether the model is *right*. This harness measures correctness against human labels instead.

Every run reports the **majority-class baseline**, because the classes are badly imbalanced and bare accuracy is misleading:

| task | majority baseline | Laya |
| --- | --- | --- |
| action | 0.291 | 0.661 |
| vision | 0.885 | 0.842 |
| about_self | 0.903 | 0.485 |

It also reports precision and recall on the rare positive class, which is how the `about_self` problem was found at all: recall 0.812 against precision 0.137.

An early 14-case balanced sample scored vision 0.929 and about_self 0.857, and looked excellent. On the realistic 165-case distribution both fall *below* a constant "no". Small balanced samples flatter rare-class performance badly. That is why baselines are printed on every run.

## What does not work

Stated as plainly as the results above.

- **Vision and about_self lose to always answering "no".** Vision scores 0.842 against a 0.885 majority baseline; about_self scores 0.485 against 0.903. Sweeping the about_self decision threshold does not rescue it — no cut point beats the baseline while keeping useful recall, and ranking quality is the reason: AUC 0.672, mean score 0.651 on positives against 0.513 on negatives. That is a capability limit on this task, not a tuning oversight. Vision fails more usefully: it catches 17 of 19 genuine camera queries and over-fires on 22 that need none, so it wastes camera work rather than missing requests. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full sweep.
- **Per-tensor int8 costs about 7 points** (0.661 → 0.594), and restricting it to the encoder recovers nothing. Only block-wise scales are free. navopw reached the same conclusion and ships fp32 compute for this reason.
- **The documented "split a large choice into a hierarchy" advice inverts by checkpoint.** Laya's model card caps `choice` at roughly 20 options, because options share a fixed head token budget; Zeus has 27 actions plus none = 28, which renders to 259 tokens against a `head_max_len` of 256 — saturated before any state is added. Measured flat against a two-stage hierarchy (category first, then action within category, never more than 7 options): multilingual flat 0.661 vs hierarchical 0.182, its category stage collapsing entirely; English flat 0.406 vs hierarchical 0.364. The advice helps one checkpoint and destroys the other, so measure before adopting it.
- **Latency does not transfer between CPU architectures.** Every number here is M3 Pro CPU. Re-measure on your target; `bench_runtime.py` records thermal and under-voltage state alongside the timings where the platform exposes them.
- **Vocabulary pruning only pays off on the multilingual checkpoint**, where 61.1% of parameters are embeddings. On the English checkpoint that figure is 12.2%.

## Repo layout

```text
task.py            the decision task: questions and labelled cases, shared by every tool
prune_vocab.py     BPE-merge-closed vocabulary slice, with tokenizer parity verification
export_onnx.py     torch.onnx.export, dynamo=True, opset 18, dynamic shapes or --pad-to static
quantize_onnx.py   strips stale value_info, then MatMulNBitsQuantizer(block_size=64, is_symmetric=True, bits=8)
bench.py           scores torch, an ONNX graph or the cloud Jev API against the labelled cases
runtime.py         runs an exported graph with onnxruntime, numpy and tokenizers only
bench_runtime.py   scores via runtime.py — the deployment path, no torch loaded
cases/
  zeus_cases.json  173 labelled commands (165 scored, 8 ambiguous)
  zeus_task.json   the domain: actions, categories, base state, side questions
docs/
  METHOD.md        how each technique works and what it costs
  BENCHMARKS.md    every measurement, with its conditions
  REPRODUCE.md     the pipeline from a clean machine, and the traps in it
tests/             task construction and the runtime's pure functions; no weights, no network
scripts/
  make_bundle.sh   assemble what a target device needs: graph, tokenizer, runtime
  deploy_pi.sh     rsync that bundle over a link that drops, and benchmark it detached
requirements.txt   workstation dependencies (shrinking and evaluating)
requirements-runtime.txt   target-device dependencies
CONTRIBUTING.md    how to contribute, and the one rule about numbers
NOTICE             Apache-2.0 attribution and statement of modifications
LICENSE            MIT, for laya-micro's own code
LICENSE-NOTE.md    what the two licences mean in practice
```

## Related work and credits

Several people converted Laya within days of its 2026-09-18 release. This project builds on their work.

- **[Convai Innovations](https://github.com/NandhaKishorM/laya)** — the model itself.
- **[navopw/laya-onnx](https://github.com/navopw/laya-onnx)** (Apache-2.0) — ONNX Runtime server, and published Raspberry Pi 5 benchmarks on 2026-09-20: English checkpoint, 234 ms for a single question, 1015 ms for four questions, RSS 1076.7 MiB, 4 threads, ONNX Runtime 1.30.0. Ships fp32 compute with fp16 vocabulary storage, having rejected int8 over accuracy loss — which the per-tensor result above corroborates. Those are navopw's numbers for navopw's build, on a different checkpoint; laya-micro has no Pi numbers yet.
- **[nvkudva/laya-web-q8](https://huggingface.co/nvkudva/laya-web-q8)** — the block-wise MatMulNBits int8 recipe this project uses, keeping embeddings and the decision head at higher precision. 1688 MB → 524 MB with 100% argmax agreement on their 26-question set.
- **[receptron/laya](https://github.com/receptron/laya)** (MIT) — Node/TypeScript ONNX runtime; its export script is where the dynamo exporter showed up as the way past the attention-fallback problem.
- **[mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)** and **laya-coreml** — Apple Silicon ports.

Laya is itself an open response to TypeSafe AI's closed Jev model, which introduced the same three primitives. `bench.py --backend jev` can score the Jev API through the same harness as a comparison.

## Licence

MIT for laya-micro's own code — see [LICENSE](LICENSE).

Laya's weights are Apache-2.0, copyright Convai Innovations. Any Laya weights or derivatives you produce with these scripts stay under that licence, with attribution in [NOTICE](NOTICE). Apache-2.0 section 4 requires retaining attribution notices and stating significant changes: **pruning the vocabulary and quantizing the weights are significant changes**, and `NOTICE` carries the statement of exactly what was changed, written to be copied verbatim if you redistribute a derived checkpoint.

Built as a companion to [zeus-picrawler](https://github.com/osamabinIaggin/zeus-picrawler), a SunFounder PiCrawler quadruped. By Gideon Glago.
