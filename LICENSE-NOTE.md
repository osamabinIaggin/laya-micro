# Licensing, in plain English

| What | Licence |
| --- | --- |
| laya-micro's own code (`prune_vocab.py`, `export_onnx.py`, `quantize_onnx.py`, `bench.py`), its docs and its case files | MIT |
| Laya model weights and tokenizer, from Convai Innovations | Apache-2.0 |
| Any checkpoint you produce with these scripts — pruned, exported or quantized | Apache-2.0 |

This page summarises what the licence texts say. It is a summary, not legal
advice.

## The two licences stack, they do not mix

The scripts are this project's work and are MIT. The weights are Convai
Innovations' work and are Apache-2.0. Running an MIT script over Apache-2.0
weights does not relicense the weights: the output is a derivative of the
input, so a pruned or quantized checkpoint is Apache-2.0, and stays Apache-2.0
through any number of further conversions.

## This repository ships no weights

Clone it and you get four Python scripts, a labelled case file and
documentation, all MIT. Laya checkpoints are downloaded from Convai
Innovations onto your own machine and every derivative artifact is built
locally. Apache-2.0 obligations start applying to you when you redistribute
something built from those weights.

## If you redistribute a model you produced

Apache-2.0 section 4 is the operative part. Ship these alongside the artifact:

1. **A copy of the Apache License 2.0** — section 4(a).
2. **A prominent statement that the files are modified** — section 4(b).
   Vocabulary pruning and quantization are significant changes and have to be
   stated. Section 2 of this repository's `NOTICE` is written to be copied
   verbatim for exactly this purpose.
3. **The upstream attribution notices** — sections 4(c) and 4(d). Keep Convai
   Innovations' copyright and attribution notices, and carry a readable copy of
   the `NOTICE` attributions in one of: your own NOTICE file, your source or
   documentation, or a display your software generates.
4. **A statement of anything further you changed** on top of what laya-micro
   changed.

You may add your own copyright notice and your own terms to the parts you
added, provided the Apache-2.0 terms still govern the Apache-2.0 parts.

## Commercial use

Permitted under both licences. MIT and Apache-2.0 are both permissive: you can
use, modify, sell and embed this in a closed-source product, and neither
obliges you to publish your own source.

Two differences worth knowing:

- **Patents.** Apache-2.0 section 3 grants you a patent licence from the
  contributors, and that grant terminates if you initiate patent litigation
  alleging the Work infringes a patent. MIT grants no patent rights either way.
- **Attribution.** MIT asks you to keep the copyright notice and licence text.
  Apache-2.0 asks for that, plus the modification statement and the NOTICE
  attributions.

## Trademarks

Apache-2.0 section 6 grants no trademark rights. "Laya" and "Convai
Innovations" appear here only to identify the upstream model. laya-micro is an
independent project, not affiliated with or endorsed by Convai Innovations,
TypeSafe AI, or any project credited in `NOTICE`.

## No warranty

Both licences disclaim warranties and limit liability. Nothing here promises
accuracy or fitness for a purpose. Two specifics worth stating plainly: a
vocabulary-pruned checkpoint is exact only over the token set it was built for,
and quantization changes weight values, so a quantized artifact is not
guaranteed to reproduce upstream decisions on your data. Measure it on your own
labelled cases.

## Not verified by this project

- The licences of the base models Laya is built on (ModernBERT-large,
  mmBERT-base). If you redistribute weights, check them yourself.
- The licence of `nvkudva/laya-web-q8`, whose block-wise quantization recipe
  this project follows. No code or weights from it are redistributed — only the
  method is used.
