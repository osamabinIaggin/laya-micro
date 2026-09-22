# Contributing

Contributions are welcome, particularly measurements on hardware this has not run on.

## The one rule that matters

**Every number in this repository traces to a measurement someone actually took.**

No estimates, no extrapolations presented as results, no figures carried over from a
similar configuration. If something has not been measured, the docs say so. If you add
a number, say what hardware produced it and how to reproduce it.

This is the whole point of the project. Published conversions of this model report speed
and output-preservation — whether the converted model matches fp32. Almost none report
whether the model is *right*. That is the gap this repository exists to fill, and it only
works if the numbers are trustworthy.

## Running the tests

```bash
pip install numpy
python -m unittest discover -s tests -v
```

They need no model weights and no network, and run in well under a second. They cover
task construction, case-label integrity, and the runtime's pure functions — the port of
Laya's tokenizing and decoding, where a small change silently shifts every probability
the model reports.

The full pipeline needs about 4 GB of disk and a checkpoint download; see
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## Reporting measurements on new hardware

The most useful contribution. To add a platform:

```bash
scripts/make_bundle.sh <pruned-checkpoint> <quantized.onnx>
python bench_runtime.py --threads 4 --out results.json
```

Open an issue or PR with `results.json` and:

- the exact device, CPU and RAM
- OS, Python and onnxruntime versions
- thread count
- whether anything else was running — an idle-machine number is much less useful than one
  taken under the load the device will really carry
- on a Raspberry Pi, the `vcgencmd get_throttled` value before and after. `bench_runtime.py`
  records it automatically. **A run that throttled is still worth reporting — say so.**

Latency does not transfer between CPU architectures, so please do not extrapolate from a
platform to one you have not run on.

## Adding your own domain

The task definition is data, not code. Copy `cases/zeus_task.json`, replace the actions,
categories, base state and side questions, and supply labelled cases in the shape of
`cases/zeus_cases.json`. Then:

```bash
python bench.py --task my_task.json --cases my_cases.json --backend laya --model <checkpoint>
```

Label honestly, including cases where the right answer is genuinely arguable — mark those
`"ambiguous": true` and they are excluded from headline scores. And keep the class balance
realistic rather than even: an early 14-case balanced sample of this task made two questions
look excellent that in fact score below a constant "no" on a realistic distribution. The
harness prints majority-class baselines on every run for exactly that reason. Please leave
that in.

## Code style

Match the surrounding code. It is plain Python with no framework: standard library, numpy,
and onnxruntime or torch where unavoidable. `runtime.py` must stay free of torch and
transformers — it is what runs on the target device, and its dependency list is the point.

Comments should state the invariant or the surprise, not narrate the change.

## Licensing

Code contributions are under the repository's MIT licence. Note that the Laya weights and
any checkpoint produced by these scripts remain Apache-2.0; see [NOTICE](NOTICE) and
[LICENSE-NOTE.md](LICENSE-NOTE.md).
