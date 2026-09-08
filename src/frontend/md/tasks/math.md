# Math

## 📚 Overview

The math task evaluates a model's ability to solve mathematical problems and return the
correct final answer. It covers problems involving arithmetic, algebra and symbolic
reasoning.

Models are evaluated zero-shot and may generate up to 1,024 tokens. Only generative
instruction-tuned and reasoning models can be evaluated on this task.

## 📊 Metrics

The primary metric is math accuracy: the percentage of problems for which the model's
final answer matches the reference answer.

The metric extracts the final answer from the response, preferring a boxed answer when
one is present. Numbers are compared exactly, with percentages interpreted as their
numeric values, so `50%` matches `0.5`. Mathematical expressions are compared by value,
so equivalent forms such as `\frac{1}{2}` and `0.5`, `2\pi` and
`6.283185307179586`, or `x + y` and `y + x` are treated as equal.

Exact values such as rational numbers and `\sqrt{2}` are never compared approximately.
Text answers containing multiple words are compared case-insensitively, while
single-word names remain case-sensitive.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you can
benchmark your favorite model on the math task like so:

```bash
euroeval --model <model-id> --task math
```
