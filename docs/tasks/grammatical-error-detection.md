# Grammatical Error Detection

## 📚 Overview

Grammatical error detection is a task of identifying tokens in a sentence that contain
grammatical errors. In EuroEval, this task focuses on detecting misplaced verbs in
Germanic languages. Sentences are created from well-formed source sentences by
permuting a subset of the verbs to incorrect positions within their phrase boundaries,
and the model must identify which tokens are misplaced.

When evaluating generative models, we allow the model to generate 128 tokens on this
task.

## 📊 Metrics

The primary metric we use when evaluating the performance of a model on the grammatical
error detection task is the [micro-average
F1-score](https://en.wikipedia.org/wiki/F-score#Micro_F1) without MISC, computed as the
total number of true positives for all (non-trivial) error tokens, divided by the total
number of predicted error tokens.

We also report the micro-average F1-score, computed the same way, but where we include
all token labels. Since there are no MISC tags in this task, both metrics are identical.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the grammatical error detection task like so:

```bash
euroeval --model <model-id> --task grammatical-error-detection
```
