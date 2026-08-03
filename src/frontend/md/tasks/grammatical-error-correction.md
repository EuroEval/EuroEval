# Grammatical Error Correction

## 📚 Overview

Grammatical error correction is the task of correcting the grammatical errors in a
given text. The model is presented with a sentence containing one or more grammatical
errors and has to generate a corrected version of the sentence, which is evaluated
against a reference correction. Unlike grammatical error detection, which only requires
identifying where the errors are, this task requires the model to produce the corrected
text itself.

When evaluating generative models, we allow the model to generate 256 tokens on this
task.

## 📊 Metrics

The metric used to evaluate the performance of a model on the grammatical error
correction task is exact match, being the percentage of generated corrections that are
identical to the reference correction. As the datasets for this task consist of minimal
pairs, where the corrected sentence differs from the erroneous sentence by only a small
edit, exact match is a fair and strict measure: similarity-based metrics such as chrF or
SARI would assign high scores to a model that simply copies its input, while exact
match only rewards producing the actual correction.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the grammatical error correction task like so:

```bash
euroeval --model <model-id> --task grammatical-error-correction
```
