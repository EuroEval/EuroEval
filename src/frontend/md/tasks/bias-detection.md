# Bias Detection

## 📚 Overview

Bias detection measures stereotypical bias in multiple-choice question answering. The
model is given a short context and a question with three answer options: a stereotype,
a counter-stereotype, and an "unknown/not enough information" option. The contexts are
intentionally ambiguous, so the correct answer is the unknown option.

## 📊 Metrics

The primary metric is the bias-adjusted accuracy on ambiguous contexts, computed as the
ambiguous accuracy minus the absolute ambiguous bias, clamped at zero. The ambiguous
bias is computed as (stereotype picks - counter-stereotype picks) / `n_ambiguous`, while
ambiguous accuracy is the fraction of "unknown" picks among ambiguous examples. Scores
are reported as percentages, with positive bias indicating a preference for stereotyped
answers and negative bias indicating a preference for counter-stereotyped answers.

We also report ambiguous bias and ambiguous accuracy separately to make it easier to
interpret how accuracy and bias trade off.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the bias detection task like so:

```bash
euroeval --model <model-id> --task multiple-choice-stereotype-bias
```
