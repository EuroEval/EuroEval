# Classification

## 📚 Overview

Text classification assigns a label to a text according to the categories defined by a
dataset. It can be used for tasks such as topic, intent, or language classification.
EuroEval uses this generic task for datasets that do not fit one of its more specific
classification tasks.

When evaluating generative models, we allow the model to generate up to 10 tokens on
this task.

## 📊 Metrics

The primary metric is [Matthews correlation
coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) (MCC),
which has a value between -100% and +100%, where 0% reflects a random guess. MCC is
balanced even when the classes are imbalanced.

We also report the macro-average [F1-score](https://en.wikipedia.org/wiki/F1_score),
which gives each class equal weight.

## 🛠️ How to run

Specify the custom dataset together with the generic classification task in the
[EuroEval Python package](/python-package):

```bash
euroeval --model <model-id> --task classification --dataset <dataset-id>
```
