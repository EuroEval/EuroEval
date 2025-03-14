# Sentiment Classification

## 📚 Overview

Sentiment classification is a classical task of determining the sentiment of a given
text, which can be positive, negative, or neutral. It thus tests whether the model is
able to understand the overall semantics of a given document.

When evaluating generative models, we allow the model to generate 5 tokens on this task.


## 📊 Metrics

The primary metric we use when evaluating the performance of a model on the sentiment
classification task, we use [Matthews correlation
coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) (MCC),
which has a value between -100% and +100%, where 0% reflects a random guess. The primary
benefit of MCC is that it is balanced even if the classes are imbalanced.

We also report the macro-average F1-score, being the average of the
[F1-score](https://en.wikipedia.org/wiki/F1_score) for each class, thus again weighing
each class equally.


## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package.md), you
can benchmark your favorite model on the sentiment classification task like so:

```bash
$ euroeval --model <model-id> --task sentiment-classification
```
