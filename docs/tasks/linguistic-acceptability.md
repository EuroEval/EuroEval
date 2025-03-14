# Linguistic Acceptability

## 📚 Overview

Linguistic acceptability is a task of determining whether a given text is grammatically
correct or not. It thus tests whether the model is able to understand the detailed
syntax of a given document, and not just understand the overall gist of it. It roughly
corresponds to when a native speaker would say "this sentence sounds weird".

When evaluating generative models, we allow the model to generate 5 tokens on this task.


## 📊 Metrics

The primary metric we use when evaluating the performance of a model on the linguistic
acceptability task, we use [Matthews correlation
coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) (MCC),
which has a value between -100% and +100%, where 0% reflects a random guess. The primary
benefit of MCC is that it is balanced even if the classes are imbalanced.

We also report the macro-average [F1-score](https://en.wikipedia.org/wiki/F1_score),
being the average of the F1-score for each class, thus again weighing each class
equally.


## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package.md), you
can benchmark your favorite model on the linguistic acceptability task like so:

```bash
$ euroeval --model <model-id> --task linguistic-acceptability
```
