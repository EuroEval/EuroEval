# Knowledge

## 📚 Overview

The knowledge task is testing how much factual knowledge a model has. The task is set up
as a multiple-choice question answering task, where the model is given a question and a
set of possible answers, and it has to choose the correct answer. Crucially, it is not
given any context in which the answer appears, so it has to answer purely based on its
knowledge of the world.

When evaluating generative models, we allow the model to generate 5 tokens on this task.


## 📊 Metrics

The primary metric we use when evaluating the performance of a model on the knowledge
task, we use [Matthews correlation
coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) (MCC),
which has a value between -100% and +100%, where 0% reflects a random guess. The primary
benefit of MCC is that it is balanced even if the classes are imbalanced.

We also report the accuracy score, as this is the most common metric used for this task,
enabling comparisons with other benchmarks.


## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package.md), you
can benchmark your favorite model on the knowledge task like so:

```bash
$ euroeval --model <model-id> --task knowledge
```
