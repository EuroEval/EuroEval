# Common-sense Reasoning

## 📚 Overview

Common-sense reasoning is testing whether a model is able to understand basic deduction
about the world. For instance, if the model is given the statement "It is raining
outside, and Peter is in his garden without an umbrella", it should be able to deduce
that Peter is getting wet. The task is set up as a multiple-choice question answering
task, where the model is given a question and a set of possible answers, and it has to
choose the correct answer.

When evaluating generative models, we allow the model to generate 5 tokens on this task.


## 📊 Metrics

The primary metric we use when evaluating the performance of a model on the common-sense
reasoning task, we use [Matthews correlation
coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) (MCC),
which has a value between -100% and +100%, where 0% reflects a random guess. The primary
benefit of MCC is that it is balanced even if the classes are imbalanced.

We also report the accuracy score, as this is the most common metric used for this task,
enabling comparisons with other benchmarks.


## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package.md), you
can benchmark your favorite model on the common-sense reasoning task like so:

```bash
$ euroeval --model <model-id> --task common-sense-reasoning
```
