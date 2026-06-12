# Natural Language Inference

## 📚 Overview

Natural language inference (NLI) is the task of determining the logical relationship
between two statements: a *premise* and a *hypothesis*. The model must decide whether
the hypothesis is entailed by the premise (`entailment`), contradicts the premise
(`contradiction`), or has an indeterminate relationship with it (`neutral`).

This tests whether the model can reason about the consequences and entailments of
statements in natural language, including understanding causality, negation, and
world knowledge.

When evaluating generative models, we allow the model to generate 5 tokens on this task.

## 📊 Metrics

The primary metric used when evaluating the performance of a model on the natural
language inference task is [Matthews correlation
coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) (MCC),
which has a value between -100% and +100%, where 0% reflects a random guess. The primary
benefit of MCC is that it is balanced even if the classes are imbalanced.

We also report the macro-average [F1-score](https://en.wikipedia.org/wiki/F1_score),
being the average of the F1-score for each class, thus again weighing each class
equally.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the natural language inference task like so:

```bash
euroeval --model <model-id> --task natural-language-inference
```
