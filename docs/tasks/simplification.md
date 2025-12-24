# Simplification

## 📚 Overview

Simplification is a task of generating a simpler, easier to understand, version of a
given text, preserving the original
meaning but using less complex phrases and linguistic structures.
While the task is similar to summarisation, the goal is different: simplification
specifically rewrites the text to be
less complex and not necessarily shorter.

When evaluating generative models, we allow the model to generate 128 tokens on this
task.

## 📊 Metrics

The primary metric used to evaluate the performance of a model on the simplification
task
is [METEOR](https://aclanthology.org/W05-0909.pdf), which aligns
the reference with the generated text based on (chunked) unigrams, matching on exact
matches, stem matches, synonyms, and paraphrases, and then combining precision and
recall with a penalty for fragmented matches.
While originally a machine translation metric, the
generalized matching makes it effective for evaluating simplifications and is
shown to correlate well
with human judgment on grammaticality and meaning preservation.

We also report [SARI](https://aclanthology.org/Q16-1029.pdf), a specific text
simplification metric which compares the
generated simplified sentences
against the reference and the source sentence. It explicitly measures the quality of
words that are added, deleted and
kept compared to the source.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package.md), you
can benchmark your favorite model on the simplification task like so:

```bash
euroeval --model <model-id> --task simplification
```
