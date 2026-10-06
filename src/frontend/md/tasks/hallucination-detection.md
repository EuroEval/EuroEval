# Hallucination Detection

## 📚 Overview

Hallucination detection measures the degree to which a model generates text that is not
supported by the provided context. The task is set up as a question answering task,
where the model is given a context, a question, and it has to generate an answer based
on the context. The generated answer is then classified at the token level to determine
how many tokens are hallucinated, i.e., not grounded in the context.

The hallucination detection is performed using the
[LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) library, which uses a
transformer-based classifier to predict hallucination at the token level. The method is
described by [Thoresen and Smart (2026)](https://doi.org/10.48550/arXiv.2605.02504).
The hallucination detection classifiers are trained on the publicly available
hallucination datasets
[MultiWikiQA-Synthetic-Hallucinations](https://huggingface.co/datasets/alexandrainst/multi-wiki-qa-synthetic-hallucinations)
and [Translated-RAGTruth-Hallucinations](https://huggingface.co/datasets/alexandrainst/ragtruth-translated-hallucinations).

When evaluating generative models, we allow the model to generate 512 tokens on this task.

## 📊 Metrics

The primary metric used to evaluate the performance of a model on the hallucination
detection task is the hallucination rate, computed as the ratio of hallucinated tokens to
total tokens in the generated answers. A lower hallucination rate indicates that the
model generates more faithful answers grounded in the provided context. On the leaderboard,
the rate is shown as an optional, orthogonal percentage: missing results are allowed, and
the value does not affect eligibility, rank, or the aggregate rank score.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the hallucination detection task like so:

```bash
euroeval --model <model-id> --task hallucination
```
