# Summarization

## 📚 Overview

Summarization is a task of generating a shorter version of a given text, while
preserving the main points of the original text. The model receives a long text and has
to generate a shorter version of it, typically a handful of sentences long. This is
abstractive summarization, meaning that the summary typically do not appear verbatim in
the original text, but that the model has to generate new text based on the input.

When evaluating generative models, we allow the model to generate 256 tokens on this
task.


## 📊 Metrics

The primary metric used to evaluate the performance of a model on the summarization task
is the [BERTScore](https://doi.org/10.48550/arXiv.1904.09675), which uses a pretrained
encoder model to encode each token in both the reference summary and the generated
summary, and then uses cosine similarity to measure how the tokens match up. Using an
encoder model allows for the model to phrase a summary differently than the reference,
while still being rewarded for capturing the same meaning. We use the
`microsoft/mdeberta-v3-base` encoder model for all languages, as it is the best
performing encoder model consistently across all languages in the framework.

We also report the [ROUGE-L](https://www.aclweb.org/anthology/W04-1013/) score, which
measures the longest sequence of words that the generated summary and the reference
summary have in common. This is a more traditional metric for summarization, which is
why we report it as well, but it correlates less well with human judgments than
BERTScore.


## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package.md), you
can benchmark your favorite model on the summarization task like so:

```bash
$ euroeval --model <model-id> --task summarization
```
