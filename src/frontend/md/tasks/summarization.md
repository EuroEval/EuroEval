# Summarization

## 📚 Overview

Summarization is a task of generating a shorter version of a given text while preserving
the main points of the original text. The model receives a long text and has to generate
a shorter version of it, typically a handful of sentences long. This is abstractive
summarization, meaning that the summary typically does not appear verbatim in the
original text, but is generated from the input.

When evaluating generative models, we allow the model to generate 256 tokens on this
task.

## 📊 Metrics

The primary metric used to evaluate the performance of a model on the summarization task
is [CHRF3++](https://aclanthology.org/W17-4770/), which measures the quality of a
summary by combining character-level n-gram F-scores with word n-gram information. The
"++" indicates that it includes word n-grams up to bigrams (`word_order=2`), and the "3"
indicates that it uses `beta=3` to weight recall more than precision. CHRF is
particularly well-suited for summarization as it is robust to paraphrasing and works
well across different languages.

We also report [CHRF4++](https://aclanthology.org/W17-4770/), which uses the same
word-order setting but increases the recall weight to `beta=4`. Both metrics are
computed using SacreBLEU and are reported as percentages. For both metrics, per-sentence
scores are penalized if the predicted summary is not in the correct target language, in
which case the score for that sentence is set to 0.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you can
benchmark your favorite model on the summarization task like so:

```bash
euroeval --model <model-id> --task summarization
```
