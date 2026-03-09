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
is [CHRF3++](https://www.aclweb.org/anthology/W18-2346/), which measures the quality of
a summary by combining character-level n-gram F-scores with word order information. The
"++" indicates that it uses bi-grams (word_order=2) in addition to unigrams, and we use
beta=3 to weight precision and recall. CHRF is particularly well-suited for
summarization as it is robust to paraphrasing and works well across different languages.

We also report [CHRF4++](https://www.aclweb.org/anthology/W18-2346/), which uses
tri-grams (beta=4) for even more context-aware evaluation. Both metrics are computed
using SacreBLEU and are reported as percentages.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the summarization task like so:

```bash
euroeval --model <model-id> --task summarization
```
