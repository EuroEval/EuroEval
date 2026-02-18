# Translation

## 📚 Overview

Translation is a task of translating a text from one language to another. The model
receives a text in one language and has to generate a text in another language.

When evaluating generative models, we allow the model to generate 256 tokens on this
task.

## 📊 Metrics

The primary metric used to evaluate the performance of a model on the translation task
is the [BERTScore](https://doi.org/10.48550/arXiv.1904.09675), which uses a pretrained
encoder model to encode each token in both the reference translation and the generated
translation, and then uses cosine similarity to measure how the tokens match up. Using an
encoder model allows for the model to phrase a translation differently than the reference
translation, while still being rewarded for capturing the same meaning. We use the
`microsoft/mdeberta-v3-base` encoder model for all languages, as it is the best
performing encoder model consistently across all languages in the framework.

We also report the [ROUGE-L](https://www.aclweb.org/anthology/W04-1013/) score, which
measures the longest sequence of words that the generated translation and the reference
translation have in common. This is a more traditional metric for translation, which is
why we report it as well, but it correlates less well with human judgments than
BERTScore.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the translation task like so:

```bash
euroeval --model <model-id> --task translation
```
