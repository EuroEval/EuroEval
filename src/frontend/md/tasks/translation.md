# Translation

## 📚 Overview

Translation is a task of translating a text from one language to another. The model
receives a text in one language and has to generate a text in another language.

When evaluating generative models, we allow the model to generate 256 tokens on this
task.

## 📊 Metrics

The primary metric used to evaluate the performance of a model on the translation task
is [CHRF3++](https://www.aclweb.org/anthology/W18-2346/), which measures the quality of
a translation by combining character-level n-gram F-scores with word n-gram information.
The "++" indicates that it includes word n-grams up to bigrams (`word_order=2`), and the
"3" indicates that it uses `beta=3` to weight recall more than precision. CHRF is
particularly well-suited for translation evaluation as it is robust to paraphrasing and
works well across different languages.

We also report [CHRF4++](https://www.aclweb.org/anthology/W18-2346/), which uses the
same word-order setting but increases the recall weight to `beta=4`. Both metrics are
computed using SacreBLEU and are reported as percentages. For both metrics, per-sentence
scores are penalized if the predicted translation is not in the correct target language,
in which case the score for that sentence is set to 0.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you can
benchmark your favorite model on the translation task like so:

```bash
euroeval --model <model-id> --task translation
```
