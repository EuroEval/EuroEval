# Translation

## 📚 Overview

Translation is a task of translating a text from one language to another. The model
receives a text in one language and has to generate a text in another language.

When evaluating generative models, we allow the model to generate 256 tokens on this
task.

## 📊 Metrics

The primary metric used to evaluate the performance of a model on the translation task
is [CHRF3++](https://www.aclweb.org/anthology/W18-2346/), which measures the quality of
a translation by combining character-level n-gram F-scores with word order information.
The "++" indicates that it uses bi-grams (word_order=2) in addition to unigrams, and we
use beta=3 to weight precision and recall. CHRF is particularly well-suited for
translation evaluation as it is robust to paraphrasing and works well across different
languages.

We also report [CHRF4++](https://www.aclweb.org/anthology/W18-2346/), which uses
tri-grams (beta=4) for even more context-aware evaluation. Both metrics are computed
using SacreBLEU and are reported as percentages.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the translation task like so:

```bash
euroeval --model <model-id> --task translation
```
