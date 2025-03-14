---
hide:
    - toc
---
# Tasks

👈 Choose a task on the left to see detailed information about that task.

## 📚 Overview

This page covers all the evaluation tasks used in EuroEval. These tasks fall under two
categories, corresponding to whether the models should merely _understand_ the input
documents (NLU), or rather they are also required to _generate_ new text (NLG).


### NLU Tasks

NLU tasks are tasks where the model is required to understand the natural language input
and provide an output based on this understanding. The outputs are typically very short,
often just a single label or a couple of words. The performance on these tasks is thus
relevant to you if you primarily aim to use the language models for processing documents
rather than generating entirely new documents. Both encoder and decoder models can be
evaluated on these tasks, enabling you to compare the performance across all language
models out there. The tasks in this category are:

1. [Sentiment Classification](sentiment-classification.md)
2. [Named Entity Recognition](named-entity-recognition.md)
3. [Linguistic Acceptability](linguistic-acceptability.md)
4. [Reading Comprehension](reading-comprehension.md)


### NLG Tasks

NLG tasks are tasks where the model is required to generate natural language output
based on some input. The outputs are typically longer than in NLU tasks, often multiple
paragraphs. The performance on these tasks is thus relevant to you if you aim to use the
language models for generating new documents. Only decoder models can be evaluated on
these tasks, as encoder models do not have the capability to generate text. The tasks in
this category are:

1. [Summarization](summarization.md)
2. [Knowledge](knowledge.md) ＊
3. [Common-sense Reasoning](common-sense-reasoning.md) ＊

＊ These tasks should be considered as NLU tasks, but currently encoder models have not
been set up to be evaluated on them. This will be added in a future version of
EuroEval - see the progress in [this
issue](https://github.com/EuroEval/EuroEval/issues/653).
