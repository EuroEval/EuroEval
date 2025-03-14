# Reading Comprenhension

## 📚 Overview

Reading comprehension is a task of determining whether a model is able to understand a
given text and answer questions about it. The model receives a text passage and a
question about the text, and it has to provide the answer as it is stated in the text.
This is very related to Retrieval-augmented Generation (RAG) applications, where a
generative model is used to answer a question based on one or more retrieved documents.

When evaluating generative models, we allow the model to generate 32 tokens on this
task.


## 📊 Metrics

The primary metric we use when evaluating the performance of a model on the reading
comprehension task is the exact match (EM) score, which is the percentage of questions
for which the model provides the exact answer.

We also report the [F1-score](https://en.wikipedia.org/wiki/F1_score) on a
character-basis, which is more lenient than the EM score, as it allows for small
differences in the answer.


## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package.md), you
can benchmark your favorite model on the reading comprehension task like so:

```bash
$ euroeval --model <model-id> --task reading-comprehension
```
