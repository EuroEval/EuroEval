# Named Entity Recognition

## 📚 Overview

Named entity recognition is a task of determining the named entities in a given text,
such as named of persons, organisations, or locations. It thus both tests the knowledge
the model has about these things, as well as being able to extract multiple pieces of
information from a document at once.

When evaluating generative models, we allow the model to generate 128 tokens on this
task.


## 📊 Metrics

The primary metric we use when evaluating the performance of a model on the named entity
recognition task, we use the [micro-average
F1-score](https://en.wikipedia.org/wiki/F-score#Micro_F1) without MISC, computed as the
total number of true positives for all (non-trivial) entities except `MISC`, divided by
the total number of predicted positives for all entities except `MISC`.

We also report the micro-average F1-score, computed the same way, but where we include
the `MISC` entity as well. This is useful for comparing with other benchmarks, as it is
the most common metric used for this task. We find that excluding `MISC` gives a more
accurate picture of the model's performance, however, as the the `MISC` entity is not
well-defined and varies across datasets.


## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package.md), you
can benchmark your favorite model on the named entity recognition task like so:

```bash
$ euroeval --model <model-id> --task named-entity-recognition
```
