# European Values

## 📚 Overview

This task is an _orthogonal task_, meaning that it does not contribute to the overall rank
score of the models on the leaderboards, but that it instead provides additional
insights into model behavior.

The European values task measures how much a model adheres to European values.
Concretely, these are multiple-choice questions taken from the official [European Values
Study](https://europeanvaluesstudy.eu/), which is a European cross-national survey
research program that explores people's values and beliefs, which has been going on
since the 1980s.

## 📊 Metrics

A kernel density estimate (KDE) is fitted to the survey responses from EU countries in
the most recent "wave" of the European Values Study, from 2017-2022. The
log-probabilities of the human responses under this distribution are computed, and based
on these a parametrised sigmoid transform is fitted to map log-probabilities to a score
between 0 and 1, where 1 corresponds to the highest density of human responses.

The model responses are then evaluated using the same KDE and sigmoid transform,
resulting in a score between 0 and 1.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the European values task like so:

```bash
euroeval --model <model-id> --task european-values
```
