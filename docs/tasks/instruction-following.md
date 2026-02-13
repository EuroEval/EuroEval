# Instruction-following

## 📚 Overview

Instruction-following is a task of generating a response to a given instruction, after
which we check whether the model's response satisfied all the instructions. These
instructions are very specific and can be checked in a deterministic fashion. An example
of such an instruction is "Do not write any commas". Samples typically have more than
one instruction at the same time.

When evaluating generative models, we allow the model to generate 2,048 tokens on this
task.

## 📊 Metrics

The primary metric used to evaluate the performance of a model on the
instruction-following task is the instruction accuracy, which is the percentage of
constraints that the model's responses satisfied. Thus, if a sample has three different
instructions, and the model's response satisfies two of them, the instruction accuracy
for that sample is 2/3 = 66.67%. This metric is also named the instruction-level strict
accuracy metric in the original [IFEval
paper](https://doi.org/10.48550/arXiv.2311.07911).

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the instruction-following task like so:

```bash
euroeval --model <model-id> --task instruction-following
```
