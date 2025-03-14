# Speed

## 📚 Overview

Speed is a task of measuring how quickly a model can process a given input. The model
receives text passages of varying lengths, and it has to process the documents as
quickly as possible, which includes tokenisation of the input. We let the model process
the input repeatedly for 3 seconds, and then we measure how quick it was. We use the
`pyinfer` package to perform the speed measurement.

The speed is of course very dependent on available hardware, and for APIs it also
fluctuates depending on the number of requests in the queue, so the speed benchmark
should be taken as only a rough estimate of the model's speed, rather than an exact
measurement.


## 📊 Metrics

The primary metric used to evaluate the performance of a model on the speed task is the
average number of GPT-2 tokens processed per second on GPUs, when the model is
processing documents with roughly 100, 200, ..., 1,000 tokens. If the model is only
accessible through an API then the speed is measured on the API. The GPUs used here
vary, depending on the size of the model - we preferably use an NVIDIA RTX 3090 Ti GPU,
if the model has less than ~8B parameters, and one or more NVIDIA A100 GPUs is larger.

The secondary metric is the same, but where the documents are shorter, with roughly
12.5, 15, ..., 125 tokens.


## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package.md), you
can benchmark your favorite model on the speed task like so:

```bash
$ euroeval --model <model-id> --task speed
```
