# Tool calling

## 📚 Overview

Tool calling is a task of using the right functions (picked from a list) and
using them with the right arguments - understanding the user request.

We use a uniform custom prompt and structured JSON output generation to obtain
predictions from a given model.

When evaluating generative models, we allow the model to generate 500 tokens on this
task.

## 📊 Metrics

We use simple accuracy where a prediction is positive only if the list of functions
and required arguments exactly matches one of the options given in the ground truth.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the tool calling task like so:

```bash
euroeval --model <model-id> --task tool-calling --language en
```

Currently only the english language is supported
