---
hide:
    - navigation
---
# The `euroeval` Python Package

The `euroeval` Python package is the Python package used to evaluate language models in
EuroEval. This page will give you a brief overview of the package and how to use it.
You can also check out the [full API reference](/api/euroeval/) for more details.

## Installation

To install the package simply write the following command in your favorite terminal:

```bash
pip install euroeval[all]
```

This will install the EuroEval package with all extras. You can also install the
minimal version by leaving out the `[all]`, in which case the package will let you know
when an evaluation requires a certain extra dependency, and how you install it.

## Quickstart

### Benchmarking

`euroeval` allows for benchmarking both via. script and using the command line.

/// tab | Using the command line

The easiest way to benchmark pretrained models is via the command line interface. After
having installed the package, you can benchmark your favorite model like so:

```bash
euroeval --model <model-id>
```

Here `model` is the HuggingFace model ID, which can be found on the [HuggingFace
Hub](https://huggingface.co/models). By default this will benchmark the model on all
the tasks available. If you want to benchmark on a particular task, then use the
`--task` argument:

```bash
euroeval --model <model-id> --task sentiment-classification
```

We can also narrow down which languages we would like to benchmark on. This can be done
by setting the `--language` argument. Here we thus benchmark the model on the Danish
sentiment classification task:

```bash
euroeval --model <model-id> --task sentiment-classification --language da
```

Multiple models, datasets and/or languages can be specified by just attaching multiple
arguments. Here is an example with two models:

```bash
euroeval --model <model-id1> --model <model-id2>
```

The specific model version/revision to use can also be added after the suffix '@':

```bash
euroeval --model <model-id>@<commit>
```

This can be a branch name, a tag name, or a commit id. It defaults to 'main' for latest.

See all the arguments and options available for the `euroeval` command by typing

```bash
euroeval --help
```

///

/// tab | Using a script

In a script, the syntax is similar to the command line interface. You simply initialise
an object of the `Benchmarker` class, and call this benchmark object with your favorite
model:

```python
>>> from euroeval import Benchmarker
>>> benchmarker = Benchmarker()
>>> benchmarker.benchmark(model="<model-id>")
```

To benchmark on a specific task and/or language, you simply specify the `task` or
`language` arguments, shown here with same example as above:

```python
>>> benchmarker.benchmark(
...     model="<model-id>",
...     task="sentiment-classification",
...     language="da",
... )
```

If you want to benchmark a subset of all the models on the Hugging Face Hub, you can
simply leave out the `model` argument. In this example, we're benchmarking all Danish
models on the Danish sentiment classification task:

```python
>>> benchmarker.benchmark(task="sentiment-classification", language="da")
```

///

/// tab | Using Docker

A Dockerfile is provided in the repo, which can be downloaded and run, without needing
to clone the repo and installing from source. This can be fetched programmatically by
running the following:

```bash
wget https://raw.githubusercontent.com/EuroEval/EuroEval/main/Dockerfile
```

Next, to be able to build the Docker image, first ensure that the NVIDIA Container
Toolkit is
[installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
and
[configured](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).
Ensure that the the CUDA version stated at the top of the Dockerfile matches the CUDA
version installed (which you can check using `nvidia-smi`). After that, we build the
image as follows:

```bash
docker build --pull -t euroeval .
```

With the Docker image built, we can now evaluate any model as follows:

```bash
docker run --rm --gpus 1 euroeval <euroeval-arguments>
```

Here `<euroeval-arguments>` consists of the arguments added to the `euroeval` CLI
argument. This could for instance be `--model <model-id> --task
sentiment-classification`.
///

## Benchmarking custom inference APIs

If the model you want to benchmark is hosted by a custom inference provider, such as a
[vLLM server](https://docs.vllm.ai/en/stable/), then this is also supported in EuroEval.

When benchmarking, you simply have to set the `--api-base` argument (`api_base` when
using the `Benchmarker` API) to the URL of the inference API, and optionally the
`--api-key` argument (`api_key`) to the API key, if authentication is required.

If you're benchmarking an Ollama model, then you're urged to add the prefix
`ollama_chat/` to the model name, as that will also fetch model metadata as well as pull
the models from the Ollama model repository before evaluating it, e.g.:

```bash
euroeval --model ollama_chat/mymodel --api-base http://localhost:11434
```

For all other OpenAI-compatible inference APIs, you simply provide the model name as
is, e.g.:

```bash
euroeval --model my-model --api-base http://localhost:8000
```

Again, if the inference API requires authentication, you simply add the `--api-key`
argument:

```bash
euroeval --model my-model --api-base http://localhost:8000 --api-key my-secret-key
```

If your model is a reasoning model, then you need to specify this as follows:

```bash
euroeval --model my-reasoning-model --api-base http://localhost:8000 --generative-type reasoning
```

Likewise, if it is a pretrained decoder model (aka a completion model), then you specify
this as follows:

```bash
euroeval --model my-base-decoder-model --api-base http://localhost:8000 --generative-type base
```

When using the `Benchmarker` API, the same applies. Here is an example of benchmarking
an Ollama model hosted locally:

```python
>>> benchmarker.benchmark(
...     model="ollama_chat/mymodel",
...     api_base="http://localhost:11434",
... )
```

## Benchmarking in an offline environment

If you need to benchmark in an offline environment, you need to download the models,
datasets and metrics beforehand. For example to download the model you want and all of
the Danish sentiment classification datasets:

/// tab | Using the command line
This can be done by adding the `--download-only` argument, from the command line:

```bash
euroeval --model <model-id> --task sentiment-classification --language da --download-only
```

///
/// tab | Using a script
This can be done using the `download_only` argument, if benchmarking from a script:

```python
benchmarker.benchmark(
  model="<model-id>",
  task="sentiment-classification",
  language="da",
  download_only=True,
)
```

///

!!! note
    Offline benchmarking of adapter models is not currently supported, meaning that we
    still require an internet connection during the evaluation of these. If offline
    support of adapters is important to you, please consider [opening an
    issue](https://github.com/EuroEval/EuroEval/issues).

## Benchmarking custom datasets

If you want to benchmark models on your own custom dataset, this is also possible.
First, you need to set up your dataset to be compatible with EuroEval. This means
splitting up your dataset in a training, validation and test split, and ensuring that
the column names are correct. We use `text` as the column name for the input text, and
the output column name depends on the type of task:

- **Text or multiple-choice classification**: `label`
- **Token classification**: `labels`
- **Reading comprehension**: `answers`
- **Free-form text generation**: `target_text`

Text and multiple-choice classification tasks are by far the most common. Then you can
decide whether your dataset should be accessible locally (good for testing, and good for
sensitive datasets), or accessible via the Hugging Face Hub (good for allowing others to
benchmark on your dataset).

/// tab | Local dataset

For a local dataset, you store your three dataset splits as three different CSV files
with the desired two columns, and then you create a file called `custom_datasets.py`
in which you define the associated `DatasetConfig` objects for your dataset. Here
is an example of a simple text classification dataset with two classes:

```python title="custom_datasets.py"
from euroeval import DatasetConfig, TEXT_CLASSIFICATION
from euroeval.languages import ENGLISH

MY_CONFIG = DatasetConfig(
    name="my-dataset",
    pretty_name="My Dataset",
    source=dict(train="train.csv", val="val.csv", test="test.csv"),
    task=TEXT_CLASSIFICATION,
    languages=[ENGLISH],
    labels=["positive", "negative"],
)
```

You can then benchmark your custom dataset by simply running

```bash
euroeval --dataset my-dataset --model <model-id>
```

You can also run the benchmark from a Python script, by simply providing your custom
dataset configuration directly into the `benchmark` method:

```python
from euroeval import Benchmarker

benchmarker = Benchmarker()
benchmarker.benchmark(model="<model-id>", dataset=MY_CONFIG)
```

///
/// tab | Hugging Face Hub dataset

For a dataset that is accessible via the Hugging Face Hub, you simply need to create a
file called `euroeval_config.py` in the root of your repository, in which you define
the associated dataset configuration. Note that you don't need to specify the `name`,
`pretty_name` or `source` arguments in this case, as these are automatically inferred
from the repository name. Here is an example of a simple text classification
dataset with two classes:

```python title="euroeval_config.py"
from euroeval import DatasetConfig, TEXT_CLASSIFICATION
from euroeval.languages import ENGLISH

CONFIG = DatasetConfig(
    task=TEXT_CLASSIFICATION,
    languages=[ENGLISH],
    labels=["positive", "negative"],
)
```

!!! note

    To benchmark a dataset from the Hugging Face Hub, you always need to set the
    `--trust-remote-code` flag (or `trust_remote_code=True` if using the `Benchmarker`),
    as the dataset configuration is loaded from the remote code. We advise you to always
    look at the code of the dataset configuration before running the benchmark.

You can then benchmark your custom dataset by simply running

```bash
euroeval --dataset <org-id>/<repo-id> --model <model-id> --trust-remote-code
```

You can also run the benchmark from a Python script, by simply providing your repo ID to
the `benchmark` method:

```python
from euroeval import Benchmarker

benchmarker = Benchmarker()
benchmarker.benchmark(
    model="<model-id>", dataset="<org-id>/<repo-id>", trust_remote_code=True
)
```

You can try it out with our [test
dataset](https://huggingface.co/datasets/EuroEval/test_dataset):

```bash
euroeval --dataset EuroEval/test_dataset --model <model-id> --trust-remote-code
```

///

We have included three convenience tasks to make it easier to set up custom datasets:

- `TEXT_CLASSIFICATION`, which is used for text classification tasks. This requires you
  to set the `labels` argument in the `DatasetConfig`, and requires the columns `text`
  and `label` to be present in the dataset.
- `MULTIPLE_CHOICE`, which is used for multiple-choice classification tasks. This
  also requires you to set the `labels` argument in the `DatasetConfig`. Note that for
  multiple choice tasks, you need to set up your `text` column to also list all the
  choices, and all the samples should have the same number of choices. This requires the
  columns `text` and `label` to be present in the dataset.
- `TOKEN_CLASSIFICATION`, which is used when classifying individual tokens in a text.
  This also require you to set the `labels` argument in the `DatasetConfig`. This
  requires the columns `tokens` and `labels` to be present in the dataset, where
  `tokens` is a list of tokens/words in the text, and `labels` is a list of the
  corresponding labels for each token (so the two lists have the same length).

On top of these three convenience tasks, there are of course also the tasks that we use
in the official benchmark, which you can use if you want to use one of these tasks with
your own bespoke dataset:

- `LA`, for linguistic acceptability datasets.
- `NER`, for named entity recognition datasets with the standard BIO tagging scheme.
- `RC`, for reading comprehension datasets in the SQuAD format.
- `SENT`, for sentiment classification datasets.
- `SUMM`, for text summarisation datasets.
- `KNOW`, for multiple-choice knowledge datasets (e.g., MMLU).
- `MCRC`, for multiple-choice reading comprehension datasets (e.g., Belebele).
- `COMMON_SENSE`, for multiple-choice common-sense reasoning datasets (e.g., HellaSwag).

These can all be imported from `euroeval.tasks` module.

### Creating your own custom task

You are of course also free to define your own task from scratch, which allows you to
customise the prompts used when evaluating generative models, for instance. Here is an
example of a custom free-form text generation task, where the goal for the model is to
generate a SQL query based on a natural language input:

```python title="custom_datasets.py"
from euroeval import DatasetConfig
from euroeval.data_models import Task, PromptConfig
from euroeval.enums import TaskGroup, ModelType
from euroeval.languages import ENGLISH
from euroeval.metrics import rouge_l_metric

sql_generation_task = Task(
    name="sql-generation",
    task_group=TaskGroup.TEXT_TO_TEXT,
    template_dict={
        ENGLISH: PromptConfig(
            default_prompt_prefix="The following are natural language texts and their "
            "corresponding SQL queries.",
            default_prompt_template="Natural language query: {text}\nSQL query: "
            "{target_text}",
            default_instruction_prompt="Generate the SQL query for the following "
            "natural language query:\n{text!r}",
            default_prompt_label_mapping=dict(),
        ),
    },
    metrics=[rouge_l_metric],
    default_num_few_shot_examples=3,
    default_max_generated_tokens=256,
    default_allowed_model_types=[ModelType.GENERATIVE],
)

MY_SQL_DATASET = DatasetConfig(
    name="my-sql-dataset",
    pretty_name="My SQL Dataset",
    source=dict(train="train.csv", val="val.csv", test="test.csv"),
    task=sql_generation_task,
    languages=[ENGLISH],
)
```

Again, with this you can benchmark your custom dataset by simply running

```bash
euroeval --dataset my-sql-dataset --model <model-id>
```

## Analysing the results of generative models

If you're evaluating a generative model and want to be able to analyse the model results
more in-depth, you can run your evaluation with the `--debug` flag (or `debug=True` if
using the `Benchmarker`), which will output all the model outputs and all the dataset
metadata (including the ground truth labels, if present) to both the terminal as well as
to a JSON file in your current working directory, named
`<model-id>-<dataset-name>-model-outputs.json`.

It is a JSON dictionary with keys being hashes of the input (which you can just ignore,
it's used for caching during generation), and values being dictionaries with the
following keys:

- `index`: The row index of the sample in the dataset. This allows you to match up the
  sample with the corresponding sample in the dataset.
- `text`/`messages`: The full input prompt used for generation. If the model is a base
  decoder then this will be a string stored in `text`, and if it's an instruction-tuned
  model then this will be an array of dictionaries stored in `messages`. This will
  include all few-shot examples, if any - see the below `prompt` to get the content of
  the present sample, without any few-shot examples.
- `prompt`: The actual example, without any few-shot examples. This is not exactly the
  input to the model (unless you're conducting zero-shot evaluation), but it can be
  handy to separate the actual query that the model was asked to answer.
- `sequence`: The generated sequence by the model.
- `predicted_label`: The predicted label for the generated sequence, if the task has a
  label. This allows you to compare directly with the ground truth label, if present.
- `scores`: An array of shape (`num_tokens_generated`, `num_logprobs_per_token`, 2),
  where the first dimension is the index of the token in the generated sequence, the
  second dimension is the index of the logprob for that token (ordered by most likely
  token to be generated to least likely), and the third dimension is a pair (token,
  logprob) for the token and its logprob. This will only be present if the task requires
  logprobs, and will otherwise be null.
- Any metadata for the sample that was present in the dataset, including the ground truth
  label, if present.

If you sort the rows by this index, you will get the samples in the same order as they
appear in the dataset, effectively just recreating the entire dataset, with the
additional model output features mentioned above. Here's an example of how you can do
this in Python:

```python
>>> import json
>>> import pandas as pd
>>> with open("<model-id>-<dataset-name>-model-outputs.json") as f:
...    model_outputs = json.load(f)
>>> df = pd.DataFrame(model_outputs.values()).set_index('index').sort_index()
>>> df.head()
      sequence predicted_label                                             scores          corruption_type      label                                           messages                                             prompt
index
0          nej       incorrect  [[[nej, -1.735893965815194e-05], [ja, -11.0000...         flip_nogle_nogen  incorrect  [{'content': 'Sætning: Styrkeforholdet må være...  Sætning: Peter Elmegaard med nogen af sine hæs...
0          nej       incorrect  [[[nej, -3.128163257315464e-07], [ja, -15.125]...         flip_nogle_nogen  incorrect  [{'content': 'Sætning: Ægteparret hævdede, at ...  Sætning: Peter Elmegaard med nogen af sine hæs...
0          nej       incorrect  [[[nej, -0.0009307525469921529], [ja, -7.00093...         flip_nogle_nogen  incorrect  [{'content': 'Sætning: Samtidig lægger hans on...  Sætning: Peter Elmegaard med nogen af sine hæs...
0          nej       incorrect  [[[nej, -4.127333340875339e-06], [ja, -12.5000...         flip_nogle_nogen  incorrect  [{'content': 'Sætning: Hej til Bente som jeg v...  Sætning: Peter Elmegaard med nogen af sine hæs...
1          nej       incorrect  [[[nej, 0.0], [ja, -16.75], [ne, -19.0], [n, -...  flip_indefinite_article  incorrect  [{'content': 'Sætning: Ægteparret hævdede, at ...  Sætning: Der blev afprøvet et masse ting.\n\nB...
```

Note that the `index` column is not unique, which is because the model is generating
multiple answers for each sample, but with different few-shot examples. You can see
these few-shot examples in the `messages` column.

??? example

    Here is a (truncated) example of a model output file:

    ```json
    {
      "cb3f9ea749fec9d2f83ca6d3a8744cce": {
        "index": 181,
        "sequence": "ja",
        "predicted_label": "correct",
        "scores": [
          [
            [
              "ja",
              -0.5232800841331482
            ],
            [
              "nej",
              -0.8982800841331482
            ],
            [
              "ne",
              -8.773280143737793
            ],
            [
              "j",
              -13.710780143737793
            ],
            [
              "n",
              -13.960780143737793
            ],
            [
              "!",
              -100.0
            ],
            [
              "\"",
              -100.0
            ],
            [
              "#",
              -100.0
            ]
          ]
        ],
        "corruption_type": null,
        "label": "correct",
        "messages": [
          {
            "content": "Sætning: Styrkeforholdet må være det afgørene, siger de begge.\n\nBestem om sætningen er grammatisk korrekt eller ej. Svar kun med 'ja' eller 'nej', og intet andet.",
            "role": "user"
          },
          {
            "content": "nej",
            "role": "assistant"
          },
          (...more few-shot examples...)
          {
            "content": "Sætning: Rør peberfrugt i og steg igen et par minutter.\n\nBestem om sætningen er grammatisk korrekt eller ej. Svar kun med 'ja' eller 'nej', og intet andet.",
            "role": "user"
          }
        ],
        "prompt": "Sætning: Rør peberfrugt i og steg igen et par minutter.\n\nBestem om sætningen er grammatisk korrekt eller ej. Svar kun med 'ja' eller 'nej', og intet andet."
      },
      "a8fab2c68e9ec63184636341eaf43f6c": {
        (...)
      },
      (...)
    }
    ```
