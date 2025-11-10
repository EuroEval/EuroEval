<!-- This disables the requirement that the first line is a top-level heading -->
<!-- markdownlint-configure-file { "MD041": false } -->

<div align='center'>
<img
    src="https://raw.githubusercontent.com/EuroEval/EuroEval/main/gfx/euroeval.png"
    height="500"
    width="372"
>
</div>

### The robust European language model benchmark

(formerly known as ScandEval)

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://euroeval.com)
[![PyPI Status](https://badge.fury.io/py/euroeval.svg)](https://pypi.org/project/euroeval/)
[![First paper](https://img.shields.io/badge/arXiv-2304.00906-b31b1b.svg)](https://arxiv.org/abs/2304.00906)
[![Second paper](https://img.shields.io/badge/arXiv-2406.13469-b31b1b.svg)](https://arxiv.org/abs/2406.13469)
[![License](https://img.shields.io/github/license/EuroEval/EuroEval)](https://github.com/EuroEval/EuroEval/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/EuroEval/EuroEval)](https://github.com/EuroEval/EuroEval/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-74%25-yellow.svg)](https://github.com/EuroEval/EuroEval/tree/main/tests)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/EuroEval/EuroEval/blob/main/CODE_OF_CONDUCT.md)

## Maintainer

- Dan Saattrup Smart ([@saattrupdan](https://github.com/saattrupdan), <dan.smart@alexandra.dk>)

## Installation

To install the package simply write the following command in your favorite terminal:

```bash
pip install euroeval[all]
```

This will install the EuroEval package with all extras. You can also install the
minimal version by leaving out the `[all]`, in which case the package will let you know
when an evaluation requires a certain extra dependency, and how you install it.

## Quickstart

### Benchmarking from the command line

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

### Benchmarking from a script

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

### Benchmarking from Docker

A Dockerfile is provided in the repo, which can be downloaded and run, without needing
to clone the repo and installing from source. This can be fetched programmatically by
running the following:

```bash
wget https://raw.githubusercontent.com/EuroEval/EuroEval/main/Dockerfile.cuda
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
docker build --pull -t euroeval -f Dockerfile.cuda .
```

With the Docker image built, we can now evaluate any model as follows:

```bash
docker run -e args="<euroeval-arguments>" --gpus 1 --name euroeval --rm euroeval
```

Here `<euroeval-arguments>` consists of the arguments added to the `euroeval` CLI
argument. This could for instance be `--model <model-id> --task
sentiment-classification`.

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
datasets and metrics beforehand. This can be done by adding the `--download-only`
argument, from the command line, or the `download_only` argument, if benchmarking from a
script. For example to download the model you want and all of the Danish sentiment
classification datasets:

```bash
euroeval --model <model-id> --task sentiment-classification --language da --download-only
```

Or from a script:

```python
>>> benchmarker.benchmark(
... model="<model-id>",
... task="sentiment-classification",
... language="da",
... download_only=True,
... )
```

Please note: Offline benchmarking of adapter models is not currently supported, meaning
that we still require an internet connection during the evaluation of these. If offline
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

Text and multiple-choice classification tasks are by far the most common. Next, you
store your three dataset splits as three different CSV files with the desired two
columns. Finally, you create a file called `custom_datasets.py` script in which you
define the associated `DatasetConfig` objects for your dataset. Here is an example of a
simple text classification dataset with two classes:

```python
from euroeval import DatasetConfig, TEXT_CLASSIFICATION
from euroeval.languages import ENGLISH

MY_CONFIG = DatasetConfig(
    name="my-dataset",
    pretty_name="My Dataset",
    source=dict(train="train.csv", val="val.csv", test="test.csv"),
    task=TEXT_CLASSIFICATION,
    languages=[ENGLISH],
    _labels=["positive", "negative"],
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

We have included three convenience tasks to make it easier to set up custom datasets:

- `TEXT_CLASSIFICATION`, which is used for text classification tasks. This requires you
  to set the `_labels` argument in the `DatasetConfig`, and requires the columns `text`
  and `label` to be present in the dataset.
- `MULTIPLE_CHOICE`, which is used for multiple-choice classification tasks. This
  also requires you to set the `_labels` argument in the `DatasetConfig`. Note that for
  multiple choice tasks, you need to set up your `text` column to also list all the
  choices, and all the samples should have the same number of choices. This requires the
  columns `text` and `label` to be present in the dataset.
- `TOKEN_CLASSIFICATION`, which is used when classifying individual tokens in a text.
  This also require you to set the `_labels` argument in the `DatasetConfig`. This
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

```python
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

## Reproducing the evaluation datasets

All datasets used in this project are generated using the scripts located in the
[src/scripts](src/scripts) folder. To reproduce a dataset, run the corresponding script
with the following command

```bash
uv run src/scripts/<name-of-script>.py
```

Replace <name-of-script> with the specific script you wish to execute, e.g.,

```bash
uv run src/scripts/create_allocine.py
```

## Contributors :pray:

A huge thank you to all the contributors who have helped make this project a success!

<a href="https://github.com/peter-sk">
    <img
        src="https://avatars.githubusercontent.com/u/6168908"
        width=50
        alt="Contributor avatar for peter-sk"
    />
</a>
<a href="https://github.com/AJDERS">
    <img
        src="https://avatars.githubusercontent.com/u/38854604"
        width=50
        alt="Contributor avatar for AJDERS"
    />
</a>
<a href="https://github.com/oliverkinch">
    <img
        src="https://avatars.githubusercontent.com/u/71556498"
        width=50
        alt="Contributor avatar for oliverkinch"
    />
</a>
<a href="https://github.com/versae">
    <img
        src="https://avatars.githubusercontent.com/u/173537"
        width=50
        alt="Contributor avatar for versae"
    />
</a>
<a href="https://github.com/KennethEnevoldsen">
    <img
        src="https://avatars.githubusercontent.com/u/23721977"
        width=50
        alt="Contributor avatar for KennethEnevoldsen"
    />
</a>
<a href="https://github.com/viggo-gascou">
    <img
        src="https://avatars.githubusercontent.com/u/94069687"
        width=50
        alt="Contributor avatar for viggo-gascou"
    />
</a>
<a href="https://github.com/mathiasesn">
    <img
        src="https://avatars.githubusercontent.com/u/27091759"
        width=50
        alt="Contributor avatar for mathiasesn"
    />
</a>
<a href="https://github.com/Alkarex">
    <img
        src="https://avatars.githubusercontent.com/u/1008324"
        width=50
        alt="Contributor avatar for Alkarex"
    />
</a>
<a href="https://github.com/marksverdhei">
    <img
        src="https://avatars.githubusercontent.com/u/46672778"
        width=50
        alt="Contributor avatar for marksverdhei"
    />
</a>
<a href="https://github.com/Mikeriess">
    <img
        src="https://avatars.githubusercontent.com/u/19728563"
        width=50
        alt="Contributor avatar for Mikeriess"
    />
</a>
<a href="https://github.com/ThomasKluiters">
    <img
        src="https://avatars.githubusercontent.com/u/8137941"
        width=50
        alt="Contributor avatar for ThomasKluiters"
    />
</a>
<a href="https://github.com/BramVanroy">
    <img
        src="https://avatars.githubusercontent.com/u/2779410"
        width=50
        alt="Contributor avatar for BramVanroy"
    />
</a>
<a href="https://github.com/peregilk">
    <img
        src="https://avatars.githubusercontent.com/u/9079808"
        width=50
        alt="Contributor avatar for peregilk"
    />
</a>
<a href="https://github.com/Rijgersberg">
    <img
        src="https://avatars.githubusercontent.com/u/8604946"
        width=50
        alt="Contributor avatar for Rijgersberg"
    />
</a>
<a href="https://github.com/duarteocarmo">
    <img
        src="https://avatars.githubusercontent.com/u/26342344"
        width=50
        alt="Contributor avatar for duarteocarmo"
    />
</a>
<a href="https://github.com/slowwavesleep">
    <img
        src="https://avatars.githubusercontent.com/u/44175589"
        width=50
        alt="Contributor avatar for slowwavesleep"
    />
</a>
<a href="https://github.com/mrkowalski">
    <img
        src="https://avatars.githubusercontent.com/u/6357044"
        width=50
        alt="Contributor avatar for mrkowalski"
    />
</a>

### Contribute to EuroEval

We welcome contributions to EuroEval! Whether you're fixing bugs, adding features, or
contributing new datasets, your help makes this project better for everyone.

- **General contributions**: Check out our [contribution guidelines](CONTRIBUTING.md)
  for information on how to get started.
- **Adding datasets**: If you're interested in adding a new dataset to EuroEval, we have
  a [dedicated guide](NEW_DATASET_GUIDE.md) with step-by-step instructions.

### Special thanks

- Thanks to [Google](https://google.com/) for sponsoring Gemini credits as part of their
  [Google Cloud for Researchers Program](https://cloud.google.com/edu/researchers).
- Thanks [@Mikeriess](https://github.com/Mikeriess) for evaluating many of the larger
  models on the leaderboards.
- Thanks to [OpenAI](https://openai.com/) for sponsoring OpenAI credits as part of their
  [Researcher Access Program](https://openai.com/form/researcher-access-program/).
- Thanks to [UWV](https://www.uwv.nl/) and [KU
  Leuven](https://www.arts.kuleuven.be/ling/ccl) for sponsoring the Azure OpenAI
  credits used to evaluate GPT-4-turbo in Dutch.
- Thanks to [Miðeind](https://mideind.is/en) for sponsoring the OpenAI
  credits used to evaluate GPT-4-turbo in Icelandic and Faroese.
- Thanks to [CHC](https://chc.au.dk/) for sponsoring the OpenAI credits used to
  evaluate GPT-4-turbo in German.

## Citing EuroEval

If you want to cite the framework then feel free to use this:

```bibtex
@article{smart2024encoder,
  title={Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on Multilingual NLU Tasks},
  author={Smart, Dan Saattrup and Enevoldsen, Kenneth and Schneider-Kamp, Peter},
  journal={arXiv preprint arXiv:2406.13469},
  year={2024}
}
@inproceedings{smart2023scandeval,
  author = {Smart, Dan Saattrup},
  booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
  month = may,
  pages = {185--201},
  title = {{ScandEval: A Benchmark for Scandinavian Natural Language Processing}},
  year = {2023}
}
```
