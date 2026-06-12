---
hide:
    - navigation
---
# The `euroeval` Python Package

`euroeval` is the engine behind the EuroEval leaderboards. It runs the same task
definitions, prompts, metrics, and bootstrap confidence intervals that produce the
numbers on [euroeval.com](https://euroeval.com), so a result you produce locally is
directly comparable to anything published on the site.

You can drive it three ways:

- **Command line** — `euroeval --model <model-id>`; the fastest path to a result.
- **Python API** — `from euroeval import Benchmarker`; same arguments as keyword
  parameters, plus programmatic access to the returned `BenchmarkResult` objects.
- **Docker** — a self-contained image with all extras; useful for CI and shared GPUs.

Every CLI flag has a matching keyword argument on `Benchmarker.benchmark()`, so the two
interfaces stay in lockstep. For the complete list, run `euroeval --help` at your
terminal or browse the [full API reference](/api).

## Installation

```bash
pip install euroeval[all]
```

The `[all]` extra pulls in every optional dependency (vLLM, HuggingFace extras,
evaluation backends). If you'd rather stay minimal, install plain `euroeval` and the
package will print the exact `pip install` command you need the first time it hits a
missing dependency.

## Quickstart

/// tab | Command line

```bash
euroeval --model meta-llama/Llama-3.1-8B-Instruct
```

That's it. EuroEval picks every dataset compatible with the model (encoder vs decoder,
generative or not) across every supported language and writes one line per
(model, dataset) combination to `euroeval_benchmark_results.jsonl` in the current
directory.

Most runs narrow the scope with `--task` and/or `--language`:

```bash
euroeval --model <model-id> --task sentiment-classification --language da
```

Any flag can be repeated to fan out:

```bash
euroeval --model gpt-4o-mini --model claude-sonnet-4-5 --language en --language de
```

Pin a model revision (branch, tag, or commit) by suffixing `@`:

```bash
euroeval --model <model-id>@<sha>
```

Run `euroeval --help` for the complete flag list.

///
/// tab | Python

```python
from euroeval import Benchmarker

benchmarker = Benchmarker()
results = benchmarker.benchmark(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="sentiment-classification",
    language="da",
)
```

Every CLI flag maps to a snake_case keyword argument. Omitting `model` benchmarks every
matching model on the Hugging Face Hub — for example, every Danish model on the Danish
sentiment task:

```python
benchmarker.benchmark(task="sentiment-classification", language="da")
```

`benchmark()` returns a list of `BenchmarkResult` objects; see
[Output format](#output-format-every-eval-ever-eee) for the schema.

///
/// tab | Docker

```bash
wget https://raw.githubusercontent.com/EuroEval/EuroEval/main/Dockerfile
docker build --pull -t euroeval .
docker run --rm --gpus 1 euroeval --model <model-id>
```

Two prerequisites on the host:

- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
  is installed and configured for Docker.
- The CUDA version at the top of the `Dockerfile` matches the one reported by
  `nvidia-smi` on the host.

Everything after `euroeval` on the `docker run` line is passed through as CLI flags.

///

## Recipes

Concrete, end-to-end tasks. Each is self-contained — copy, adapt, run.

??? example "Evaluate a HF model on a single language"

    ```bash
    euroeval --model meta-llama/Llama-3.1-8B-Instruct --language da
    ```

??? example "Evaluate a model served by vLLM, Ollama, or another OpenAI-compatible API"

    Point `--api-base` at the inference server.  For Ollama, prefix the model name with
    `ollama_chat/` so EuroEval fetches metadata and triggers a `pull` for you:

    ```bash
    euroeval --model ollama_chat/mymodel --api-base http://localhost:11434
    ```

    For any other OpenAI-compatible API, pass the model name as-is and supply an API key if
    the server requires one:

    ```bash
    euroeval --model my-model --api-base http://localhost:8000 --api-key sk-secret
    ```

    If the served model is a reasoning model or a base (completion-only) decoder, declare
    it explicitly so EuroEval picks the right prompts:

    ```bash
    euroeval --model my-reasoning-model --api-base http://localhost:8000 --generative-type reasoning
    euroeval --model my-base-decoder    --api-base http://localhost:8000 --generative-type base
    ```

    The Python equivalent mirrors the flags as keyword arguments:

    ```python
    benchmarker.benchmark(
        model="ollama_chat/mymodel",
        api_base="http://localhost:11434",
    )
    ```

??? example "Evaluate a checkpoint stored on disk"

    If your model directory contains `config.json`, weights, and tokenizer files (i.e.
    anything `transformers.AutoModel.from_pretrained()` can load), point `--model` directly
    at the path — no upload, no inference server:

    ```bash
    euroeval --model /path/to/my-checkpoint
    ```

    This works for both encoder and decoder checkpoints. If you'd rather serve the model
    through an inference server and benchmark over HTTP, see the previous recipe.

??? example "Override missing or incorrect model metadata"

    Some models on the Hub report no maximum context length or vocabulary size, which leaves
    blanks on the leaderboard. Patch both at the command line:

    ```bash
    euroeval --model <model-id> --max-context-length 4096 --vocabulary-size 32000
    ```

    The Python equivalent:

    ```python
    benchmarker.benchmark(
        model="<model-id>",
        max_context_length=4096,
        vocabulary_size=32000,
    )
    ```

??? example "Run offline / in an air-gapped environment"

    First, pre-download everything you'll need with `--download-only`:

    ```bash
    euroeval --model <model-id> --task sentiment-classification --language da --download-only
    ```

    Then disconnect and re-run the same command without `--download-only` — the data,
    model, and metric files are now served from your Hugging Face cache. The Python form
    takes `download_only=True`.

    !!! note
        Adapter models still require a live connection during evaluation. If offline adapter
        support matters to you, please open an
        [issue](https://github.com/EuroEval/EuroEval/issues).

??? example "Evaluate on a private dataset (no upload)"

    Store your three splits as local CSVs and describe them in a `custom_datasets.py` next
    to the data:

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

    Then either reference the dataset by name on the CLI, or pass the config object
    directly from Python:

    ```bash
    euroeval --dataset my-dataset --model <model-id>
    ```

    ```python
    benchmarker.benchmark(model="<model-id>", dataset=MY_CONFIG)
    ```

    For the full set of options (column mapping, custom preprocessing, multiple-choice
    datasets, etc.), see [Custom datasets](#benchmarking-custom-datasets) below.

??? example "Publish a dataset on HF Hub so others can evaluate against it"

    Drop an `eval.yaml` at the root of your Hugging Face dataset repo — no Python code
    required, no `--trust-remote-code` flag. The format is shared with
    [Inspect AI's `eval.yaml`](https://inspect.aisi.org.uk/tasks.html#hugging-face), so the
    same file works for both frameworks. See
    [Hugging Face Hub dataset (YAML config)](#benchmarking-custom-datasets) for the schema.

??? example "Inspect why a generative model failed"

    Add `--debug` (or `debug=True` in Python). EuroEval writes the full prompts, model
    outputs, predicted labels, and logprobs to
    `<model-id>-<dataset-name>-model-outputs.json`. Load and sort it with pandas:

    ```python
    import json, pandas as pd

    with open("<model-id>-<dataset-name>-model-outputs.json") as f:
        data = json.load(f)
    df = pd.DataFrame(data.values()).set_index("index").sort_index()
    ```

    See [Detailed model outputs](#detailed-model-outputs) for the full schema and
    [Failed instances](#failed-instances) for parsing per-iteration failure lists.

??? example "Read existing results back into Python"

    The results file is JSON Lines in the
    [Every Eval Ever (EEE)](#output-format-every-eval-ever-eee) format. Loading every line
    back into a `BenchmarkResult` is a one-liner:

    ```python
    import json
    from euroeval.data_models import BenchmarkResult

    with open("euroeval_benchmark_results.jsonl") as f:
        results = [
            BenchmarkResult.from_dict(json.loads(line))
            for line in f
            if line.strip()
        ]
    ```

## Benchmarking custom datasets

If you want to benchmark models on your own custom dataset, this is also possible.
First, you need to set up your dataset to be compatible with EuroEval. This means
splitting up your dataset in a training, validation and test split. By default, EuroEval
expects these standard column names:

- **Text or multiple-choice classification**: `text` and `label`
- **Token classification**: `tokens` and `labels`
- **Reading comprehension**: `text` and `answers`
- **Free-form text generation**: `text` and `target_text`

If your dataset uses different column names, you can specify the mapping via
`input_column`, `target_column`, and `choices_column` in `DatasetConfig` (see
[Custom column names](#custom-column-names) below) — no need to rename your columns
beforehand.

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
/// tab | Hugging Face Hub dataset (YAML config)

The simplest and most secure way to add a EuroEval configuration to a Hugging Face Hub
dataset is via a YAML file. No Python code is written, so no `--trust-remote-code` flag
is required.

Create a file called `eval.yaml` in the root of your dataset repository. The file
follows the [Inspect AI `eval.yaml` format](https://inspect.aisi.org.uk/tasks.html#hugging-face)
and works with both Inspect AI and EuroEval:

```yaml title="eval.yaml"
name: My Dataset
tasks:
  - id: my_dataset
    split: test
    field_spec:
      input: review
      target: sentiment
    solvers:
      - name: generate
    scorers:
      - name: choice
# EuroEval-specific keys (optional; ignored by Inspect AI)
task: classification
languages:
  - en
labels:
  - positive
  - negative
```

The EuroEval-specific keys (`task`, `languages`, `labels`, and all other
`DatasetConfig` arguments) are placed at the top level alongside the standard Inspect
AI `tasks` block.  Inspect AI silently ignores keys it does not recognise, so the same
file works for both frameworks.

The value of `task` must be one of the task names used in EuroEval
(e.g. `classification`, `sentiment-classification`,
`named-entity-recognition`, `multiple-choice`, etc.).  `languages` is a list of
[ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) language codes.

All other `DatasetConfig` arguments are also supported:

```yaml title="eval.yaml"
name: My Dataset
tasks:
  - id: my_dataset
    split: test
    field_spec:
      input: review
      target: sentiment
    solvers:
      - name: generate
    scorers:
      - name: choice
# EuroEval-specific keys (optional; ignored by Inspect AI)
task: classification
languages:
  - en
labels:
  - positive
  - negative
num_few_shot_examples: 12
max_generated_tokens: 5
prompt_label_mapping:
  positive: positive
  negative: negative
```

The EuroEval-specific `task` and `languages` keys are **optional** — EuroEval will
infer them automatically when they are absent:

- **`task`** is inferred from the Inspect AI `tasks` block: a solver with
  `name: multiple_choice` **or** a `field_spec.choices` entry both map to the
  `multiple-choice` task.
- **`languages`** are read from the Hugging Face Hub repository metadata
  (the `language` field in the dataset card).  If the language cannot be determined,
  EuroEval defaults to English and logs a warning.

This means a standard Inspect AI `eval.yaml` with no EuroEval-specific keys works
out of the box:

```yaml title="eval.yaml"
# Pure Inspect AI format — no EuroEval keys required
name: My Dataset
description: My dataset description.
tasks:
  - id: my_dataset
    split: test
    field_spec:
      input: question
      target: answer
      choices: options
    solvers:
      - name: multiple_choice
    scorers:
      - name: choice
```

Column names can also be supplied as flat top-level keys (`input_column`,
`target_column`, `choices_column`) instead of inside the `field_spec` block;
top-level keys take precedence when both are present.  Note that Inspect AI allows
`field_spec.target` values such as `"literal:A"` (a hard-coded answer string) and
bare integers (mapped to letters A, B, C … by Inspect AI); EuroEval silently ignores
both forms because they are not column names.

The standard Inspect AI task keys are also used directly by EuroEval:

- **`tasks[0].split`** — the evaluation split to use (e.g. `test`, `validation`).
  EuroEval uses this as the test split, so no separate EuroEval key is needed.
- **`tasks[0].config`** — the Hugging Face dataset config/subset name (e.g. `main`,
  `default`).  EuroEval automatically appends it when loading the dataset.

You can then benchmark your custom dataset by simply running

```bash
euroeval --dataset <org-id>/<repo-id> --model <model-id>
```

or from a Python script:

```python
from euroeval import Benchmarker

benchmarker = Benchmarker()
benchmarker.benchmark(model="<model-id>", dataset="<org-id>/<repo-id>")
```

///
/// tab | Hugging Face Hub dataset (Python config)

For a dataset that is accessible via the Hugging Face Hub, you can also create a
file called `euroeval_config.py` in the root of your repository, in which you define
the associated dataset configuration. This gives you full Python flexibility (e.g.
custom preprocessing functions) but requires the `--trust-remote-code` flag.  Note
that you don't need to specify the `name`, `pretty_name` or `source` arguments in this
case, as these are automatically inferred from the repository name. Here is an example
of a simple text classification dataset with two classes:

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

    To benchmark a dataset from the Hugging Face Hub using a Python config, you always
    need to set the `--trust-remote-code` flag (or `trust_remote_code=True` if using
    the `Benchmarker`), as the dataset configuration is loaded from the remote code.
    We advise you to always look at the code of the dataset configuration before running
    the benchmark.

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

### Custom column names

If your dataset uses column names that differ from EuroEval's expected names, you can
specify a column mapping directly in `DatasetConfig` using the `input_column`,
`target_column`, and `choices_column` arguments. EuroEval will rename (or merge) the
columns at load time, so you don't need to preprocess your dataset beforehand.

**`input_column`** — the name of the column containing the input text. Defaults to
`"text"` (no rename). If set to a different value, that column is renamed to `"text"`.

```python
DatasetConfig(
    name="my-dataset",
    ...,
    input_column="review",   # rename "review" → "text"
)
```

**`target_column`** — the name of the column containing the label. If set, the column
is renamed to the task-appropriate standard name (`"label"` for classification,
`"labels"` for token classification, `"target_text"` for text-to-text).

```python
DatasetConfig(
    name="my-dataset",
    ...,
    target_column="sentiment",   # rename "sentiment" → "label" (for classification)
)
```

**`choices_column`** — for multiple-choice tasks, the column (or list of columns)
containing the answer choices. A single string names a column that holds a *list* of
choice strings. A list of strings names separate columns, each holding one choice
string. When set, the input text and choices are automatically merged into the
formatted `"text"` column that EuroEval expects.

```python
# Single column holding a list of choices
DatasetConfig(
    name="my-mcq-dataset",
    ...,
    choices_column="choices",
    target_column="answer",
)

# Separate columns, one per choice
DatasetConfig(
    name="my-mcq-dataset",
    ...,
    input_column="question",
    choices_column=["choice_a", "choice_b", "choice_c", "choice_d"],
    target_column="answer",
)
```

**`preprocessing_func`** — for full control, you can supply an arbitrary preprocessing
function that receives a `DatasetDict` and returns a `DatasetDict`. If this argument
is provided together with any of the column arguments above, `preprocessing_func` takes
precedence and the column arguments are ignored (a warning is logged in this case).

```python
def my_preprocess(dataset):
    for split_name, split in dataset.items():
        split = split.rename_column("review", "text")
        split = split.rename_column("stars", "label")
        dataset[split_name] = split
    return dataset

DatasetConfig(
    name="my-dataset",
    ...,
    preprocessing_func=my_preprocess,
)
```

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
customise the prompts used when evaluating generative models, for instance. When
creating a custom task you need to specify a `task_group`, which determines the overall
type of task and the required dataset columns. Below are examples for each supported
task group.

The `PromptConfig` object defines the prompts used for evaluation and accepts the
following arguments:

- `default_prompt_prefix`: Introductory text shown before the few-shot examples (only
  required for base decoders).
- `default_prompt_template`: Template used to format each example in few-shot
  evaluation (only required for base decoders). Available placeholders depend on the
  task group (see examples below).
- `default_instruction_prompt`: Template used for instruction-tuned models (zero-shot
  or instruction-style evaluation). Available placeholders depend on the task group (see
  examples below).
- `default_prompt_label_mapping`: A mapping from label strings to human-readable
  phrases used in the prompts (e.g., `{"b-per": "person"}`). Set to `"auto"` for a 1:1
  mapping or to an empty `dict()` for tasks that don't use labels in prompts.

/// tab | Sequence classification

**Task group**: `TaskGroup.SEQUENCE_CLASSIFICATION`

**Required dataset columns**: `text` (string), `label` (string)

The `label` column should contain the class label as a string. You must provide the
list of possible labels in the `DatasetConfig`.

**Available placeholders** in `PromptConfig`:

- `{text}`: The input text.
- `{label}`: The label for the example (empty string for the new sample).
- `{labels_str}`: A formatted string listing all possible labels.

```python title="custom_datasets.py"
from euroeval import DatasetConfig
from euroeval.data_models import Task, PromptConfig
from euroeval.enums import TaskGroup
from euroeval.languages import DANISH
from euroeval.metrics import mcc_metric, macro_f1_metric
from euroeval.constants import NUM_GENERATION_TOKENS_FOR_CLASSIFICATION

my_classification_task = Task(
    name="my-classification",
    task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
    template_dict={
        DANISH: PromptConfig(
            default_prompt_prefix="The following are texts and their categories, which "
            "can be {labels_str}.",
            default_prompt_template="Text: {text}\nCategory: {label}",
            default_instruction_prompt="Text: {text}\n\nClassify the text into one of "
            "the categories {labels_str}, and answer with only the category.",
            default_prompt_label_mapping="auto",
        ),
    },
    metrics=[mcc_metric, macro_f1_metric],
    default_num_few_shot_examples=12,
    default_max_generated_tokens=NUM_GENERATION_TOKENS_FOR_CLASSIFICATION,
    uses_logprobs=True,
)

MY_DATASET = DatasetConfig(
    name="my-classification-dataset",
    pretty_name="My Classification Dataset",
    source=dict(train="train.csv", val="val.csv", test="test.csv"),
    task=my_classification_task,
    languages=[DANISH],
    labels=["sports", "politics", "entertainment"],
)
```

///
/// tab | Multiple-choice classification

**Task group**: `TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION`

**Required dataset columns**: `text` (string), `label` (string)

The `label` column should be the letter of the correct choice (e.g., `"a"`). The `text`
column must include both the question and the formatted answer choices. You can either
pre-format the `text` column yourself, or use `choices_column` in `DatasetConfig` to
have EuroEval merge a separate choices column (or per-choice columns) into `text`
automatically. The merged format is:

```text
<question>
Choices:
a. <choice 0>
b. <choice 1>
...
```

All samples must have the same number of choices. You must provide the list of possible
label letters in the `DatasetConfig` (e.g., `["a", "b", "c", "d"]`).

**Available placeholders** in `PromptConfig`:

- `{text}`: The full question text including choices.
- `{label}`: The correct answer letter (empty string for the new sample).
- `{labels_str}`: A formatted string listing all possible answer letters.

```python title="custom_datasets.py"
from euroeval import DatasetConfig
from euroeval.data_models import Task, PromptConfig
from euroeval.enums import TaskGroup, ModelType
from euroeval.languages import FRENCH
from euroeval.metrics import mcc_metric, accuracy_metric
from euroeval.constants import NUM_GENERATION_TOKENS_FOR_CLASSIFICATION

my_multiple_choice_task = Task(
    name="my-multiple-choice",
    task_group=TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
    template_dict={
        FRENCH: PromptConfig(
            default_prompt_prefix="The following are multiple-choice questions "
            "(with answers).",
            default_prompt_template="Question: {text}\nAnswer: {label}",
            default_instruction_prompt="Question: {text}\n\nAnswer the question above "
            "by replying with {labels_str}, and nothing else.",
            default_prompt_label_mapping="auto",
        ),
    },
    metrics=[mcc_metric, accuracy_metric],
    default_num_few_shot_examples=5,
    default_max_generated_tokens=NUM_GENERATION_TOKENS_FOR_CLASSIFICATION,
    default_allowed_model_types=[ModelType.GENERATIVE],
    uses_logprobs=True,
)

# If your dataset has a single column with a list of choices and a separate answer column:
MY_DATASET = DatasetConfig(
    name="my-multiple-choice-dataset",
    pretty_name="My Multiple Choice Dataset",
    source=dict(train="train.csv", val="val.csv", test="test.csv"),
    task=my_multiple_choice_task,
    languages=[FRENCH],
    labels=["a", "b", "c", "d"],
    choices_column="choices",   # column containing a list of choice strings
    target_column="answer",     # column containing the correct answer letter
)

# Or if each choice is in its own column:
MY_DATASET = DatasetConfig(
    name="my-multiple-choice-dataset",
    pretty_name="My Multiple Choice Dataset",
    source=dict(train="train.csv", val="val.csv", test="test.csv"),
    task=my_multiple_choice_task,
    languages=[FRENCH],
    labels=["a", "b", "c", "d"],
    input_column="question",
    choices_column=["choice_a", "choice_b", "choice_c", "choice_d"],
    target_column="answer",
)
```

///
/// tab | Token classification

**Task group**: `TaskGroup.TOKEN_CLASSIFICATION`

**Required dataset columns**: `tokens` (list of strings), `labels` (list of strings)

The `tokens` column is a list of word tokens in the text, and the `labels` column is a
list of corresponding BIO tags (e.g., `["o", "b-per", "i-per", "o"]`). The two lists
must have the same length. You must provide the full list of possible labels (including
`"o"`) in the `DatasetConfig`. The `default_prompt_label_mapping` should map BIO labels
to human-readable category names, and for each entity type both `b-X` and `i-X` must map
to the same category string (e.g., `{"b-per": "person", "i-per": "person"}`).

**Available placeholders** in `PromptConfig`:

- `{text}`: The tokens joined into a string.
- `{label}`: A JSON dictionary mapping category names to lists of matching spans
  (empty string for the new sample).
- `{labels_str}`: A formatted string listing all category names from the label mapping.

```python title="custom_datasets.py"
from euroeval import DatasetConfig
from euroeval.data_models import Task, PromptConfig
from euroeval.enums import TaskGroup
from euroeval.languages import GERMAN
from euroeval.metrics import micro_f1_metric

my_token_classification_task = Task(
    name="my-token-classification",
    task_group=TaskGroup.TOKEN_CLASSIFICATION,
    template_dict={
        GERMAN: PromptConfig(
            default_prompt_prefix="Below are texts and JSON dictionaries with the "
            "categories that appear in the given text.",
            default_prompt_template="Text: {text}\nCategories: {label}",
            default_instruction_prompt="Text: {text}\n\nIdentify the categories in "
            "the text. Print this as a JSON dictionary with the keys being "
            "{labels_str}. The values should be lists of the spans of that category, "
            "exactly as they appear in the text.",
            default_prompt_label_mapping={
                "b-product": "product",
                "i-product": "product",
                "b-company": "company",
                "i-company": "company",
            },
        ),
    },
    metrics=[micro_f1_metric],
    default_num_few_shot_examples=8,
    default_max_generated_tokens=128,
    uses_structured_output=True,
)

MY_DATASET = DatasetConfig(
    name="my-token-classification-dataset",
    pretty_name="My Token Classification Dataset",
    source=dict(train="train.csv", val="val.csv", test="test.csv"),
    task=my_token_classification_task,
    languages=[GERMAN],
    labels=["o", "b-product", "i-product", "b-company", "i-company"],
)
```

///
/// tab | Question answering

**Task group**: `TaskGroup.QUESTION_ANSWERING`

**Required dataset columns**: `context` (string), `question` (string), `answers` (dict)

The `context` column is the passage to read, `question` is the question to answer, and
`answers` is a dict with `"text"` (a list of answer strings) and `"answer_start"` (a
list of character-level start positions of those answers in the context). This follows
the SQuAD format.

**Available placeholders** in `PromptConfig`:

- `{text}`: The context passage.
- `{question}`: The question.
- `{label}`: The answer text (empty string for the new sample).

```python title="custom_datasets.py"
from euroeval import DatasetConfig
from euroeval.data_models import Task, PromptConfig
from euroeval.enums import TaskGroup
from euroeval.languages import SWEDISH
from euroeval.metrics import f1_metric, em_metric

my_qa_task = Task(
    name="my-reading-comprehension",
    task_group=TaskGroup.QUESTION_ANSWERING,
    template_dict={
        SWEDISH: PromptConfig(
            default_prompt_prefix="Below are texts with questions and answers.",
            default_prompt_template="Text: {text}\nQuestion: {question}\nAnswer in "
            "max 3 words: {label}",
            default_instruction_prompt="Text: {text}\n\nAnswer the following question "
            "about the text above in max 3 words.\n\nQuestion: {question}",
            default_prompt_label_mapping=dict(),
        ),
    },
    metrics=[f1_metric, em_metric],
    default_num_few_shot_examples=4,
    default_max_generated_tokens=32,
)

MY_DATASET = DatasetConfig(
    name="my-reading-comprehension-dataset",
    pretty_name="My Reading Comprehension Dataset",
    source=dict(train="train.csv", val="val.csv", test="test.csv"),
    task=my_qa_task,
    languages=[SWEDISH],
)
```

///
/// tab | Text-to-text

**Task group**: `TaskGroup.TEXT_TO_TEXT`

**Required dataset columns**: `text` (string), `target_text` (string)

The `text` column is the input to the model, and `target_text` is the expected output.
This covers tasks such as summarization, translation, simplification, and free-form text
generation.

**Available placeholders** in `PromptConfig`:

- `{text}`: The input text.
- `{target_text}`: The expected output text (empty string for the new sample).

Here is an example of a custom text-to-text task where the goal is to generate a SQL
query from a natural language input:

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

///

With any of these custom tasks you can then benchmark your dataset by running

```bash
euroeval --dataset <dataset-name> --model <model-id>
```

## Output format: Every Eval Ever (EEE)

Each entry written to `euroeval_benchmark_results.jsonl` conforms to the
[Every Eval Ever (EEE) JSON schema v0.2.1](https://github.com/evaleval/every_eval_ever/blob/main/eval.schema.json),
a community-standard format for evaluation reporting.  The file can therefore be
consumed directly by any tool that understands the EEE schema.

Every line is a self-contained JSON object with the following top-level structure:

| Field | Description |
| --- | --- |
| `schema_version` | EEE schema version (`"0.2.1"`) |
| `evaluation_id` | Unique run identifier in `dataset/model/timestamp` format |
| `evaluation_timestamp` | ISO 8601 timestamp of when the evaluation ran |
| `retrieved_timestamp` | Unix epoch timestamp of when the record was written |
| `source_metadata` | EuroEval organisation info and evaluator relationship |
| `model_info` | Model identifier and EuroEval-specific metadata |
| `eval_library` | Library name/version and evaluation context |
| `evaluation_results` | Array of per-metric scores with uncertainty estimates |

Here is an abbreviated example:

```json
{
  "schema_version": "0.2.1",
  "evaluation_id": "angry-tweets/meta-llama/Llama-3.1-8B-Instruct/1741260000",
  "evaluation_timestamp": "2025-03-06T11:00:00+00:00",
  "retrieved_timestamp": "1741260000",
  "source_metadata": {
    "source_name": "EuroEval",
    "source_type": "evaluation_run",
    "source_organization_name": "EuroEval",
    "source_organization_url": "https://euroeval.com",
    "evaluator_relationship": "third_party"
  },
  "model_info": {
    "id": "meta-llama/Llama-3.1-8B-Instruct",
    "name": "meta-llama/Llama-3.1-8B-Instruct",
    "additional_details": {
      "num_model_parameters": "8000000000",
      "max_sequence_length": "131072",
      "vocabulary_size": "128256",
      "generative": "true",
      "generative_type": "instruction_tuned"
    }
  },
  "eval_library": {
    "name": "euroeval",
    "version": "16.16.1",
    "additional_details": {
      "task": "sentiment-classification",
      "languages": "[\"da\"]",
      "few_shot": "true",
      "raw_results": "[{\"test_mcc\": 0.40}, {\"test_mcc\": 0.45}]"
    }
  },
  "evaluation_results": [
    {
      "evaluation_name": "test_mcc",
      "source_data": { "dataset_name": "angry-tweets", "source_type": "hf_dataset" },
      "metric_config": {
        "lower_is_better": false,
        "score_type": "continuous",
        "min_score": 0,
        "max_score": 100
      },
      "score_details": {
        "score": 42.5,
        "details": { "num_failed_instances": "0.0" },
        "uncertainty": {
          "confidence_interval": {
            "lower": 41.3,
            "upper": 43.7,
            "confidence_level": 0.95,
            "method": "bootstrap"
          },
          "num_samples": 10
        }
      }
    }
  ]
}
```

!!! note
    The `_se` values stored internally are 95 % confidence interval half-widths
    (`1.96 × SE`).  They are exposed in the EEE output as a
    `confidence_interval { lower, upper, confidence_level: 0.95 }`, which is the
    correct EEE field for this statistic.

The EEE format can be read back into a `BenchmarkResult` object without any loss of
information:

```python
import json
from euroeval.data_models import BenchmarkResult

with open("euroeval_benchmark_results.jsonl") as f:
    for line in f:
        if line.strip():
            result = BenchmarkResult.from_dict(json.loads(line))
```

## Analysing the results of generative models

### Failed instances

When evaluating a generative model, some samples may fail silently — for example when
the model's output cannot be parsed as JSON (for NER tasks), or when no valid label can
be matched to the model's output (for classification tasks). These failures are recorded
in `euroeval_benchmark_results.jsonl` using the EEE format.

The total number of failed instances across all iterations is stored as a string in
`evaluation_results[N].score_details.details.num_failed_instances` for each metric.

The raw per-iteration scores, including per-iteration `failed_instances` lists, are
stored as a JSON-encoded string in `eval_library.additional_details.raw_results`. Each
element in the decoded list is a per-iteration score dictionary that may contain a
`failed_instances` key with a list of failed samples. Every item in the list has:

- `sample_index` — the 0-based index of the sample within the bootstrapped batch for
  that iteration.
- `error` — a short description of why the sample failed (e.g.
  `"Could not parse JSON from model output"` or
  `"No candidate label found in model output"`).

Here is an abbreviated example (omitting top-level EEE fields such as
`schema_version`, `evaluation_id`, `model_info`, etc. — see the
[Output format](#output-format-every-eval-ever-eee) section for the full structure):

```json
{
  "schema_version": "0.2.1",
  "evaluation_id": "ner-dataset/my-model/1741260000",
  "eval_library": {
    "name": "euroeval",
    "version": "16.17.0",
    "additional_details": {
      "raw_results": "[{\"micro_f1\": 0.82, \"failed_instances\": [{\"sample_index\": 4, \"error\": \"Could not parse JSON from model output\"}]}]"
    }
  },
  "evaluation_results": [
    {
      "evaluation_name": "test_micro_f1",
      "score_details": {
        "score": 0.82,
        "details": { "num_failed_instances": "3.0" }
      }
    }
  ]
}
```

If a model never fails (e.g. encoder/fine-tuned models, or a flawless generative run),
`num_failed_instances` will be `"0.0"` and `failed_instances` will be an empty list for
every iteration.

To inspect failed instances programmatically, you can load a result via
`BenchmarkResult.from_dict()`, which transparently decodes the EEE format (including
the JSON-encoded `raw_results` string) into a `BenchmarkResult` object. The decoded
per-iteration scores are then available via `result.results["raw"]`:

```python
import json
from euroeval.data_models import BenchmarkResult

with open("euroeval_benchmark_results.jsonl") as f:
    for line in f:
        if line.strip():
            # from_dict() decodes the EEE format transparently
            result = BenchmarkResult.from_dict(json.loads(line))
            raw = result.results.get("raw", [])
            for iteration_idx, iteration_scores in enumerate(raw):
                for failure in iteration_scores.get("failed_instances", []):
                    print(
                        f"Iteration {iteration_idx}, "
                        f"sample {failure['sample_index']}: {failure['error']}"
                    )
```

### Detailed model outputs

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
