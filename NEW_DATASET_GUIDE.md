# Contributing a Dataset to EuroEval

This guide describes how to contribute a new dataset to EuroEval.

For general contribution guidelines, see the [Contributing Guide](CONTRIBUTING.md). If
you have questions during this process, open an issue on the
[EuroEval GitHub repository](https://github.com/EuroEval/EuroEval/issues).

## Step 0: Prerequisites

Before beginning:

1. Check if your dataset matches
   [one of the supported tasks](https://euroeval.com/tasks/). If your dataset does not
   match any supported task, you have two options:
   1. Adapt it to fit an existing task, for example by reformatting it or adding
      multiple-choice options.
   2. Open an issue on the EuroEval repository requesting a new task type.
2. [Fork the EuroEval repository](https://github.com/EuroEval/EuroEval/fork) and create
   a new branch for your dataset contribution.

## Step 1: Create the dataset processing script

Create a script in `src/scripts/dataset_creation` that processes your dataset into the
EuroEval format.

The dataset creation script roughly follows this pattern:

```python
from datasets import load_dataset

# Load your dataset.
raw_dataset = load_dataset("path_to_your_dataset")

# Process the dataset to fit the EuroEval format.
dataset = process_raw_dataset(raw_dataset=raw_dataset)

# Push the dataset to the Hugging Face Hub.
dataset.push_to_hub("EuroEval/your_dataset_name", private=True)
```

### Tips for dataset processing

- Examine existing scripts for datasets with the same task as a reference.
- Look at [existing datasets in your language](https://euroeval.com/datasets/) to see
  the expected format and structure for dataset entries.
- Use the standard EuroEval column names where possible:
  - `text` for the input text.
  - `label`, `labels`, or `target_text`, depending on the task group.
- Split your dataset into train, validation, and test splits. A common target is 1,024 /
  256 / 2,048 samples for train / validation / test, respectively, but smaller or
  different split sizes are fine when the source data requires it.
- If your dataset already has splits, preserve their semantics. For example, the
  EuroEval train split should be a subset of the original train split.

## Step 2: Add the dataset configuration

Dataset configurations in EuroEval are organised by language. Each language has its own
file at `src/euroeval/dataset_configs/{language}.py`. Add your configuration to the file
for the dataset's main language.

A dataset configuration is created with `DatasetConfig`. Import the language constants
from `src/euroeval/languages.py` and the task constants from `src/euroeval/tasks.py`.
For example, an English knowledge dataset uses `ENGLISH` and `KNOW`:

```python
from ..data_models import DatasetConfig
from ..languages import ENGLISH
from ..tasks import KNOW

RIZZLER_KNOWLEDGE_CONFIG = DatasetConfig(
    name="rizzler-knowledge",
    pretty_name="Rizzler Knowledge",
    source="EuroEval/rizzler-knowledge",
    task=KNOW,
    languages=[ENGLISH],
    unofficial=True,
)
```

Set these fields for most new datasets:

- `name`: The dataset ID used in EuroEval. Use lowercase with hyphens, and keep it
  unique across all datasets.
- `pretty_name`: The human-readable dataset name used in logs and documentation.
- `source`: The Hugging Face dataset ID, such as `EuroEval/rizzler-knowledge`, or a
  dictionary of local CSV paths for custom local datasets.
- `task`: The task constant, such as `SENT`, `KNOW`, `RC`, `MCRC`, or `SUMM`.
- `languages`: A list of language constants, such as `[ENGLISH]`.
- `unofficial=True`: New datasets should usually start as unofficial.

Use these optional fields when they apply:

- `train_split`, `val_split`, and `test_split` if the source split names differ from
  `train`, `val`, and `test`, or if a split is missing. Use `None` for missing train or
  validation splits.
- `labels` or `prompt_label_mapping` if the dataset has task-specific labels.
- `input_column`, `target_column`, and `choices_column` if the source dataset can be
  mapped to the EuroEval format by renaming or combining columns.
- `preprocessing_func` if the dataset needs custom preprocessing. Do not use this
  together with the column-mapping arguments unless you intentionally want the custom
  preprocessing function to take precedence.
- `bootstrap_samples=False` for datasets where samples should not be bootstrapped, such
  as some zero-shot evaluation datasets.

Each `src/euroeval/dataset_configs/{language}.py` file has two sections:

- `# Official datasets ###`
- `# Unofficial datasets ###`

An unofficial dataset is not included in the
[official leaderboard](https://euroeval.com/leaderboards/) unless it is explicitly
selected for evaluation. As a starting point, place your dataset in the unofficial
section and set `unofficial=True`. This can be changed later.

## Step 3: Document your dataset

Dataset documentation in EuroEval is organised by language. Each language has its own
file at `src/frontend/md/datasets/{language}.md`. Within each language file,
documentation is organised by task. Within a task section, official datasets come first,
followed by unofficial datasets.

Navigate to the documentation file for your dataset's language and add your dataset's
documentation in the appropriate task section. If your dataset is unofficial, prefix the
dataset heading with "Unofficial: ", for example `### Unofficial: Rizzler Knowledge`.

The documentation should include:

1. **General description**: Explain the dataset's origin and purpose.
2. **Split details**: Describe how splits were created and their sizes.
3. **Example samples**: Provide three representative examples from the training split,
   if there is a training split.
4. **Evaluation setup**: Explain how models are evaluated on this dataset.
5. **Evaluation command**: Show how to evaluate a model on your dataset.

To do this:

1. Find an existing dataset of the same task in
   `src/frontend/md/datasets/{language}.md`.
2. Copy the full documentation section for that dataset.
3. Use it as a template and modify all details to match your dataset.
4. Update all dataset-specific information, including description, split sizes, example
   samples, and evaluation commands.

## Step 4: Update the changelog

Add an entry under the existing `[Unreleased]` section in `CHANGELOG.md`. Use the
`### Added` subsection unless another subsection is more appropriate.

For example:

```md
### Added

- Added the English knowledge dataset rizzler-knowledge, measuring knowledge about
  rizzling. It is marked as `unofficial` for now. This was contributed by
  @your-github-username ✨
```

Do not create a second `[Unreleased]` header if one already exists.

## Step 5: Validate the contribution

Before submitting the pull request, run the relevant checks locally:

```bash
uv run pytest tests/test_dataset_configs.py
CHECK_DATASET=rizzler-knowledge uv run pytest tests/test_data_loading.py
make check
```

Run `make test` as well if the change is large or touches code beyond the dataset
configuration and documentation. The full test suite can take a long time.

## Step 6: Make a pull request

When you have completed the previous steps, create a pull request to the EuroEval
repository. Follow the pull request guidance in [CONTRIBUTING.md](CONTRIBUTING.md):

- Fill the "Ready for review" template.
- Link the pull request to the issue if it solves one.
- Enable maintainer edits on your fork branch.
- Respond to review comments and mark resolved conversations as resolved.

Thank you for helping expand EuroEval's multilingual evaluation coverage.
