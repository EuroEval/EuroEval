# Contributing a Dataset to EuroEval

This guide will walk you through the process of contributing a new dataset to EuroEval.

For general contribution guidelines, please refer to our [Contributing Guide](CONTRIBUTING.md).

If you have any questions during this process, please open an issue on the [EuroEval GitHub repository](https://github.com/EuroEval/EuroEval/issues).


## Step 0: Prerequisites

Before beginning:
1. Check if your dataset matches [one of the supported tasks](https://euroeval.com/tasks/). If your dataset doesn't match any supported task, you have two options:
   1. Try to adapt it to fit an existing task (e.g., by reformatting it or adding multiple choice options)
   2. Open an issue on the EuroEval repository requesting to add a new task type
2. If it does, [fork the EuroEval repository](https://github.com/EuroEval/EuroEval/fork) and create a new branch to work on your dataset contribution


## Step 1: Create the Dataset Processing Script

Create a script in the `src/scripts` directory that processes your dataset into the EuroEval format.

The dataset creation script roughly follows this pattern:

```python
# Load your dataset.
raw_dataset = load_dataset("path_to_your_dataset")

# Process the dataset to fit the EuroEval format.
dataset = process_raw_dataset(raw_dataset=raw_dataset)

# Push the dataset to the Hugging Face Hub.
dataset.push_to_hub("EuroEval/your_dataset_name", private=True)
```

### Tips for Dataset Processing:
- Examine existing scripts for datasets with the same task for a reference on how to process your dataset.
- Take a look at [existing datasets in your language](https://euroeval.com/datasets/) to see how these are usually set up. Study these examples to understand the expected format and structure for your own dataset's entries.
- Split your dataset into train / val / test sets, ideally with 1,024 / 256 / 2,048 samples, respectively
- If your dataset already has splits, maintain consistency (e.g., the EuroEval train split should be a subset of the original train split)


## Step 2: Add Dataset Configuration

Dataset configurations in EuroEval are organised by language, with each language having its own file at `src/euroeval/dataset_configs/{language}.py`. A configuration is made with the `DatasetConfig` class. Here is an example for the fictive English Knowledge dataset `Rizzler`.

```python
RIZZLER_KNOWLEDGE_CONFIG = DatasetConfig(
    name="rizzler_knowledge", # The name of the dataset
    pretty_name="the truncated version of the English knowledge dataset Rizzler", # The pretty name of the dataset used in logs.
    huggingface_id="EuroEval/rizzler_knowledge", # The same id as used in the dataset creation script
    task=KNOW, # The task of the dataset
    languages=[EN], # The language of the dataset
    unofficial=True, # Whether the dataset is unofficial
)
```

Every `src/euroeval/dataset_configs/{language}.py` file has two sections:
- `### Official datasets ###`
- `### Unofficial datasets ###`

An unofficial dataset means that the resulting evaluation will not be included in the [official leaderboard](https://euroeval.com/leaderboards/).

As a starting point, make your dataset unofficial. This can always be changed later.


## Step 3: Document Your Dataset

Dataset documentation in EuroEval is organised by language, with each language having its own file at `docs/datasets/{language}.md`. Within each language file, documentation is further organised by task.

Navigate to the documentation file for your dataset's language and add your dataset's documentation in the appropriate task section.

The documentation should include the following information:

1. **General description**: Explain the dataset's origin and purpose
2. **Split details**: Describe how splits were created and their sizes
3. **Example samples**: Provide 3 representative examples from the training split
4. **Evaluation setup**: Explain how models are evaluated on this dataset
5. **Evaluation command**: Show how to evaluate a model on your dataset

To do this, you can follow these steps:
1. Find an existing dataset of the same task in `docs/datasets/{language}.md`
2. Copy the entire documentation section for that dataset
3. Use this as a template and modify all details to match your new dataset
4. Ensure you update all dataset-specific information (description, split sizes, example samples, etc.)


## Step 4: Modify the Change Log

After completing the previous steps, add an entry to the project's changelog to document your contribution. The entry should be added under the `[Unreleased]` section with a short description of the dataset you have added. Here is an example of a new entry.

```md
## [Unreleased]
### Added
- Added the English knowledge dataset [rizzler_knowledge](https://huggingface.co/datasets/Example-User/rizzler_knowledge). The split is given by 1,024 / 256 / 2,048 samples for train / val / test, respectively. It is marked as `unofficial` for now. This was contributed by [@your_name](https://github.com/your_name) ✨
```


## Step 5: Make a Pull Request

When you have completed all the previous steps, create a pull request to the EuroEval repository.


### Thank you!
This concludes the process of contributing a dataset to EuroEval. Your contribution helps expand the multilingual evaluation capabilities of the benchmark and is greatly appreciated by the research community!

Thank you for your valuable contribution! 🎉
