# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the MMLU-hr dataset and upload to HF Hub."""

from collections import Counter

import pandas as pd
from constants import (
    CHOICES_MAPPING,
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the MMLU-hr dataset and upload to HF Hub."""
    # Read source dataset
    repo_id = "aialt/MuBench"
    dataset = load_dataset(path=repo_id, name="MMLUDataset_local_template_hr")
    df_test = dataset["test"].to_pandas()
    df_validation = dataset["validation"].to_pandas()
    df = pd.concat([df_test, df_validation], ignore_index=True)

    # Extract the question and options from the prompt
    df["instruction"], df["choices"] = zip(*df["prompt"].apply(extract_prompt_details))

    # Convert the labels to single lowercase letters
    df.label = df.label.map(lambda x: "abcd"[x])

    # Double check that there are exactly 4 choices per question
    assert df.choices.apply(len).eq(4).all(), "Not all questions have exactly 4 choices"

    # Split up the choices into separate columns
    df["option_a"] = df.choices.apply(lambda x: x[0])
    df["option_b"] = df.choices.apply(lambda x: x[1])
    df["option_c"] = df.choices.apply(lambda x: x[2])
    df["option_d"] = df.choices.apply(lambda x: x[3])

    # Remove the samples with overly short or long texts
    df = df[
        (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & (df.option_a.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_a.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_c.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_c.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_d.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_d.str.len() <= MAX_NUM_CHARS_IN_OPTION)
    ]

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    df = df[
        ~df.instruction.apply(is_repetitive)
        & ~df.option_a.apply(is_repetitive)
        & ~df.option_b.apply(is_repetitive)
        & ~df.option_c.apply(is_repetitive)
        & ~df.option_d.apply(is_repetitive)
    ]
    assert isinstance(df, pd.DataFrame)

    # Make a `text` column with all the options in it
    df["text"] = [
        row.instruction.replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING['hr']}:\n"
        "a. " + row.option_a.replace("\n", " ").strip() + "\n"
        "b. " + row.option_b.replace("\n", " ").strip() + "\n"
        "c. " + row.option_c.replace("\n", " ").strip() + "\n"
        "d. " + row.option_d.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Make the `label` column case-consistent with the `text` column
    df.label = df.label.str.lower()

    # Only keep the `text` and `label` columns
    df = df[["text", "label"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Make splits
    val_size = 256
    train_size = 1024
    test_size = 2048

    df = df.sample(frac=1, random_state=4242).reset_index(drop=True)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size : train_size + val_size].reset_index(drop=True)
    test_df = df.iloc[
        train_size + val_size : train_size + val_size + test_size
    ].reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/mmlu-hr-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def extract_prompt_details(prompt: str) -> tuple[str, list[str]]:
    """Extract the question and options from the prompt.

    Args:
        prompt: The prompt to extract the question and options from.

    Returns:
        A tuple of the question and options.
    """
    lines = prompt.split("\n")
    question: str = ""
    options: list[str] = []

    for line in lines:
        line = line.strip()
        if line.startswith("Pitanje:"):
            question = line.replace("Pitanje:", "").strip()
        elif line.startswith("Izbor A:"):
            options.append(line.replace("Izbor A:", "").strip())
        elif line.startswith("Izbor B:"):
            options.append(line.replace("Izbor B:", "").strip())
        elif line.startswith("Izbor C:"):
            options.append(line.replace("Izbor C:", "").strip())
        elif line.startswith("Izbor D:"):
            options.append(line.replace("Izbor D:", "").strip())

    return question, options


if __name__ == "__main__":
    main()
