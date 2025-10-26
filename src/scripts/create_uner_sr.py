# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "conllu==4.5.3",
# ]
# ///

"""Create the UNER-sr NER dataset and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the UNER-Serbian NER dataset and uploads it to the HF Hub."""
    # Load the Universal NER dataset
    repo_id = "universalner/universal_ner"

    # Load the Serbian subset from Universal NER
    dataset = load_dataset(repo_id, "sr_set", trust_remote_code=True)
    assert isinstance(dataset, DatasetDict)

    # Convert to dataframes
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Process dataframes
    train_df = process_df(df=train_df)
    val_df = process_df(df=val_df)
    test_df = process_df(df=test_df)

    # Make splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    final_val_df = val_df.sample(n=val_size, random_state=4242).reset_index(drop=True)
    remaining_val_samples = val_df[~val_df.index.isin(final_val_df.index)]
    test_df = pd.concat([test_df, remaining_val_samples], ignore_index=True)

    final_train_df = train_df.sample(n=train_size, random_state=4242).reset_index(
        drop=True
    )
    remaining_train_samples = train_df[~train_df.index.isin(final_train_df.index)]

    n_missing_test_samples = test_size - len(test_df)
    additional_test_samples = remaining_train_samples.sample(
        n=n_missing_test_samples, random_state=4242
    )
    final_test_df = pd.concat([test_df, additional_test_samples], ignore_index=True)
    assert len(final_test_df) == test_size

    final_train_df.reset_index(drop=True, inplace=True)
    final_val_df.reset_index(drop=True, inplace=True)
    final_test_df.reset_index(drop=True, inplace=True)

    # Create dataset dictionary
    dataset_dict = DatasetDict(
        train=Dataset.from_pandas(final_train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(final_val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(final_test_df, split=Split.TEST),
    )

    # Push to Hub
    dataset_id = "EuroEval/uner-sr-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process the DataFrame to extract tokens and labels.

    Args:
        df: The DataFrame to process.

    Returns:
        The processed DataFrame with tokens and labels columns.
    """
    # Rename ner_tags to labels for consistency
    df = df.rename(columns={"ner_tags": "labels"})

    # Convert numeric labels to string labels
    ner_conversion_dict = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
    }

    df["labels"] = df["labels"].apply(
        lambda tags: [ner_conversion_dict[tag] for tag in tags]
    )

    # Remove duplicates
    df["text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Keep only tokens and labels columns
    df = df[["tokens", "labels"]]
    return df


if __name__ == "__main__":
    main()
