"""Create the Dutch CoLA dataset and upload it to the HF Hub."""

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Dutch CoLA dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "GroNLP/dutch-cola"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    dataset = (
        dataset.select_columns(["Sentence", "Acceptability"])
        .rename_columns({"Sentence": "text", "Acceptability": "label"})
        .shuffle(4242)
    )

    label_mapping = {0: "incorrect", 1: "correct"}
    dataset = dataset.map(lambda sample: {"label": label_mapping[sample["label"]]})
    dataset["val"] = dataset.pop("validation")

    full_dataset_id = "ScandEval/dutch-cola-full"
    dataset_id = "ScandEval/dutch-cola"

    # Remove the dataset from Hugging Face Hub if it already exists
    for id in [dataset_id, full_dataset_id]:
        try:
            api = HfApi()
            api.delete_repo(id, repo_type="dataset", missing_ok=True)
        except HTTPError:
            pass

    dataset.push_to_hub(full_dataset_id, private=True)
    
    # Convert the dataset to a dataframe
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Create validation split
    val_size = 256
    traintest_arr, val_arr = train_test_split(df, test_size=val_size, random_state=4242)
    traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
    val_df = pd.DataFrame(val_arr, columns=df.columns)

    # Create test split
    test_size = 2048
    train_arr, test_arr = train_test_split(
        traintest_df, test_size=test_size, random_state=4242
    )
    train_df = pd.DataFrame(train_arr, columns=df.columns)
    test_df = pd.DataFrame(test_arr, columns=df.columns)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(train_size, random_state=4242)

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
