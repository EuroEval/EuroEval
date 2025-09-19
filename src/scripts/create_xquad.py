# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
#     "pandas==2.2.0",
# ]
# ///

"""Create the XQuAD datasets and upload them to the HF Hub."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_CONTEXT, MIN_NUM_CHARS_IN_CONTEXT
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from huggingface_hub.hf_api import HfApi
from tqdm.auto import tqdm

LANGUAGES = ["de", "en", "es"]


def main() -> None:
    """Create the XQuAD datasets and upload it to the HF Hub."""
    dataset_id = "google/xquad"

    for lang in tqdm(LANGUAGES):
        target_dataset_id = f"EuroEval/xquad-{lang}"

        dataset = load_dataset(
            dataset_id, name=f"xquad.{lang}", split="validation", keep_in_memory=True
        )
        assert isinstance(dataset, Dataset)

        df = dataset.to_pandas()

        # Ensure that `df` is indeed a Pandas DataFrame
        assert isinstance(df, pd.DataFrame)

        # Only work with samples where the context is not very large or small
        lengths = df.context.str.len()
        df = df[lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)]

        # Make splits
        val_size = 64
        test_size = 512

        val_df = df.sample(n=val_size, random_state=42)
        df = df.drop(val_df.index)
        test_df = df.sample(n=test_size, random_state=42)
        train_df = df.drop(test_df.index)
        assert len(train_df) > 500, "The training set should have at least 500 samples."

        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        # Push the dataset to the Hugging Face Hub
        HfApi().delete_repo(target_dataset_id, repo_type="dataset", missing_ok=True)
        dataset.push_to_hub(target_dataset_id, private=True)


if __name__ == "__main__":
    main()
