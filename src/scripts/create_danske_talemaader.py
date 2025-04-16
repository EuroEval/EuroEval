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

"""Create the the Danske Talemåder dataset and upload it to the HF Hub."""

import io
from zipfile import ZipFile

import pandas as pd
import requests as rq
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the Danske Talemåder dataset and upload it to the HF Hub."""
    # Download the ZIP file
    url = (
        "https://sprogtek-ressources.digst.govcloud.dk/1000%20danske%20talemaader"
        "%20og%20faste%20udtryk/talemaader_csv.zip"
    )
    response = rq.get(url=url)
    response.raise_for_status()

    # Get the data from the ZIP file
    with ZipFile(file=io.BytesIO(initial_bytes=response.content)) as zip_file:
        no_labels_csv_file = [
            zip_file.read(name=file_name)
            for file_name in zip_file.namelist()
            if file_name == "talemaader_leverance_2_uden_labels.csv"
        ][0]
        only_labels_csv_file = [
            zip_file.read(name=file_name)
            for file_name in zip_file.namelist()
            if file_name == "talemaader_leverance_2_kun_labels.csv"
        ][0]

        no_labels_df = pd.read_csv(
            filepath_or_buffer=io.BytesIO(initial_bytes=no_labels_csv_file),
            delimiter="\t",
        )
        only_labels_df = pd.read_csv(
            filepath_or_buffer=io.BytesIO(initial_bytes=only_labels_csv_file),
            delimiter="\t",
        )

    # Set up the data as a dataframe
    df = pd.merge(left=no_labels_df, right=only_labels_df)
    df["text"] = [
        "Hvad betyder udtrykket '"
        + row.talemaade_udtryk.replace("\n", " ").strip()
        + "'?\n"
        "Svarmuligheder:\n"
        "a. " + row.A.replace("\n", " ").strip() + "\n"
        "b. " + row.B.replace("\n", " ").strip() + "\n"
        "c. " + row.C.replace("\n", " ").strip() + "\n"
        "d. " + row.D.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]
    df["label"] = df.korrekt_def.map({0: "a", 1: "b", 2: "c", 3: "d"})
    df = df[["text", "label"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create validation split
    val_size = 64
    traintest_arr, val_arr = train_test_split(df, test_size=val_size, random_state=4242)
    traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
    val_df = pd.DataFrame(val_arr, columns=df.columns)

    # Create train and test split
    train_size = 128
    train_arr, test_arr = train_test_split(
        traintest_df, train_size=train_size, random_state=4242
    )
    train_df = pd.DataFrame(train_arr, columns=df.columns)
    test_df = pd.DataFrame(test_arr, columns=df.columns)

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

    # Create dataset ID
    dataset_id = "EuroEval/danske-talemaader"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except rq.HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
