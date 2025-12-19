# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "evaluate==0.4.1",
#     "bert-score==0.3.13",
# ]
# ///

"""Create the DuidelijkeTaal simplification dataset by filtering the original dataset.

Original data is published at http://hdl.handle.net/10032/tm-a2-y8 and
https://huggingface.co/datasets/instituutnederlandsetaal/DuidelijkeTaal-v1.0.

We use the splitted version of this dataset created during the GPT-NL project
(https://huggingface.co/datasets/GPT-NL/DuidelijkeTaal-v1.0-split).
"""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from evaluate import load
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Creates Duidelijke Taal simplification dataset and uploads it to the HF Hub."""
    dataset_id = "GPT-NL/DuidelijkeTaal-v1.0-split"

    # rename columns for easier use in filtering
    column_mapping = {
        "Wordpress ID": "wordpress_id",
        "Document ID": "doc_id",
        "Niet synthetische tekst/zin (A)": "text",
        "Synthetische tekst/zin (B)": "target_text",
        "Paarsgewijze vergelijking": "clarity",
        "Paarsgewijze vergelijking Gem.": "clarity_avg",
        "Accuratesse": "acc",
        "Accuratesse Gem.": "acc_avg",
        "Fluency (A)": "fluency_a",
        "Fluency (A) Gem.": "fluency_a_avg",
        "Fluency (B)": "fluency_b",
        "Fluency (B) Gem.": "fluency_b_avg",
        "Complexiteit (A)": "complexity_a",
        "Complexiteit (A) Gem.": "complexity_a_avg",
        "Complexiteit (B)": "complexity_b",
        "Complexiteit (B) Gem.": "complexity_b_avg",
    }

    dataset_original = load_dataset(dataset_id, token=True)

    # create 50%/50% train/val split
    # original split is 50%/50% train/test, so total split becomes 25%/25%/50%
    train_val = dataset_original["train"].train_test_split(test_size=0.5, seed=42)

    dataset = DatasetDict(
        {
            "train": train_val["train"],
            "val": train_val["test"],
            "test": dataset_original["test"],
        }
    )

    for split in ["train", "val", "test"]:
        df = dataset[split].to_pandas().rename(columns=column_mapping)
        df_filtered = filter_dataset(df)[["text", "target_text"]]
        dataset[split] = Dataset.from_pandas(df_filtered)

    processed_dataset_id = "EuroEval/duidelijke-taal-nl"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(processed_dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(processed_dataset_id, private=True)


def filter_dataset(
    df: pd.DataFrame,
    min_complexity_difference: int = 10,
    min_n_participants: int = 2,
    min_acc_avg: int = 60,
    min_bertscore: float = 0.76,
    drop_explanations: bool = True,
    verbosity_threshold: int = 2,
) -> pd.DataFrame:
    """Filter Duidelijke Taal dataset.

    Dataset is filtered based on complexity difference between original sentence and
    synthesized simplification, number of human annotations and average accuracy of
    the simplification.

    Parameters
    ----------
    df : pandas.DataFrame
        The unfiltered Duidelijke Taal dataset.
    min_complexity_difference : int, default=10
        The minimum difference in complexity between original and simplification, to
        create a subset with clearly differentiated examples.
    min_n_participants : int, default=2
        Minimum number of participants from the crowd-sourcing experiment that rated
        the sentence pair to ensure we take a somewhat general opinion.
    min_acc_avg : int, default=60
        Minimum average accuracy percentage of the simplification to filter out
        inaccurate simplifications.
    min_bertscore: int, default=0.76
        Minimum BERTScore (F1) similarity between the sentence pairs to filter pairs
        that vary too much in meaning.
    drop_explanations: bool, default=True
        Flag to set to drop pairs where the synthetic example contains explanations
        that are not in the original sentence (e.g. "Dat betekent dat...")
    verbosity_threshold: int, default=2
        Threshold with the intention to drop pairs where the synthetic example contains
        too verbose and too long explanations that deviate from the original input text.
        Drops simplifications that contain more than threshold * number of words of the
        input text. Setting this parameter to 1 will not filter any sentences.

    Returns:
    -------
    pandas.DataFrame
        The filtered dataset.
    """
    bert_score = load("bertscore")
    bert_scores = bert_score.compute(
        references=df["text"].to_list(),
        predictions=df["target_text"].to_list(),
        lang="nl",
    )["f1"]
    df["bert"] = bert_scores

    df["complexity_diff"] = df[df["complexity_a_avg"].astype(str).str.len() > 0][
        "complexity_a_avg"
    ].astype(float) - df[df["complexity_b_avg"].astype(str).str.len() > 0][
        "complexity_b_avg"
    ].astype(float)

    df_mask = (
        (
            (~df["complexity_diff"].isna()) & (df["complexity_diff"] > 0)
        )  # only keep the sentences where original is harder
        & (df["complexity_diff"].abs() > min_complexity_difference)
        & (
            df["complexity_a"].apply(
                lambda x: (len(x) >= min_n_participants if x is not None else False)
            )
        )
        & (
            df["complexity_b"].apply(
                lambda x: (len(x) >= min_n_participants if x is not None else False)
            )
        )
        & (df["acc_avg"].apply(lambda x: x != "" and float(x) >= min_acc_avg))
        & (df["bert"] >= min_bertscore)
    )

    if drop_explanations:
        df_mask &= ~df["target_text"].str.contains("betekent")

    df_filtered = df[df_mask]

    if verbosity_threshold > 1:
        df_filtered = df_filtered[
            verbosity_threshold * df_filtered["text"].str.split().str.len()
            >= df_filtered["target_text"].str.split().str.len()
        ]

    df_filtered_dedup = df_filtered.drop_duplicates(subset="text")
    print(f"Dropped {len(df_filtered) - len(df_filtered_dedup)} duplicate examples.")
    print(
        f"Filtered dataset split contains {len(df_filtered_dedup)} simplification "
        f"pairs."
    )
    return df_filtered_dedup


if __name__ == "__main__":
    main()
