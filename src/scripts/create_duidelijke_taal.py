# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.4.2",
#     "huggingface-hub==0.36.0",
#     "pandas==2.2.3",
#     "requests==2.32.3",
#     "evaluate==0.4.6",
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
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from evaluate import load
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Creates Duidelijke Taal simplification dataset and uploads it to the HF Hub."""
    dataset_id = "GPT-NL/DuidelijkeTaal-v1.0-split"
    dataset_original = load_dataset(dataset_id, token=True)

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

    dataset_filtered = DatasetDict()
    for split in ["train", "test"]:
        df = dataset_original[split].to_pandas().rename(columns=column_mapping)
        df_filtered = filter_dataset(df)
        dataset_filtered[split] = Dataset.from_pandas(df_filtered).cast_column(
            "complexity_text", ClassLabel(num_classes=4)
        )

    # create 50%/50% train/val split
    # original split is 50%/50% train/test, so total split becomes 25%/25%/50%
    train_val = dataset_filtered["train"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="complexity_text"
    )

    drop_columns = [
        col
        for col in dataset_filtered["train"].column_names
        if col not in ["text", "target_text"]
    ]

    dataset = DatasetDict(
        {
            "train": train_val["train"].remove_columns(drop_columns),
            "val": train_val["test"].remove_columns(drop_columns),
            "test": dataset_filtered["test"].remove_columns(drop_columns),
        }
    )
    processed_dataset_id = "EuroEval/duidelijke-taal"

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
    """Filter the Duidelijke Taal dataset.

    The dataset is filtered based on the complexity difference between the original
    sentence and its synthesized simplification, the number of human annotations,
    and the average accuracy of the simplification.

    Args:
        df (pandas.DataFrame): The unfiltered Duidelijke Taal dataset.
        min_complexity_difference (int, optional): The minimum difference in
            complexity between the original sentence and the simplification to
            create a subset with clearly differentiated examples. Defaults to 10.
        min_n_participants (int, optional): The minimum number of participants from
            the crowd-sourcing experiment who rated the sentence pair, ensuring a
            somewhat general opinion. Defaults to 2.
        min_acc_avg (int, optional): The minimum average accuracy percentage of the
            simplification used to filter out inaccurate simplifications.
            Defaults to 60.
        min_bertscore (float, optional): The minimum BERTScore (F1) similarity
            between the sentence pairs to filter out pairs that vary too much in
            meaning. Defaults to 0.76.
        drop_explanations (bool, optional): Whether to drop sentence pairs where the
            synthetic example contains explanations not present in the original
            sentence (e.g., "Dat betekent dat..."). Defaults to True.
        verbosity_threshold (int, optional): Threshold used to drop pairs where the
            synthetic example contains overly verbose or long explanations that
            deviate from the original text. Simplifications are dropped if they
            contain more than `threshold * number_of_words` of the input text.
            Setting this parameter to 1 disables this filtering. Defaults to 2.

    Returns:
        pandas.DataFrame: The filtered dataset.
    """
    # load default model (bert-base-multilingual-cased), which is sufficient as filter
    bert_score = load("bertscore")
    df["bert"] = bert_score.compute(
        references=df["text"].to_list(),
        predictions=df["target_text"].to_list(),
        lang="nl",
    )["f1"]

    df["complexity_diff"] = df[df["complexity_a_avg"].astype(str).str.len() > 0][
        "complexity_a_avg"
    ].astype(float) - df[df["complexity_b_avg"].astype(str).str.len() > 0][
        "complexity_b_avg"
    ].astype(float)

    # create binned complexity column of input text for stratified dataset split
    df["complexity_text"] = pd.qcut(
        df["complexity_a_avg"],
        q=4,  # 4 bins
        labels=False,
        duplicates="drop",
    )

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

    # 'betekent' = 'means' in Dutch
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
