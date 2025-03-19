# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "click==8.1.3",
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the FQuAD-mini NER dataset and upload it to the HF Hub."""

from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_CONTEXT,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_CONTEXT,
    MIN_NUM_CHARS_IN_QUESTION,
)
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests import HTTPError


@click.command(
    help="Create the FQuAD-mini NER dataset and upload it to the HF Hub. Note that you "
    "need to request the raw data from https://fquad.illuin.tech/#download before you "
    "can run this script."
)
@click.argument("data_dir", type=str)
def main(data_dir: str) -> None:
    """Create the FQuAD-mini NER dataset and uploads it to the HF Hub."""
    train_path = Path(data_dir) / "train.json"
    train_df = pd.read_json(train_path)

    valtest_path = Path(data_dir) / "valid.json"
    valtest_df = pd.read_json(valtest_path)

    def set_up_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Set up the dataframe."""
        data_dict: dict[str, list] = defaultdict(list)
        for article_dict in df.data:
            for paragraph_dict in article_dict["paragraphs"]:
                context = paragraph_dict["context"]
                for qa_dict in paragraph_dict["qas"]:
                    id_ = qa_dict["id"]
                    question = qa_dict["question"]
                    answer_dict = qa_dict["answers"]
                    data_dict["id"].append(id_)
                    data_dict["context"].append(context)
                    data_dict["question"].append(question)
                    data_dict["answers"].append(
                        dict(
                            text=[answer["text"] for answer in answer_dict],
                            answer_start=[
                                answer["answer_start"] for answer in answer_dict
                            ],
                        )
                    )
        return pd.DataFrame(data_dict)

    train_df = set_up_dataframe(df=train_df)
    valtest_df = set_up_dataframe(df=valtest_df)

    # Remove duplicates
    train_df = train_df.drop_duplicates(subset="id")
    valtest_df = valtest_df.drop_duplicates(subset="id")

    # Only work with samples where the context is not very large or small
    train_lengths = train_df.context.str.len()
    valtest_lengths = valtest_df.context.str.len()
    train_df = train_df[
        train_lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)
    ]
    valtest_df = valtest_df[
        valtest_lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)
    ]

    # Only work with samples where the question is not very large or small
    train_question_lengths = train_df.question.str.len()
    valtest_question_lengths = valtest_df.question.str.len()
    train_df = train_df[
        train_question_lengths.between(
            MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION
        )
    ]
    valtest_df = valtest_df[
        valtest_question_lengths.between(
            MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION
        )
    ]

    # Extract information on which examples contain an answer
    def has_answer(example: dict) -> bool:
        return len(example["text"]) > 0 and example["text"][0] != ""

    train_has_answer: pd.Series = train_df.answers.map(has_answer)
    valtest_has_answer: pd.Series = valtest_df.answers.map(has_answer)

    # Only work with the questions having answers in the context
    train_df_with_answer: pd.DataFrame = train_df.loc[train_has_answer]
    valtest_df_with_answer: pd.DataFrame = valtest_df.loc[valtest_has_answer]

    # Create validation split
    val_size = 256
    val_df = valtest_df_with_answer.sample(n=val_size, random_state=4242)

    # Create test split
    test_size = 2048
    valtest_df_with_answer_filtered: pd.DataFrame = valtest_df_with_answer.loc[
        ~valtest_df_with_answer.index.isin(val_df.index)
    ]
    test_df = valtest_df_with_answer_filtered.sample(n=test_size, random_state=4242)

    # Create train split
    train_size = 1024
    train_df = train_df_with_answer.sample(n=train_size, random_state=4242)

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    def add_answer_start(example: dict) -> dict:
        possible_answers: list[str] = example["answers"]["text"]
        if possible_answers:
            answers: list[str] = list()
            answer_starts: list[int] = list()
            for answer in possible_answers:
                answer_start = example["context"].find(answer)
                if answer_start >= 0:
                    answers.append(answer)
                    answer_starts.append(answer_start)
            example["answers"]["text"] = answers
            example["answers"]["answer_start"] = answer_starts
        else:
            example["answers"]["answer_start"] = []
        return example

    dataset = dataset.map(function=add_answer_start)

    # Create dataset ID
    mini_dataset_id = "EuroEval/fquad-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(mini_dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(mini_dataset_id, private=True)


if __name__ == "__main__":
    main()
