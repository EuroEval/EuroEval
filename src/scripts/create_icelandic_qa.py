# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "openai==1.66.5",
#     "pandas==2.2.0",
#     "python-dotenv==1.0.1",
#     "requests==2.32.3",
# ]
# ///

"""Create the Icelandic QA dataset and upload it to the HF Hub."""

import os
import re

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_CONTEXT,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_CONTEXT,
    MIN_NUM_CHARS_IN_QUESTION,
)
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from dotenv import load_dotenv
from huggingface_hub.hf_api import HfApi
from openai import OpenAI
from requests.exceptions import HTTPError

load_dotenv()


def main() -> None:
    """Create the Icelandic QA dataset and upload it to the HF Hub."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    repo_id = "mideind/icelandic_qa_scandeval"

    dataset = load_dataset(path=repo_id, token=True, split="train")

    df = dataset.to_pandas().rename(columns={"example_id": "id"})

    # Change all the answers on format "Árið xxxx." to "xxxx", e.g. only keep the year.
    df["answer"] = df["answer"].apply(lambda x: re.sub(r"^Árið (\d{4}).?$", r"\1", x))

    # Only work with samples where the context is not very large or small
    lengths = df.context.str.len()
    df = df[lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)]

    # Only work with samples where the question is not very large or small
    lengths = df.question.str.len()
    df_with_no_outliers = df[
        lengths.between(MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION)
    ]

    def rephrase_answer(question: str, answer: str, context: str) -> str:
        """Rephrase the answer such that it is in the context.

        Args:
            question:
                The question.
            answer:
                The answer.
            context:
                The context.

        Returns:
            The rephrased answer (if the answer is already in the context, it is
            returned as is).
        """
        answer = answer[:-1] if answer.endswith(".") else answer

        if answer.lower() in context.lower():
            return answer

        # Use OpenAI to rephrase the answer
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Given the following context: {context!r}, please rephrase "
                        f"the answer {answer!r} so that it matches exactly with a "
                        "phrase from the context in a case-insensitive way. I.e., it "
                        "must be possible to find the rephrased answer in the context "
                        "without any modifications. The rephrased answer should be "
                        "as concise as possible, preferable not more than 7 words, "
                        "and ideally 3 or less words. Here is the related question: "
                        f"{question!r}"
                    ),
                }
            ],
            model="gpt-4o",
            temperature=0.0,
            seed=4242,
        )

        rephrased_answer = chat_completion.choices[0].message.content
        if rephrased_answer is None:
            raise ValueError(
                f"OpenAI could not rephrase the answer {answer!r} for the question "
                f"{question!r} and context {context!r}."
            )
        return rephrased_answer

    # Rephrase the answers
    df_with_no_outliers["answer"] = df_with_no_outliers.apply(
        lambda row: rephrase_answer(row["question"], row["answer"], row["context"]),
        axis=1,
    )

    # Only work with the questions having answers in the context
    has_answer: pd.Series = df_with_no_outliers.apply(
        lambda row: row["answer"].lower() in row["context"].lower(), axis=1
    )

    df_with_answer: pd.DataFrame = df_with_no_outliers.loc[has_answer]

    # Make the `answers` column a dictionary with the answer and the answer start
    df_with_answer["answers"] = df_with_answer.apply(
        lambda row: {
            "text": [row["answer"]],
            "answer_start": row["context"].lower().index(row["answer"].lower()),
        },
        axis=1,
    )

    # Remove the old `answer` column
    df_with_answer = df_with_answer.drop(columns=["answer"])

    # Create validation split
    val_size = 128
    val_df = df_with_answer.sample(n=val_size, random_state=4242)

    # Create test split
    test_size = 1024
    df_with_answer_filtered: pd.DataFrame = df_with_answer.loc[
        ~df_with_answer.index.isin(val_df.index)
    ]
    test_df = df_with_answer_filtered.sample(n=test_size, random_state=4242)

    # Create train split, which is the rest of the data
    train_size = len(df_with_answer) - val_size - test_size
    full_train_df_with_answer = df_with_answer_filtered.loc[
        ~df_with_answer_filtered.index.isin(test_df.index)
    ]
    train_df = full_train_df_with_answer.sample(n=train_size, random_state=4242)

    train_num_samples_lower_bound = 500
    assert len(train_df) > train_num_samples_lower_bound, (
        f"There are only {len(train_df):,} samples in the training set - there should "
        f"be at least {train_num_samples_lower_bound:,}."
    )

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/icelandic-qa"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
