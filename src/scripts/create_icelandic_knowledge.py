# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "openai==1.66.5",
#     "pandas==2.2.0",
#     "pydantic==2.6.0",
#     "python-dotenv==1.0.1",
#     "requests==2.32.3",
#     "tqdm==4.67.1",
# ]
# ///

"""Create mideind/icelandic_qa_scandeval as a knowledge task dataset."""

import json
import logging
import os
import random
import re

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel
from requests import HTTPError
from tqdm.auto import tqdm

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_icelandic_knowledge")


load_dotenv()


class CandidateAnswers(BaseModel):
    """Candidate answers from the OpenAI API."""

    first: str
    second: str
    third: str


LABELS = ["a", "b", "c", "d"]


def main() -> None:
    """Create the icelandic knowledge dataset."""
    # Define the base download URL
    repo_id = "mideind/icelandic_qa_scandeval"

    # Download the dataset
    dataset = load_dataset(path=repo_id, split="train")
    assert isinstance(dataset, Dataset)

    dataset = drop_duplicate_questions(dataset=dataset)
    assert isinstance(dataset, Dataset)

    # Build the knowledge dataset using a language model
    df = build_dataset_with_llm(dataset=dataset)

    # Create splits
    val_size = 128
    test_size = 1024

    val_df = df.sample(val_size, random_state=42)
    df = df.drop(val_df.index.tolist())

    test_df = df.sample(test_size, random_state=42)
    df = df.drop(test_df.index.tolist())

    train_df = df
    assert len(train_df) > 800, "The training set should have at least 800 samples."

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
    dataset_id = "EuroEval/icelandic-knowledge"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def drop_duplicate_questions(dataset: Dataset) -> Dataset:
    """Drop duplicate questions from the dataset.

    Args:
        dataset:
            The dataset to drop duplicates from.

    Returns:
        The dataset without duplicates.
    """
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Strip all leading and trailing whitespace
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Remove trailing periods
    df["answer"] = df["answer"].str.rstrip(".")

    # Drop duplicates
    df = df.drop_duplicates(subset="question")

    return Dataset.from_pandas(df)


def build_dataset_with_llm(dataset: Dataset) -> pd.DataFrame:
    """Build the knowledge dataset using a language model.

    Args:
        dataset:
            The dataset to build the knowledge dataset from.

    Returns:
        The knowledge dataset.
    """
    df = dataset.to_pandas()

    assert isinstance(df, pd.DataFrame)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    cache_file = "icelandic_qa_scandeval_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    texts: list[str] = []
    correct_labels: list[str] = []
    df_len = len(df)
    for i, row in tqdm(df.iterrows(), total=df_len, desc="Computing LLM responses"):
        id_ = str(i)

        if id_ not in cache:
            logger.info(f"Processing id: {id_}/{df_len}")
            messages: list[ChatCompletionUserMessageParam] = list()
            user_message = ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"For the question: {row.question} where the correct answer is: "
                    f"{row.answer}, please provide 3 plausible alternatives in "
                    "Icelandic. You should return the alternatives in a JSON "
                    "dictionary, with keys 'first', 'second', and 'third'. The values "
                    "should be the alternatives only, without any numbering or "
                    "formatting. The alternatives should be unique and not contain the "
                    "correct answer."
                ),
            )
            messages.append(user_message)

            completion = client.beta.chat.completions.parse(
                model="gpt-4o", messages=messages, response_format=CandidateAnswers
            )

            # Store response
            event = completion.choices[0].message.parsed
            assert event is not None, f"Expected a response, but got {event}."
            cache[id_] = dict(event)
            with open(cache_file, "w") as f:
                json.dump(cache, f)

        # Make text value: question + options
        options = cache[id_]

        random.shuffle(LABELS)
        options = {
            LABELS[0]: re.sub(r"^[0-9]\. *", "", options["first"]),
            LABELS[1]: re.sub(r"^[0-9]\. *", "", options["second"]),
            LABELS[2]: re.sub(r"^[0-9]\. *", "", options["third"]),
            LABELS[3]: row.answer,
        }
        if len(set(options.values())) != 4:
            logger.warning(
                f"The options are not unique for the document {row.question}, got "
                f"{options}. Skipping."
            )
            continue
        correct_label = [k for k, v in options.items() if v == row.answer][0]

        text = (
            f"{row.question}\nSvarmöguleikar:\na. {options['a']}\nb. {options['b']}\n"
            f"c. {options['c']}\nd. {options['d']}"
        )

        # Sanity check that the texts are formatted correctly
        sections = text.split("\n")
        choice_idxs = [
            idx
            for idx, section in enumerate(sections)
            if re.match(pattern=r"^[a-e]\. ", string=section) is not None
        ]
        if not all(
            choice_idx == len(sections) - i
            for i, choice_idx in enumerate(sorted(choice_idxs, reverse=True), start=1)
        ):
            logger.warning(
                "Choices are not at the end of the document for the document "
                f"{text}. Skipping."
            )
            continue

        texts.append(text)
        correct_labels.append(correct_label)

    df_llm = pd.DataFrame({"text": texts, "label": correct_labels})
    return df_llm


if __name__ == "__main__":
    main()
