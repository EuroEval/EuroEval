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

"""Create the Norwegian knowledge dataset Idioms-no."""

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
logger = logging.getLogger("create_idioms_no")


load_dotenv()


class CandidateAnswers(BaseModel):
    """Candidate answers from the OpenAI API."""

    first: str
    second: str
    third: str


LABELS = ["a", "b", "c", "d"]


def main() -> None:
    """Create the Idioms-no knowledge dataset."""
    # Define the base download URL
    repo_id = "Sprakbanken/Norwegian_idioms"

    # Download the dataset (only the test split is available)
    dataset = load_dataset(path=repo_id, split="test")
    assert isinstance(dataset, Dataset)

    dataset = drop_duplicate_idioms(dataset=dataset)
    assert isinstance(dataset, Dataset)

    # Build the knowledge dataset using a language model
    df = build_dataset_with_llm(dataset=dataset)

    # Create splits
    val_size = 256
    test_size = 2048

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
    dataset_id = "EuroEval/idioms-no"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def drop_duplicate_idioms(dataset: Dataset) -> Dataset:
    """Drop duplicate idioms from the dataset.

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

    # Drop duplicates based on idiom_start
    df = df.drop_duplicates(subset="idiom_start")

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

    cache_file = "norwegian_idioms_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    texts: list[str] = []
    correct_labels: list[str] = []
    languages: list[str] = []
    df_len = len(df)
    for i, row in tqdm(df.iterrows(), total=df_len, desc="Computing LLM responses"):
        id_ = str(i)

        if id_ not in cache:
            logger.info(f"Processing id: {id_}/{df_len}")
            messages: list[ChatCompletionUserMessageParam] = list()

            # Get the correct answer (first accepted completion)
            correct_answer = (
                row.accepted_completions[0] if len(row.accepted_completions) > 0 else ""
            )

            # Determine language for the prompt
            language_name = (
                "Norwegian Bokmål" if row.language == "nob" else "Norwegian Nynorsk"
            )

            user_message = ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"For the Norwegian idiom start: '{row.idiom_start}' where the "
                    f"correct completion is: '{correct_answer}', please provide 3 "
                    f"plausible alternative completions in {language_name}. You should "
                    "return the alternatives in a JSON dictionary, with keys 'first', "
                    "'second', and 'third'. The values should be the alternatives only,"
                    " without any numbering or formatting. The alternatives should be "
                    "unique and not contain the correct answer. Make sure the "
                    "alternatives are grammatically correct and contextually "
                    "appropriate for completing the idiom."
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

        # Make text value: idiom_start + options
        options = cache[id_]

        random.shuffle(LABELS)
        correct_answer = (
            row.accepted_completions[0] if len(row.accepted_completions) > 0 else ""
        )

        options = {
            LABELS[0]: re.sub(r"^[0-9]\. *", "", options["first"]),
            LABELS[1]: re.sub(r"^[0-9]\. *", "", options["second"]),
            LABELS[2]: re.sub(r"^[0-9]\. *", "", options["third"]),
            LABELS[3]: correct_answer,
        }
        if len(set(options.values())) != 4:
            logger.warning(
                f"The options are not unique for the idiom {row.idiom_start}, got "
                f"{options}. Skipping."
            )
            continue
        correct_label = [k for k, v in options.items() if v == correct_answer][0]

        # Create the question text
        language_display = "Bokmål" if row.language == "nob" else "Nynorsk"
        text = (
            f"Complete the {language_display} idiom:\n{row.idiom_start} _____\n\n"
            f"Svaralternativer:\na. {options['a']}\nb. {options['b']}\n"
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
        languages.append(row.language)
    df_llm = pd.DataFrame(
        {"text": texts, "label": correct_labels, "language": languages}
    )
    return df_llm


if __name__ == "__main__":
    main()
