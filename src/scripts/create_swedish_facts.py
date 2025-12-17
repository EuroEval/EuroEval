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


"""Create a Swedish knowledge dataset based on liu-nlp/swedish-facts-v1."""

import json
import logging
import os
import random
import re

import pandas as pd
from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, Split, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel
from tqdm.auto import tqdm

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_swedish_knowledge")


load_dotenv()


class CandidateAnswers(BaseModel):
    """Candidate answers from the OpenAI API."""

    first: str
    second: str
    third: str


LABELS = ["a", "b", "c", "d"]


def main() -> None:
    """Create the Swedish knowledge dataset."""
    # Define the base download URL
    repo_id = "liu-nlp/swedish-facts-v1"

    # Download the dataset
    dataset = load_dataset(path=repo_id, split="train")
    assert isinstance(dataset, Dataset)

    # Normalise columns and drop duplicate questions
    dataset = drop_duplicate_questions(dataset=dataset)
    assert isinstance(dataset, Dataset)

    # Build the knowledge dataset using a language model
    df = build_dataset_with_llm(dataset=dataset)

    # Create splits
    val_size = 64
    train_size = 128

    assert len(df) >= val_size + train_size + 1, (
        "The dataset is too small after filtering; "
        f"need at least {val_size + train_size + 1} samples, got {len(df)}."
    )

    val_df = df.sample(val_size, random_state=42)
    remaining_df = df.drop(val_df.index.tolist())

    train_df = remaining_df.sample(train_size, random_state=42)
    test_df = remaining_df.drop(train_df.index.tolist())

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset_dict = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/swedish-facts"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset_dict.push_to_hub(dataset_id, private=True)


def drop_duplicate_questions(dataset: Dataset) -> Dataset:
    """Normalise text fields and drop duplicate questions from the dataset.

    Args:
        dataset:
            The dataset to normalise and drop duplicates from.

    Returns:
        A dataset with canonical 'question' and 'answer' columns and no duplicates.
    """
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Strip all leading and trailing whitespace
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    question_col, answer_col = "Question (Swedish)", "Answer (Swedish)"

    # Remove trailing periods from answers
    df[answer_col] = df[answer_col].str.rstrip(".")

    # Rename to canonical column names
    df = df.rename(columns={question_col: "question", answer_col: "answer"})

    # Drop duplicates
    df = df.drop_duplicates(subset="question")

    # Keep only the relevant columns
    df = df[["question", "answer"]]

    return Dataset.from_pandas(df)


def build_dataset_with_llm(dataset: Dataset) -> pd.DataFrame:
    """Build the Swedish knowledge dataset using a language model.

    Args:
        dataset:
            The dataset to build the knowledge dataset from.

    Returns:
        The multiple-choice knowledge dataset.
    """
    df = dataset.to_pandas()

    assert isinstance(df, pd.DataFrame)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    cache_file = "swedish_facts_cache.json"
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
                    f"För frågan: {row.question} där det korrekta svaret är: "
                    f"{row.answer}, ge tre trovärdiga men felaktiga "
                    "svarsalternativ på svenska. Du ska returnera alternativen i "
                    "en JSON-dict med nycklarna 'first', 'second' och 'third'. "
                    "Värdena ska vara enbart svarsalternativen, utan numrering "
                    "eller annan formatering. Alternativen ska vara unika och får "
                    "inte innehålla det korrekta svaret."
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
                json.dump(cache, f, ensure_ascii=False)

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
                "The options are not unique for the document "
                f"{row.question!r}, got {options}. Skipping."
            )
            continue
        correct_label = [k for k, v in options.items() if v == row.answer][0]

        text = (
            f"{row.question}\n"
            f"{CHOICES_MAPPING['sv']}:\n"
            f"a. {options['a']}\n"
            f"b. {options['b']}\n"
            f"c. {options['c']}\n"
            f"d. {options['d']}"
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
                f"{text!r}. Skipping."
            )
            continue

        texts.append(text)
        correct_labels.append(correct_label)

    df_llm = pd.DataFrame({"text": texts, "label": correct_labels})
    return df_llm


if __name__ == "__main__":
    main()
