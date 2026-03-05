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

"""Create the MultiNRC datasets and upload them to the HF Hub."""

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
logger = logging.getLogger("create_multinrc")

load_dotenv()


class CandidateAnswers(BaseModel):
    """Candidate answers from the OpenAI API."""

    first: str
    second: str
    third: str


LABELS = ["a", "b", "c", "d"]

LANGUAGE_SUBSET_MAPPING = {"es": "spanish", "fr": "french"}

LANGUAGE_NAME_MAPPING = {"es": "Spanish", "fr": "French"}


def main() -> None:
    """Create the MultiNRC datasets and upload them to the HF Hub."""
    repo_id = "ScaleAI/MultiNRC"

    for language in LANGUAGE_SUBSET_MAPPING.keys():
        subset = LANGUAGE_SUBSET_MAPPING[language]

        # Load the dataset (only a test split is available)
        dataset = load_dataset(path=repo_id, name=subset, split="test", token=True)
        assert isinstance(dataset, Dataset)

        # Build the multiple choice dataset using a language model
        df = build_dataset_with_llm(dataset=dataset, language=language)

        # Create splits: 64 train, 128 val, rest test
        train_df = df.sample(64, random_state=4242)
        df = df.drop(train_df.index.tolist())

        val_df = df.sample(128, random_state=4242)
        test_df = df.drop(val_df.index.tolist())

        # Reset the index
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
                "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
                "test": Dataset.from_pandas(test_df, split=Split.TEST),
            }
        )

        # Create dataset ID
        dataset_id = f"EuroEval/multinrc-{language}"

        # Remove the dataset from Hugging Face Hub if it already exists
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


def build_dataset_with_llm(dataset: Dataset, language: str) -> pd.DataFrame:
    """Build the multiple choice dataset using a language model.

    Args:
        dataset:
            The dataset containing the questions and correct answers.
        language:
            The ISO language code (e.g. "es", "fr").

    Returns:
        The multiple choice dataset as a DataFrame with 'text' and 'label' columns.
    """
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    cache_file = f"multinrc_{language}_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
    else:
        cache = {}

    language_name = LANGUAGE_NAME_MAPPING[language]
    texts: list[str] = []
    correct_labels: list[str] = []
    df_len = len(df)

    for i, row in tqdm(
        df.iterrows(), total=df_len, desc=f"Computing LLM responses ({language})"
    ):
        id_ = str(i)

        if id_ not in cache:
            logger.info(f"Processing id: {id_}/{df_len}")
            messages: list[ChatCompletionUserMessageParam] = []

            user_message = ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"For the following {language_name} question: "
                    f"'{row.i18n_prompt}' where the correct answer is: "
                    f"'{row.i18n_gtfa}', please provide 3 plausible but incorrect "
                    f"alternative answers in {language_name}. Return the alternatives "
                    "in a JSON dictionary with keys 'first', 'second', and 'third'. "
                    "The values should be the alternatives only, without any numbering "
                    "or formatting. The alternatives should be unique and must not "
                    "contain or repeat the correct answer."
                ),
            )
            messages.append(user_message)

            completion = client.beta.chat.completions.parse(
                model="gpt-4.1", messages=messages, response_format=CandidateAnswers
            )

            event = completion.choices[0].message.parsed
            assert event is not None, f"Expected a response, but got {event}."
            cache[id_] = dict(event)
            with open(cache_file, "w") as f:
                json.dump(cache, f)

        options_raw = cache[id_]
        labels = LABELS.copy()
        random.shuffle(labels)

        correct_answer = str(row.i18n_gtfa)
        options = {
            labels[0]: re.sub(r"^[0-9]\. *", "", options_raw["first"]),
            labels[1]: re.sub(r"^[0-9]\. *", "", options_raw["second"]),
            labels[2]: re.sub(r"^[0-9]\. *", "", options_raw["third"]),
            labels[3]: correct_answer,
        }
        options = dict(sorted(options.items()))

        if len(set(options.values())) != 4:
            logger.warning(
                f"The options are not unique for question '{row.i18n_prompt}', got "
                f"{options}. Skipping."
            )
            continue

        correct_label = [k for k, v in options.items() if v == correct_answer][0]

        text = (
            str(row.i18n_prompt).replace("\n", " ").strip()
            + f"\n{CHOICES_MAPPING[language]}:\n"
            + f"a. {options['a']}\n"
            + f"b. {options['b']}\n"
            + f"c. {options['c']}\n"
            + f"d. {options['d']}"
        )

        # Sanity check that the choices are at the end of the text
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
                f"Choices are not at the end of the document for '{text}'. Skipping."
            )
            continue

        texts.append(text)
        correct_labels.append(correct_label)

    return pd.DataFrame({"text": texts, "label": correct_labels})


if __name__ == "__main__":
    main()
