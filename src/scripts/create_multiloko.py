# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "openai==1.66.5",
#     "pandas==2.2.0",
#     "pydantic==2.6.0",
#     "python-dotenv==1.0.1",
#     "tqdm==4.67.1",
# ]
# ///

"""Create the MultiLoKo-mini datasets and upload them to the HF Hub."""

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
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("create_multiloko")

load_dotenv()

# Mapping from ISO 2-letter language codes to config names used by facebook/multiloko
LANGUAGE_CONFIG_NAMES = {
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
    "sv": "Swedish",
}

REPO_ID = "facebook/multiloko"
CACHE_FILE = "multiloko_cache.json"
LANGUAGES = list(LANGUAGE_CONFIG_NAMES.keys())
LABELS = ["a", "b", "c", "d"]
TRAIN_SIZE = 16


class CandidateAnswers(BaseModel):
    """Candidate wrong answers from the OpenAI API."""

    first: str
    second: str
    third: str


def build_dataset_with_llm(
    df: pd.DataFrame,
    language: str,
    language_display_name: str,
    client: OpenAI,
    cache: dict,
    cache_file: str,
) -> pd.DataFrame:
    """Build the MC dataset using an LLM to generate wrong answers.

    Args:
        df:
            The source dataframe with 'question' and 'targets' columns.
        language:
            ISO 2-letter language code.
        language_display_name:
            Human-readable language name for the LLM prompt.
        client:
            OpenAI client instance.
        cache:
            Cache dict mapping cache keys to generated wrong answers.
        cache_file:
            Path to the on-disk cache file.

    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    choices_label = CHOICES_MAPPING[language]
    texts: list[str] = []
    labels: list[str] = []

    for i, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Processing {language_display_name}"
    ):
        question = str(row["question"]).strip()

        # 'targets' is a list of acceptable answers; use the first one as correct
        targets = row["targets"]
        assert isinstance(targets, list) and len(targets) > 0, (
            f"Expected a non-empty list for 'targets', got {targets!r}"
        )
        correct_answer = str(targets[0]).strip()

        if not correct_answer:
            logger.warning(
                "Empty correct answer for %s item %d, skipping.", language, i
            )
            continue

        cache_key = f"{language}_{i}"
        if cache_key not in cache:
            user_message = ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"For this {language_display_name} knowledge question: "
                    f"'{question}'\n"
                    f"The correct answer is: '{correct_answer}'\n"
                    f"Please provide 3 plausible but incorrect alternative answers "
                    f"in {language_display_name}. Return them in a JSON dictionary "
                    "with keys 'first', 'second', and 'third'. The values should be "
                    "brief answers only, without any numbering or formatting. Make "
                    "the alternatives believable but factually wrong."
                ),
            )
            completion = client.beta.chat.completions.parse(
                model="gpt-4.1",
                messages=[user_message],
                response_format=CandidateAnswers,
            )
            event = completion.choices[0].message.parsed
            if event is None:
                logger.warning(
                    "No response from OpenAI for %s item %d, skipping.", language, i
                )
                continue
            cache[cache_key] = dict(event)
            with open(cache_file, "w") as f:
                json.dump(cache, f)

        wrong_answers = cache[cache_key]

        # Shuffle the answer positions
        labels_copy = LABELS.copy()
        random.shuffle(labels_copy)
        options = {
            labels_copy[0]: re.sub(r"^[0-9]\. *", "", wrong_answers["first"]).strip(),
            labels_copy[1]: re.sub(r"^[0-9]\. *", "", wrong_answers["second"]).strip(),
            labels_copy[2]: re.sub(r"^[0-9]\. *", "", wrong_answers["third"]).strip(),
            labels_copy[3]: correct_answer,
        }

        if len(set(options.values())) != 4:
            logger.warning(
                "Options are not unique for %s item %d: %s. Skipping.",
                language,
                i,
                options,
            )
            continue

        correct_label = [k for k, v in options.items() if v == correct_answer][0]
        text = (
            question
            + f"\n{choices_label}:\n"
            + f"a. {options['a']}\n"
            + f"b. {options['b']}\n"
            + f"c. {options['c']}\n"
            + f"d. {options['d']}"
        )

        # Sanity-check that choices are at the end of the text
        sections = text.split("\n")
        choice_idxs = [
            idx
            for idx, section in enumerate(sections)
            if re.match(pattern=r"^[a-e]\. ", string=section) is not None
        ]
        if not all(
            choice_idx == len(sections) - j
            for j, choice_idx in enumerate(sorted(choice_idxs, reverse=True), start=1)
        ):
            logger.warning(
                "Choices are not at the end of the text for %s item %d. Skipping.",
                language,
                i,
            )
            continue

        texts.append(text)
        labels.append(correct_label)

    return pd.DataFrame({"text": texts, "label": labels})


def main() -> None:
    """Create the MultiLoKo-mini datasets and upload them to the HF Hub."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    else:
        cache = {}

    for language in LANGUAGES:
        language_display_name = LANGUAGE_CONFIG_NAMES[language]

        # Load only the 'dev' split (250 samples per language)
        dev_dataset = load_dataset(
            path=REPO_ID,
            name=language_display_name.lower(),
            split="dev",
            token=True,
            trust_remote_code=True,  # required for facebook/multiloko loading script
        )
        assert isinstance(dev_dataset, Dataset)

        df = dev_dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)

        # Build MC dataset: use 'question' as input, 'targets[0]' as correct answer,
        # and GPT-4.1 to generate 3 plausible wrong answers per sample.
        result_df = build_dataset_with_llm(
            df=df,
            language=language,
            language_display_name=language_display_name,
            client=client,
            cache=cache,
            cache_file=CACHE_FILE,
        )

        if len(result_df) < TRAIN_SIZE + 1:
            logger.warning(
                "Skipping language %s: only %d samples after processing, "
                "need at least %d.",
                language,
                len(result_df),
                TRAIN_SIZE + 1,
            )
            continue

        # 16 samples for train, the rest for test (no validation split)
        train_df = result_df.sample(TRAIN_SIZE, random_state=4242)
        test_df = result_df.drop(train_df.index.tolist())

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        dataset_out = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
                "test": Dataset.from_pandas(test_df, split=Split.TEST),
            }
        )

        dataset_id = f"EuroEval/multiloko-{language}-mini"
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        dataset_out.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
