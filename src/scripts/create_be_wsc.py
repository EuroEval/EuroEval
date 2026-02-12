# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "python-dotenv==1.0.1",
#     "huggingface-hub==0.24.0",
#     "openai==1.61.1",
#     "pandas==2.2.0",
#     "pydantic==2.10.6",
#     "requests==2.32.3",
# ]
# ///

"""Create the BE-WSC dataset (Winograd Schema Challenge, Belarusian).

This script converts the BelarusianGLUE `bewsc_as_wsc` split into a Winogrande-like
multiple choice dataset.

The source dataset provides one candidate antecedent (`span1_text`) and a label
indicating whether the pronoun (`span2_text`) refers to it. We use an LLM to
extract a second candidate antecedent from the sentence to form a 2-way choice.
"""

import json
import os
import textwrap
from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict, Split, disable_progress_bars, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel

from .constants import (
    CHOICES_MAPPING,
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)

load_dotenv()

CACHE_FILE: str = "be_wsc_cache.json"

candidate_cache: dict[str, str] = {}
client: OpenAI | None = None


class SecondCandidate(BaseModel):
    """Structured output: extracted second candidate span from the sentence."""

    distractor: str


def load_cache() -> dict[str, str]:
    """Load cache from CACHE_FILE if it exists.

    Returns:
        The cache.
    """
    try:
        with open(CACHE_FILE, "r") as cache_file:
            return json.load(cache_file)
    except FileNotFoundError:
        return {}


def save_cache(cache: dict) -> None:
    """Save cache to CACHE_FILE.

    Args:
        cache: The cache to save.
    """
    with open(CACHE_FILE, "w") as cache_file:
        json.dump(cache, cache_file, indent=4)


def main() -> None:
    """Create the BE-WSC dataset and upload it to the HF Hub."""
    disable_progress_bars()

    global candidate_cache, client
    candidate_cache = load_cache()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    repo_id = "maaxap/BelarusianGLUE"

    # Download the dataset
    dataset = load_dataset(path=repo_id, name="bewsc_as_wsc")
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Check for duplicate idx values (used for cache keys)
    assert train_df["idx"].is_unique, "The train dataframe has duplicate idx values"
    assert val_df["idx"].is_unique, "The validation dataframe has duplicate idx values"
    assert test_df["idx"].is_unique, "The test dataframe has duplicate idx values"

    # Add split names (used for cache keys)
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    train_df = prepare_dataframe(df=train_df)
    val_df = prepare_dataframe(df=val_df)
    test_df = prepare_dataframe(df=test_df)

    train_df, val_df, test_df = make_splits(
        train_df=train_df, val_df=val_df, test_df=test_df
    )

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Push the dataset to the Hugging Face Hub
    target_dataset_id = "EuroEval/be-wsc"

    HfApi().delete_repo(target_dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(target_dataset_id, private=True)


def make_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make splits for the dataset.

    This function creates a train/val split and moves the remaining samples into test.

    Args:
        train_df: The training dataframe.
        val_df: The validation dataframe.
        test_df: The test dataframe.

    Returns:
        The final training, validation, and test dataframes.
    """
    val_size = 64
    train_size = 128

    val_size = min(val_size, len(val_df))
    train_size = min(train_size, len(train_df))

    val_df_sample = val_df.sample(n=val_size, random_state=42)
    val_remaining_df = val_df.drop(val_df_sample.index)

    train_df_sample = train_df.sample(n=train_size, random_state=42)
    train_remaining_df = train_df.drop(train_df_sample.index)

    test_df = pd.concat(
        [train_remaining_df, val_remaining_df, test_df], ignore_index=True
    )

    # Reset indices
    train_df = train_df_sample.reset_index(drop=True)
    val_df = val_df_sample.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dataframe by filtering and processing.

    Args:
        df:
            Input dataframe.

    Returns:
        Processed dataframe.

    Raises:
        ValueError: If required columns are missing
    """
    df = df.copy()

    required_cols = {
        "idx",
        "split",
        "text",
        "span1_index",
        "span1_text",
        "span2_index",
        "span2_text",
        "label",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Extract a second candidate referent/antecedent span for each sample
    # (cached, via LLM).
    # - If label==1, this should be a plausible distractor (wrong option).
    # - If label==0, span1 is wrong, so this should be the true referent (right option).
    df["second_candidate"] = df.apply(_extract_second_candidate, axis=1)

    # Drop rows where we failed to extract a usable second candidate
    df = df.loc[df["second_candidate"].astype(str).str.strip().astype(bool)]

    # Build Winogrande-like instruction: replace the pronoun (span2) with a blank "_"
    df["instruction"] = df.apply(_make_instruction, axis=1)

    # Options: A is the provided candidate antecedent (span1), B is the extracted one
    df["option_a"] = df["span1_text"].astype(str)
    df["option_b"] = df["second_candidate"].astype(str)

    # label==1 means span2 refers to span1 -> option_a is correct, else option_b
    df["label"] = df["label"].map(lambda x: "a" if int(x) == 1 else "b")

    # Remove the samples with overly short or long texts/options
    df = df.loc[
        (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & (df.option_a.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_a.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() <= MAX_NUM_CHARS_IN_OPTION)
    ]

    # Remove overly repetitive samples
    df = df.loc[
        ~df.instruction.apply(is_repetitive)
        & ~df.option_a.apply(is_repetitive)
        & ~df.option_b.apply(is_repetitive)
    ]

    # Make a `text` column with all the options in it
    df["text"] = [
        row.instruction.replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING['be']}:\n"
        "a. " + row.option_a.replace("\n", " ").strip() + "\n"
        "b. " + row.option_b.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Only keep the `text` and `label` columns
    df = df.loc[:, ["text", "label"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def is_repetitive(text: str) -> bool:
    """Return True if the text is repetitive.

    Args:
        text: The text to check for repetition.

    Returns:
        True if the text is repetitive, False otherwise.
    """
    max_repetitions = max(Counter(text.split()).values())
    return max_repetitions > MAX_REPETITIONS


def _cache_key(row: pd.Series) -> str:
    """Create a stable cache key for the second-candidate extraction call.

    Args:
        row: The row of the dataframe.

    Returns:
        The cache key.
    """
    split = str(row["split"]).strip()
    idx = int(row["idx"])
    return f"{split}_{idx}"


def _extract_second_candidate(row: pd.Series) -> str:
    """Extract a second candidate referent span from the sentence (cached).

    Note: this is a "distractor" only when label==1. When label==0, span1 is wrong,
    so the extracted span should be the true referent.

    Args:
        row: The row of the dataframe.

    Returns:
        The second candidate span.

    Raises:
        ValueError: If the OpenAI API call fails.
    """
    key = _cache_key(row)
    if key in candidate_cache:
        cached = candidate_cache[key]
        return str(cached)

    text = str(row["text"])
    pronoun = str(row["span2_text"])
    candidate = str(row["span1_text"])
    label = int(row["label"])

    prompt = textwrap.dedent(
        f"""
        You are helping build a Winogrande-style multiple choice dataset in Belarusian.

        Return your response as a JSON object with exactly this schema:
        {{"distractor": string}}.
        Do not include any other keys, text, markdown, or explanation.

        You are given:
        - A Belarusian sentence.
        - A pronoun (or pronominal adverb) that appears in the sentence.
        - A candidate antecedent span from the same sentence.
        - A label: 1 means the pronoun DOES refer to the candidate; 0 means it DOES
          NOT.

        Your task depends on the label:
        - If label = 1: the candidate is the TRUE referent. Extract EXACTLY ONE
          OTHER span from the sentence that could plausibly be a referent (a
          distractor / wrong option).
        - If label = 0: the candidate is a FALSE option. Extract the TRUE referent
          span from the sentence (the thing the pronoun actually refers to).

        Rules (must follow):
        - Return EXACTLY a substring that appears in the sentence (copy-paste).
        - Return a content span (noun phrase / named entity / location phrase), not
          a pronoun.
        - It must NOT be identical to the candidate span.
        - It must NOT be the pronoun itself.
        - Prefer an earlier mention in the sentence when possible.
        - Keep the span short and option-like when possible. In this dataset,
          candidate spans are usually 1 word (~81%) or 2 words (~15%), sometimes 3
          (~3%), and 4+ words < 1%.
          Use a longer span only if needed for a natural referent.
        - If there is no reasonable alternative candidate, return an empty string.

        Few-shot examples (Belarusian):
        Example (label=1, need a distractor / wrong option)
        Sentence: Аліса шукала ў натоўпе сваю сяброўку Надзю. З-за таго, што яна заўсёды
        носіць чырвоны каптур, Аліса хутка заўважыла яе.
        Pronoun: яна
        Candidate: Надзю
        label: 1
        Answer: {{"distractor": "Аліса"}}

        Example (label=1, need a distractor / wrong option)
        Sentence: Дзяніс растлумачыў сваю тэорыю Марку, але ён не пераканаў яго.
        Pronoun: ён
        Candidate: Дзяніс
        label: 1
        Answer: {{"distractor": "Марку"}}

        Example (label=0, need the TRUE referent)
        Sentence: Мужчына ўзяў заплаканага хлопчыка за руку. Яго далонь была
        вялікай і цёплай.
        Pronoun: Яго
        Candidate: заплаканага хлопчыка
        label: 0
        Answer: {{"distractor": "Мужчына"}}

        Example (label=0, need the TRUE referent)
        Sentence: Доктарка паведаміла Кацярыне, што яна сыходзіць на пенсію, і
        прапанавала
        некалькі варыянтаў працягу лячэння.
        Pronoun: яна
        Candidate: Кацярыне
        label: 0
        Answer: {{"distractor": "Доктарка"}}

        Example (label=1, pronominal adverb)
        Sentence: Мы вырашылі правесці дзень на возеры, бо на ўзбярэжжы акіяна
        бачылі акулу,
        так што плаваць там было небяспечна.
        Pronoun: там
        Candidate: на ўзбярэжжы акіяна
        label: 1
        Answer: {{"distractor": "на возеры"}}

        Now do the same for this input. Remember: output ONLY valid JSON like
        {{"distractor": "..."}}

        Sentence: <text>{text}</text>
        Pronoun: <pronoun>{pronoun}</pronoun>
        Candidate: <candidate>{candidate}</candidate>
        label: <label>{label}</label>
        """
    ).strip()

    messages: list[ChatCompletionUserMessageParam] = [
        ChatCompletionUserMessageParam(role="user", content=prompt)
    ]

    completion = client.beta.chat.completions.parse(
        model="gpt-4o", messages=messages, response_format=SecondCandidate
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Parsed response is None")

    second_candidate = parsed.distractor.strip()
    if second_candidate and second_candidate not in text:
        # Enforce "exact substring" to avoid downstream mismatches.
        second_candidate = ""

    if second_candidate and second_candidate == candidate:
        second_candidate = ""
    if second_candidate and second_candidate == pronoun:
        second_candidate = ""

    candidate_cache[key] = second_candidate
    save_cache(cache=candidate_cache)
    return second_candidate


def _mask_pronoun(text: str, pronoun: str, word_index: int) -> str:
    """Replace the pronoun token (by word index) with '_'.

    Args:
        text: The text to mask the pronoun in.
        pronoun: The pronoun to mask.
        word_index: The index of the pronoun in the text.

    Returns:
        The text with the pronoun masked.
    """
    words = text.split()
    assert 0 <= word_index < len(words), "Word index is out of range"

    token = words[word_index]
    if token == pronoun:
        words[word_index] = "_"
    elif pronoun and pronoun in token:
        # Keep punctuation attached to the token (e.g., "яна," -> "_,")
        words[word_index] = token.replace(pronoun, "_", 1)
    else:
        assert False, f"Pronoun not found in token: {token} in text: {text}"

    return " ".join(words)


def _make_instruction(row: pd.Series) -> str:
    """Create the instruction string in Winogrande-like format.

    Args:
        row: The row of the dataframe.

    Returns:
        The instruction string.
    """
    text = str(row["text"]).replace("\n", " ").strip()
    pronoun = str(row["span2_text"]).strip()
    pronoun_idx = int(row["span2_index"])
    masked = _mask_pronoun(text=text, pronoun=pronoun, word_index=pronoun_idx)
    return f"{masked} Да каго або чаго адносіцца пропуск _?"


if __name__ == "__main__":
    main()
