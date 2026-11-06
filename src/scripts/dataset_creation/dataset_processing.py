"""Shared deterministic transformations for dataset-creation scripts."""

import random

import pandas as pd
from datasets import Dataset, DatasetDict, Split

SEED = 4242


def cap_dataframe(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    """Keep the first rows up to a maximum, without crossing source splits.

    Args:
        frame:
            One already-separated source split.
        maximum:
            Maximum rows to retain.

    Returns:
        The capped frame with a fresh consecutive index.
    """
    return frame.iloc[:maximum].reset_index(drop=True)


def format_multiple_choice_record(
    prompt: str,
    correct_answer: str,
    distractors: list[str],
    letters: list[str],
    choices_label: str,
    rng: random.Random,
) -> dict[str, str]:
    """Shuffle choices deterministically and return a EuroEval MCQ record.

    Args:
        prompt:
            Question text preceding the choice label.
        correct_answer:
            Correct option.
        distractors:
            Incorrect options.
        letters:
            Choice labels in display order.
        choices_label:
            Localised text introducing the choices.
        rng:
            Random generator shared across records for reproducibility.

    Returns:
        Record with formatted text and the correct answer letter.
    """
    options = [correct_answer, *distractors]
    rng.shuffle(options)
    text = f"{prompt}\n{choices_label}:\n" + "\n".join(
        f"{letter}. {option}" for letter, option in zip(letters, options)
    )
    return {"text": text, "label": letters[options.index(correct_answer)]}


def process_multilingual_tweet_sentiment(
    source: DatasetDict,
    source_label_mapping: dict[int, str],
    minimum_text_length: int,
    maximum_text_length: int,
) -> DatasetDict:
    """Pool multilingual tweet splits, deduplicate, sample and filter them.

    Source splits are intentionally pooled for these two similarly structured
    Twitter datasets. Length filtering remains after sampling to preserve their
    historical selection behaviour.

    Args:
        source:
            Dataset with train, validation and test splits.
        source_label_mapping:
            Mapping from source label IDs to EuroEval labels.
        minimum_text_length:
            Inclusive minimum tweet length.
        maximum_text_length:
            Inclusive maximum tweet length.

    Returns:
        DatasetDict with independently selected train, val and test rows.
    """
    frames = [source[split].to_pandas() for split in ("train", "validation", "test")]
    frame = pd.concat(frames, ignore_index=True)
    frame["label"] = frame["label"].map(source_label_mapping)
    frame = frame.drop_duplicates().reset_index(drop=True)
    val_frame = frame.sample(n=256, random_state=SEED)
    remainder = frame.loc[~frame.index.isin(val_frame.index)]
    test_frame = remainder.sample(n=1024, random_state=SEED)
    remainder = remainder.loc[~remainder.index.isin(test_frame.index)]
    train_frame = remainder.sample(n=1024, random_state=SEED)
    sampled = {"train": train_frame, "val": val_frame, "test": test_frame}
    bounds = {"text": (minimum_text_length, maximum_text_length)}
    filtered = {
        split: filter_by_character_lengths(frame=part, bounds=bounds)
        for split, part in sampled.items()
    }
    return dataset_dict_from_frames(frames=filtered, columns=["text", "label"])


def dataset_dict_from_frames(
    frames: dict[str, pd.DataFrame], columns: list[str]
) -> DatasetDict:
    """Convert named data frames into a DatasetDict using selected columns.

    Args:
        frames:
            Frames keyed by their output split names.
        columns:
            Columns retained in each output split, in order.

    Returns:
        DatasetDict containing the selected data.
    """
    split_names = {"train": Split.TRAIN, "val": Split.VALIDATION, "test": Split.TEST}
    return DatasetDict(
        {
            name: Dataset.from_pandas(frame[columns], split=split_names[name])
            for name, frame in frames.items()
        }
    )


def filter_by_character_lengths(
    frame: pd.DataFrame, bounds: dict[str, tuple[int, int]]
) -> pd.DataFrame:
    """Retain rows whose named text columns fall within inclusive limits.

    Args:
        frame:
            Data frame to filter.
        bounds:
            Text column names mapped to inclusive (minimum, maximum) lengths.

    Returns:
        A filtered copy of the data frame.
    """
    mask = pd.Series(True, index=frame.index)
    for column, (minimum, maximum) in bounds.items():
        mask &= frame[column].str.len().between(minimum, maximum)
    return frame.loc[mask].copy()


def process_ner_frame(
    frame: pd.DataFrame, label_mapping: dict[int, str], include_tokens: bool = True
) -> pd.DataFrame:
    """Convert token and integer-tag columns into EuroEval NER columns.

    Args:
        frame:
            One source split containing ``tokens`` and ``ner_tags``.
        label_mapping:
            Dataset-specific mapping from tag IDs to label strings.
        include_tokens (optional):
            Whether to retain the original token sequence. Defaults to True.

    Returns:
        A copy containing token sequence (when requested), joined text and labels.
    """
    result = frame.copy()
    result["text"] = result["tokens"].map(lambda tokens: " ".join(tokens))
    result["labels"] = result.pop("ner_tags").map(
        lambda tags: [label_mapping[tag] for tag in tags]
    )
    columns = ["tokens", "text", "labels"] if include_tokens else ["text", "labels"]
    return result[columns]
