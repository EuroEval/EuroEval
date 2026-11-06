"""Shared deterministic transformations for dataset-creation scripts."""

import collections.abc as c
import random

import pandas as pd
from datasets import Dataset, DatasetDict, Split

SEED = 4242


def build_split_dataset(
    load_split: c.Callable[[str], pd.DataFrame],
    source_splits: dict[str, str],
    limits: dict[str, int],
    columns: list[str],
) -> DatasetDict:
    """Load, cap and assemble independently processed source splits.

    Args:
        load_split:
            Dataset-specific callback returning a formatted source split.
        source_splits:
            Output split names mapped to original source split names.
        limits:
            Maximum row count keyed by output split.
        columns:
            Columns retained in each output split.

    Returns:
        Dataset dictionary with source splits kept separate.

    Raises:
        ValueError:
            If source split mappings and limits have different output names.
    """
    if source_splits.keys() != limits.keys():
        raise ValueError("Source splits and limits must have the same output names")
    frames = {
        output_name: load_split(source_name)
        for output_name, source_name in source_splits.items()
    }
    capped = {
        split: cap_dataframe(frame=frame, maximum=limits[split])
        for split, frame in frames.items()
    }
    return dataset_dict_from_frames(frames=capped, columns=columns)


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
    """Deduplicate, filter and cap each multilingual tweet split independently.

    Examples never move between source splits. Length filtering and deduplication
    happen before deterministic capping.

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
    source_names = {"train": "train", "val": "validation", "test": "test"}
    bounds = {"text": (minimum_text_length, maximum_text_length)}
    frames = {}
    for output_split, source_split in source_names.items():
        frame = source[source_split].to_pandas()
        assert isinstance(frame, pd.DataFrame)
        frame["label"] = frame["label"].map(source_label_mapping)
        frame = frame.drop_duplicates().reset_index(drop=True)
        frames[output_split] = filter_by_character_lengths(frame=frame, bounds=bounds)
    limits = {"train": 1024, "val": 256, "test": 1024}
    capped = process_source_splits(frames=frames, limits=limits)
    return dataset_dict_from_frames(frames=capped, columns=["text", "label"])


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


def process_source_splits(
    frames: dict[str, pd.DataFrame],
    limits: dict[str, int],
    transform: c.Callable[[str, pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Transform and independently cap source splits without moving examples.

    Each input frame remains confined to the correspondingly named output split.
    A transform may apply dataset-specific filtering or formatting before capping.

    Args:
        frames:
            Source frames keyed by output split name.
        limits:
            Maximum row count for each split.
        transform (optional):
            Dataset-specific transformation receiving split name and frame.

    Returns:
        Transformed, independently capped frames with reset indices.

    Raises:
        ValueError:
            If frame and limit split names differ.
    """
    if frames.keys() != limits.keys():
        raise ValueError("Frames and limits must have the same split names")
    prepared = {}
    for split, frame in frames.items():
        transformed = transform(split, frame) if transform is not None else frame
        prepared[split] = sample_dataframe(frame=transformed, maximum=limits[split])
    return prepared


def sample_dataframe(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    """Select at most ``maximum`` rows from one source split deterministically.

    Args:
        frame:
            One already-separated source split.
        maximum:
            Maximum rows to retain.

    Returns:
        A sampled frame with a fresh consecutive index.
    """
    if len(frame) <= maximum:
        return frame.reset_index(drop=True)
    return frame.sample(n=maximum, random_state=SEED).reset_index(drop=True)


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


def split_generated_dataset(frame: pd.DataFrame) -> tuple[DatasetDict, bool]:
    """Create deterministic Faroese-style splits and enforce EuroEval caps.

    The source data is shuffled into proportional train, validation and test
    splits before each split is independently capped. No rows cross split seams.

    Args:
        frame:
            Fully transformed and deduplicated records.

    Returns:
        Dataset dictionary and whether any split was truncated.
    """
    total = len(frame)
    train_size = int(total * 1024 / 3328)
    val_size = int(total * 256 / 3328)
    shuffled = frame.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_df = shuffled.iloc[:train_size]
    val_df = shuffled.iloc[train_size : train_size + val_size]
    test_df = shuffled.iloc[train_size + val_size :]
    limits = {"train": 1024, "val": 256, "test": 2048}
    source_frames = {"train": train_df, "val": val_df, "test": test_df}
    truncated = any(len(source_frames[name]) > limit for name, limit in limits.items())
    capped = process_source_splits(frames=source_frames, limits=limits)
    return dataset_dict_from_frames(frames=capped, columns=["text", "label"]), truncated
