"""Offline entrypoint and provenance tests for dataset processing workflows."""

import importlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from src.scripts.dataset_creation.dataset_processing import (
    prepare_mlsum_dataset,
    prepare_ner_dataset,
    process_source_splits,
    sample_dataframe,
)
from src.scripts.dataset_creation.faroese_processing import (
    FaroeseTask,
    prepare_faroese_dataset,
)
from src.scripts.dataset_creation.qa_processing import (
    Article,
    prepare_squad_dataset,
    squad_articles_to_dataset,
)


def test_sampling_preserves_identity_until_partition_exclusion() -> None:
    """Sampling must retain original indices for subsequent set subtraction."""
    frame = pd.DataFrame({"id": range(100)}, index=range(200, 300))
    selected = sample_dataframe(frame=frame, maximum=20)
    remainder = frame.loc[~frame.index.isin(selected.index)]
    assert len(remainder) == 80
    assert set(selected["id"]).isdisjoint(remainder["id"])


@pytest.mark.parametrize(
    "name", ["create_squad", "create_squad_it", "create_squad_nl_old"]
)
def test_squad_heldout_partition_is_disjoint(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct heldout questions must never appear in both output splits.

    Args:
        name:
            Script using a shared validation/test source.
        monkeypatch:
            Fixture intercepting all remote operations.
    """
    monkeypatch.syspath_prepend(str(Path("src/scripts/dataset_creation").resolve()))
    module = importlib.import_module(name)
    source = DatasetDict(
        {
            split: Dataset.from_list(
                [
                    {
                        "id": f"{split}-{index}",
                        "context": f"answer {split} unique passage {index:04d} " * 2,
                        "question": f"What is the answer for {split}-{index:04d}?",
                        "answers": {"text": ["answer"], "answer_start": [0]},
                    }
                    for index in range(size)
                ]
            )
            for split, size in {"train": 20, "validation": 700, "test": 700}.items()
        }
    )
    captured: list[DatasetDict] = []
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.repo_info",
        lambda *args, **kwargs: SimpleNamespace(private=True),
    )
    monkeypatch.setattr(
        DatasetDict,
        "push_to_hub",
        lambda dataset, *args, **kwargs: captured.append(dataset),
    )

    module.main()

    result = captured[0]
    val = set(result["val"]["id"])
    test = set(result["test"]["id"])
    assert len(val) == 256
    assert len(test) == 444
    assert val.isdisjoint(test)
    assert len(val | test) == 700
    assert all(value.startswith("train-") for value in result["train"]["id"])


SCRIPT_TARGETS = {
    "create_conll_en": "conll-en-mini",
    "create_conll_es": "conll-es-mini",
    "create_conll_nl": "conll-nl-mini",
    "create_germeval": "germeval-mini",
    "create_mlsum_de": "mlsum-de-mini",
    "create_mlsum_es": "mlsum-es-mini",
    "create_squad": "squad-mini",
    "create_squad_it": "squad-it-mini",
    "create_squad_nl": "squad-nl-v2-mini",
    "create_squad_nl_old": "squad-nl-mini",
    "create_nqii": "nqii-mini",
    "create_norquad": "norquad-mini",
    "create_fquad": "fquad-mini",
    "create_faroese_semantic_relations": "faroese-semantic-relations",
    "create_faroese_metaphorical_explanations": "faroese-metaphorical-explanations",
    "create_sb10k": "sb10k-mini",
    "create_sentipolc16": "sentipolc16-mini",
}


@pytest.mark.parametrize("name", SCRIPT_TARGETS)
@pytest.mark.parametrize("mode", ["package", "script"])
def test_dataset_entrypoints_offline(
    name: str, mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every affected entrypoint imports and processes non-empty fake inputs.

    Args:
        name:
            Dataset creation script.
        mode:
            Package import or direct script execution.
        monkeypatch:
            Fixture intercepting source loading and private Hub operations.
        tmp_path:
            Local directory for FQuAD JSON source fixtures.
    """
    directory = Path("src/scripts/dataset_creation").resolve()
    monkeypatch.syspath_prepend(str(directory))
    uploads: list[tuple[str, DatasetDict]] = []
    loads: list[dict[str, object]] = []

    def load_dataset(**kwargs: object) -> DatasetDict | Dataset:
        """Return synthetic source records instead of accessing the Hub.

        Args:
            **kwargs:
                Dataset loader configuration.

        Returns:
            Source records in the requested family format.
        """
        loads.append(kwargs)
        if "faroese" in name:
            return Dataset.from_pandas(df=faroese_frame(name=name))
        splits = {}
        for split in ("train", "validation", "dev", "test"):
            if "conll" in name or "germeval" in name:
                frame = pd.DataFrame(
                    {"tokens": [[f"{split}-one", "word"]], "ner_tags": [[0, 1]]}
                )
            elif "mlsum" in name:
                frame = pd.DataFrame(
                    {
                        "text": [f"{split} article with enough characters to retain"],
                        "target" if name.endswith("de") else "summary": ["reference"],
                        "references": [["unused"]],
                    }
                )
            elif name in ("create_sb10k", "create_sentipolc16"):
                frame = pd.DataFrame(
                    {
                        "text": [f"{split}-tweet", f"{split}-tweet", "x"],
                        "label": [0, 0, 1],
                    }
                )
            else:
                frame = qa_frame(prefix=split, size=300)
            splits[split] = Dataset.from_pandas(df=frame)
        return DatasetDict(splits)

    def push_to_hub(dataset: DatasetDict, repo_id: str, private: bool) -> None:
        """Capture the real processed output in place of uploading bytes.

        Args:
            dataset:
                Prepared output dataset.
            repo_id:
                Destination repository ID.
            private:
                Requested privacy flag.
        """
        assert private
        uploads.append((repo_id, dataset))

    def request_source(url: str) -> SimpleNamespace:
        """Return a synthetic NorQuAD response preserving URL split identity.

        Args:
            url:
                Original source URL.

        Returns:
            Response-shaped object providing local JSON data.
        """
        split = next(
            value for value in ("training", "validation", "test") if value in url
        )
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": qa_articles(prefix=split, size=3)},
        )

    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    for helper in ("faroese_processing", "tweet_sentiment"):
        for prefix in ("", "src.scripts.dataset_creation."):
            cached = sys.modules.get(f"{prefix}{helper}")
            if cached is not None:
                monkeypatch.setattr(cached, "load_dataset", load_dataset)
    monkeypatch.setattr("requests.get", request_source)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.repo_info",
        lambda *args, **kwargs: SimpleNamespace(private=True),
    )
    monkeypatch.setattr(DatasetDict, "push_to_hub", push_to_hub)
    arguments = []
    if name == "create_fquad":
        for filename, prefix in (("train.json", "train"), ("valid.json", "validation")):
            (tmp_path / filename).write_text(
                json.dumps({"data": qa_articles(prefix=prefix, size=300)})
            )
        arguments = [str(tmp_path)]
    elif "faroese" in name:
        arguments = ["fake.jsonl"]
    if mode == "script":
        monkeypatch.setattr(sys, "argv", [name, *arguments])
        if name == "create_fquad":
            with pytest.raises(SystemExit) as exit_info:
                runpy.run_path(
                    path_name=str(directory / f"{name}.py"), run_name="__main__"
                )
            assert exit_info.value.code == 0
        else:
            runpy.run_path(path_name=str(directory / f"{name}.py"), run_name="__main__")
    else:
        module = importlib.reload(
            importlib.import_module(f"src.scripts.dataset_creation.{name}")
        )
        if name == "create_fquad":
            module.main.callback(data_dir=str(tmp_path))
        elif "faroese" in name:
            module.main(source="fake.jsonl")
        else:
            module.main()

    assert len(uploads) == 1
    repository_id, result = uploads[0]
    assert repository_id == f"EuroEval/{SCRIPT_TARGETS[name]}"
    assert set(result) == {"train", "val", "test"}
    assert len(result["train"]) > 0
    assert len(result["val"]) > 0
    assert len(result["test"]) > 0
    for split in result.values():
        assert "__index_level_0__" not in split.column_names
    if "faroese" in name:
        assert sum(len(split) for split in result.values()) == 20
    else:
        column = "text" if "text" in result["train"].column_names else "context"
        assert all(
            value.startswith(("train", "training")) for value in result["train"][column]
        )
        assert not set(result["val"][column]) & set(result["test"][column])
    if "conll" in name or "germeval" in name:
        assert result["train"][0]["labels"] == [
            "O",
            "B-LOC" if "germeval" in name else "B-PER",
        ]
    if "mlsum" in name:
        assert "references" not in result["train"].column_names
        assert result["train"][0]["target_text"] == "reference"
    if name in ("create_sb10k", "create_sentipolc16"):
        assert loads[0]["name"] == ("german" if name == "create_sb10k" else "italian")
        assert len(result["train"]) == 1
        assert result["train"][0]["label"] == "negative"


def faroese_frame(name: str) -> pd.DataFrame:
    """Return task-specific, non-empty generated MCQ records.

    Args:
        name:
            Dataset script identifying the Faroese task.

    Returns:
        Twenty distinct valid raw Faroese records.
    """
    if "semantic" in name:
        return pd.DataFrame(
            {
                "orð": [f"word-{index}" for index in range(20)],
                "andheiti": "answer",
                "random_outliers": [["v", "w", "x", "y", "z"]] * 20,
            }
        )
    return pd.DataFrame(
        {
            "example": [f"example-{index}" for index in range(20)],
            "correct explanation": "answer",
            "distractor explanations": [
                [{"distractor explanation": value} for value in ("x", "y", "z")]
            ]
            * 20,
        }
    )


def qa_articles(prefix: str, size: int) -> list[Article]:
    """Create raw SQuAD articles with multiple paragraphs and questions.

    Args:
        prefix:
            Original source split identity.
        size:
            Number of passages.

    Returns:
        One raw article containing all requested passages.
    """
    return [
        {
            "paragraphs": [
                {
                    "context": f"{prefix} answer passage with enough characters "
                    f"{index}",
                    "qas": [
                        {
                            "id": f"{prefix}-{index}",
                            "question": f"What is the answer to question {index}?",
                            "answers": [
                                {"text": "answer", "answer_start": len(prefix) + 1}
                            ],
                        }
                    ],
                }
                for index in range(size)
            ]
        }
    ]


def qa_frame(prefix: str, size: int) -> pd.DataFrame:
    """Create valid and distinct question/context pairs with source provenance.

    Args:
        prefix:
            Original split identity.
        size:
            Number of questions.

    Returns:
        Frame containing valid SQuAD records.
    """
    return pd.DataFrame(
        {
            "id": [f"{prefix}-{index}" for index in range(size)],
            "context": [
                f"{prefix} answer passage with enough characters {index}"
                for index in range(size)
            ],
            "question": [
                f"What is the answer for {prefix}-{index}?" for index in range(size)
            ],
            "answers": [{"text": ["answer"], "answer_start": [len(prefix) + 1]}] * size,
        }
    )


@pytest.mark.parametrize("family", ["ner", "mlsum"])
def test_family_preparation_caps_original_splits(family: str) -> None:
    """Deep NER and MLSum processing enforce caps without changing provenance.

    Args:
        family:
            Shared processing family under test.
    """
    source = DatasetDict(
        {
            split: Dataset.from_pandas(
                pd.DataFrame(
                    {
                        "tokens": [[f"{split}-{index}"] for index in range(size)],
                        "ner_tags": [[1]] * size,
                        "text": [
                            f"{split} article long enough for filtering {index}"
                            for index in range(size)
                        ],
                        "summary": ["reference"] * size,
                    }
                )
            )
            for split, size in {"train": 1100, "validation": 300, "test": 2100}.items()
        }
    )
    limits = {"train": 1024, "val": 256, "test": 2048}
    if family == "ner":
        result = prepare_ner_dataset(
            source=source, label_mapping={1: "B-PER"}, limits=limits
        )
    else:
        result = prepare_mlsum_dataset(source=source, target_column="summary")
    for split, original in {
        "train": "train",
        "val": "validation",
        "test": "test",
    }.items():
        assert len(result[split]) == limits[split]
        assert all(text.startswith(original) for text in result[split]["text"])


def test_faroese_metaphors_skip_malformed_nested_distractors() -> None:
    """Nested explanation records reject missing fields without crashing."""
    frame = faroese_frame(name="metaphorical").iloc[:3].copy()
    frame.at[0, "distractor explanations"] = None
    frame.at[1, "distractor explanations"] = [{"wrong key": "x"}] * 3
    dataset, truncated = prepare_faroese_dataset(
        frame=frame, task="metaphorical-explanations"
    )
    assert not truncated
    assert sum(len(split) for split in dataset.values()) == 1


def test_faroese_preparation_cleans_skips_and_deduplicates() -> None:
    """Malformed and duplicate records do not survive choice randomisation."""
    frame = pd.DataFrame(
        {
            "orð": [" word\n", "word", None, "other", "last"],
            "andheiti": ["answer"] * 5,
            "random_outliers": [["v", "w", "x", "y", "z"]] * 3 + [None, ["x"]],
        }
    )
    result, truncated = prepare_faroese_dataset(frame=frame, task="semantic-relations")
    assert not truncated
    assert sum(len(split) for split in result.values()) == 1
    record = result["test"][0]
    assert record["text"].startswith("Hvat er andheitið hjá orðinum 'word'?")
    assert f"{record['label']}. answer" in record["text"]


@pytest.mark.parametrize("task", ["semantic-relations", "metaphorical-explanations"])
def test_faroese_workflow_names_only_truncated_data_mini(
    task: FaroeseTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generator naming changes only when the cleaned data exceeds split caps.

    Args:
        task:
            Faroese input schema under test.
        monkeypatch:
            Fixture intercepting loading and the final upload.
    """
    module = importlib.import_module("src.scripts.dataset_creation.faroese_processing")
    script = "semantic" if task == "semantic-relations" else "metaphorical"
    frame = faroese_frame(name=script)
    repeated = pd.concat([frame] * 200, ignore_index=True)
    subject = "orð" if task == "semantic-relations" else "example"
    repeated[subject] = [f"unique-{index}" for index in range(len(repeated))]
    captured: list[tuple[DatasetDict, str]] = []
    monkeypatch.setattr(
        module, "load_dataset", lambda **kwargs: Dataset.from_pandas(repeated)
    )
    monkeypatch.setattr(
        module,
        "upload_private",
        lambda dataset, repository_id: captured.append((dataset, repository_id)),
    )
    module.create_faroese_dataset(source="fake.jsonl", task=task)
    dataset, repository_id = captured[0]
    assert repository_id == f"EuroEval/faroese-{task}-mini"
    assert [len(dataset[name]) for name in ("train", "val", "test")] == [
        1024,
        256,
        2048,
    ]
    texts = [text for split in dataset.values() for text in split["text"]]
    assert len(texts) == len(set(texts))


def test_italian_squad_keeps_all_training_answers() -> None:
    """Only heldout Italian records are restricted to the first answer."""
    frame = qa_frame(prefix="source", size=1)
    frame.at[0, "answers"] = {"text": ["answer", "passage"], "answer_start": [7, 14]}
    source = DatasetDict(
        {split: Dataset.from_pandas(frame) for split in ("train", "dev", "test")}
    )
    result = prepare_squad_dataset(
        source=source,
        source_splits={"train": "train", "val": "dev", "test": "test"},
        answer_mode="first",
    )
    assert result["train"][0]["answers"]["text"] == ["answer", "passage"]
    assert result["test"][0]["answers"]["text"] == ["answer"]


def test_processing_retains_small_split_indices_only_until_build() -> None:
    """Source processing preserves all small splits but output drops source indices."""
    frames = {
        split: pd.DataFrame({"id": [f"{split}-a", f"{split}-b"]}, index=[8, 13])
        for split in ("train", "val", "test")
    }
    result = process_source_splits(
        frames=frames, limits={"train": 1024, "val": 256, "test": 2048}
    )
    for split, frame in result.items():
        assert frame["id"].tolist() == [f"{split}-a", f"{split}-b"]
        assert frame.index.tolist() == [0, 1]


def test_raw_squad_conversion_keeps_all_questions() -> None:
    """Raw article conversion retains every question, not only the first one."""
    articles = qa_articles(prefix="source", size=2)
    question = articles[0]["paragraphs"][0]["qas"][0].copy()
    question["id"] = "extra"
    articles[0]["paragraphs"][0]["qas"].append(question)
    result = squad_articles_to_dataset(articles=articles)
    assert len(result) == 3
    assert set(result["id"]) == {"source-0", "source-1", "extra"}


def test_squad_inferred_answers_remove_unmatched_candidates() -> None:
    """Offset inference removes unmatched candidates before choosing partitions."""
    frame = qa_frame(prefix="source", size=3)
    frame.at[0, "answers"] = {"text": ["missing", "answer"], "answer_start": []}
    frame.at[1, "answers"] = {"text": ["missing"], "answer_start": []}
    source = DatasetDict(
        {name: Dataset.from_pandas(frame) for name in ("train", "dev", "test")}
    )
    result = prepare_squad_dataset(
        source=source,
        source_splits={"train": "train", "val": "dev", "test": "test"},
        answer_mode="infer",
    )
    assert len(result["train"]) == 2
    assert result["train"][0]["answers"] == {"text": ["answer"], "answer_start": [7]}


@pytest.mark.parametrize("size", [0, 2])
def test_squad_preserves_small_original_splits(size: int) -> None:
    """Small or empty original partitions are never filled from another source.

    Args:
        size:
            Number of source records per partition.
    """
    source = DatasetDict(
        {
            split: Dataset.from_pandas(qa_frame(prefix=split, size=size))
            for split in ("train", "dev", "test")
        }
    )
    result = prepare_squad_dataset(
        source=source, source_splits={"train": "train", "val": "dev", "test": "test"}
    )
    assert {split: len(data) for split, data in result.items()} == {
        "train": size,
        "val": size,
        "test": size,
    }


def test_squad_validation_filters_before_partitioning() -> None:
    """Invalid lengths and answer spans never enter output partitions."""
    frame = qa_frame(prefix="heldout", size=700)
    frame.loc[range(0, 100), "context"] = "short"
    frame.loc[range(100, 200), "question"] = "short"
    for index in range(200, 300):
        frame.at[index, "answers"] = {"text": ["missing"], "answer_start": [0]}
    source = DatasetDict(
        {
            "train": Dataset.from_pandas(qa_frame(prefix="training", size=3)),
            "validation": Dataset.from_pandas(frame),
        }
    )
    result = prepare_squad_dataset(
        source=source,
        source_splits={"train": "train", "val": "validation", "test": "validation"},
    )
    val, test = set(result["val"]["id"]), set(result["test"]["id"])
    assert len(val) == 256
    assert len(test) == 144
    assert val.isdisjoint(test)
    assert val | test == {f"heldout-{index}" for index in range(300, 700)}
