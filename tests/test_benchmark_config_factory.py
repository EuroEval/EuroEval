"""Unit tests for the `benchmark_config_factory` module."""

import pytest
import torch
from scandeval.benchmark_config_factory import (
    get_correct_language_codes,
    prepare_dataset_tasks,
    prepare_device,
    prepare_languages,
)
from scandeval.dataset_tasks import LA, NER, get_all_dataset_tasks
from scandeval.enums import Device
from scandeval.languages import DA, EN, NB, NN, NO, get_all_languages


@pytest.mark.parametrize(
    argnames=["input_language", "expected_language"],
    argvalues=[
        ("da", ["da"]),
        (["da"], ["da"]),
        (["da", "en"], ["da", "en"]),
        ("no", ["no", "nb", "nn"]),
        (["nb"], ["nb", "no"]),
        ("all", list(get_all_languages().keys())),
    ],
    ids=[
        "single language",
        "single language as list",
        "multiple languages",
        "no -> no + nb + nn",
        "nb -> nb + no",
        "all -> all languages",
    ],
)
def test_get_correct_language_codes(input_language, expected_language):
    languages = get_correct_language_codes(language=input_language)
    assert set(languages) == set(expected_language)


@pytest.mark.parametrize(
    argnames=["input_language", "input_model_language", "expected_model_language"],
    argvalues=[
        ("da", None, [DA]),
        (["da"], None, [DA]),
        (["da", "no"], ["da"], [DA]),
        (["da", "en"], None, [DA, EN]),
        ("no", None, [NO, NB, NN]),
        (["nb"], None, [NB, NO]),
        ("all", None, list(get_all_languages().values())),
    ],
    ids=[
        "single language",
        "single language as list",
        "language takes precedence over model language",
        "multiple languages",
        "no -> no + nb + nn",
        "nb -> nb + no",
        "all -> all languages",
    ],
)
def test_prepare_languages(
    input_language, input_model_language, expected_model_language
):
    prepared_languages = get_correct_language_codes(input_language)
    model_languages = prepare_languages(
        model_language=input_model_language, languages=prepared_languages
    )
    model_languages = sorted(model_languages, key=lambda x: x.code)
    expected_model_language = sorted(expected_model_language, key=lambda x: x.code)
    assert model_languages == expected_model_language


@pytest.mark.parametrize(
    argnames=["input_task", "expected_task"],
    argvalues=[
        (None, list(get_all_dataset_tasks().values())),
        ("linguistic-acceptability", [LA]),
        (["linguistic-acceptability"], [LA]),
        (["linguistic-acceptability", "named-entity-recognition"], [LA, NER]),
    ],
    ids=[
        "all tasks",
        "single task",
        "single task as list",
        "multiple tasks",
    ],
)
def test_prepare_dataset_tasks(input_task, expected_task):
    prepared_tasks = prepare_dataset_tasks(dataset_task=input_task)
    assert prepared_tasks == expected_task


@pytest.mark.parametrize(
    argnames=["device" "expected_device"],
    argvalues=[
        (Device.CPU, torch.device("cpu")),
        (
            None,
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu"),
        ),
    ],
    ids=[
        "device provided",
        "device not provided",
    ],
)
def test_prepare_device(device, expected_device):
    prepared_device = prepare_device(device=device)
    assert prepared_device == expected_device
