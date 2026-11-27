"""Submitted-record identity, score and failure accounting outcomes."""

import pytest
from fastapi import HTTPException

from euroeval_api.eee import validate_record


@pytest.mark.parametrize(
    "changes",
    [
        {"failed": 0},
        {"failed": "[1]"},
        {"language": "en"},
        {"score": True},
        {"lower": 0.91},
    ],
)
def test_invalid_records_are_rejected(changes: dict[str, object]) -> None:
    """Reject inconsistent accounting, foreign language and invalid score evidence.

    Args:
        changes:
            One materially invalid record field.
    """
    payload = record(
        failed=changes.get("failed", 1.0),
        language=str(changes.get("language", "da")),
        score=changes.get("score", 0.9),
        lower=float(str(changes.get("lower", 0.8))),
    )
    with pytest.raises(HTTPException) as failure:
        validate_record(
            record=payload,
            model_id="org/model",
            revision="revision",
            language="da",
            euroeval_version="18.4.1",
        )
    assert failure.value.status_code == 422


def record(
    failed: object = 1.0, language: str = "da", score: object = 0.9, lower: float = 0.8
) -> dict[str, object]:
    """Build a scored record with repeated metrics for one failed instance.

    Args:
        failed (optional):
            Declared failure count, defaulting to one.
        language (optional):
            Evaluated language, defaulting to Danish.
        score (optional):
            Metric score, defaulting to 0.9.
        lower (optional):
            Confidence interval lower bound, defaulting to 0.8.

    Returns:
        Complete submitted EEE record.
    """
    return {
        "schema_version": "0.3.0",
        "model_info": {"id": "org/model", "revision": "revision"},
        "eval_library": {
            "name": "euroeval",
            "version": "18.4.1",
            "additional_details": {
                "dataset": "nørec",
                "task": "sentiment-classification",
                "languages": [language],
                "validation_split": False,
                "few_shot": None,
                "raw_results": [
                    {
                        "accuracy": {"failed_instances": [{"index": 7}]},
                        "f1": {"failed_instances": [{"index": 7}]},
                    }
                ],
            },
        },
        "evaluation_results": [
            {
                "evaluation_name": "f1",
                "score_details": {
                    "score": score,
                    "details": {"num_failed_instances": failed},
                    "uncertainty": {
                        "confidence_interval": {"lower": lower, "upper": 0.95}
                    },
                },
                "source_data": {"dataset_name": "nørec"},
                "metric_config": {
                    "lower_is_better": False,
                    "min_score": 0,
                    "max_score": 1,
                },
            }
        ],
    }


def test_record_identity_and_failure_accounting() -> None:
    """Count an instance once per iteration and preserve compact UTF-8 identity."""
    result = validate_record(
        record=record(),
        model_id="org/model",
        revision="revision",
        language="da",
        euroeval_version="18.4.1",
    )
    assert result["identity"] == '["org/model","nørec",false,null]'
    assert result["failed"] == 1
    assert result["warnings"]
