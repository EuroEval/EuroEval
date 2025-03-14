"""All benchmarks tasks used in EuroEval."""

from .data_models import MetricConfig, Task
from .enums import TaskGroup


def get_all_tasks() -> dict[str, Task]:
    """Get a list of all the dataset tasks.

    Returns:
        A mapping between names of dataset tasks and their configurations.
    """
    return {cfg.name: cfg for cfg in globals().values() if isinstance(cfg, Task)}


LA = Task(
    name="linguistic-acceptability",
    task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
    metrics=[
        MetricConfig(
            name="mcc",
            pretty_name="Matthew's Correlation Coefficient",
            huggingface_id="matthews_correlation",
            results_key="matthews_correlation",
        ),
        MetricConfig(
            name="macro_f1",
            pretty_name="Macro-average F1-score",
            huggingface_id="f1",
            results_key="f1",
            compute_kwargs=dict(average="macro"),
        ),
    ],
)


NER = Task(
    name="named-entity-recognition",
    task_group=TaskGroup.TOKEN_CLASSIFICATION,
    metrics=[
        MetricConfig(
            name="micro_f1_no_misc",
            pretty_name="Micro-average F1-score without MISC tags",
            huggingface_id="seqeval",
            results_key="overall_f1",
        ),
        MetricConfig(
            name="micro_f1",
            pretty_name="Micro-average F1-score with MISC tags",
            huggingface_id="seqeval",
            results_key="overall_f1",
        ),
    ],
)


RC = Task(
    name="reading-comprehension",
    task_group=TaskGroup.QUESTION_ANSWERING,
    metrics=[
        MetricConfig(
            name="f1",
            pretty_name="F1-score",
            huggingface_id="squad_v2",
            results_key="f1",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2f}%"),
        ),
        MetricConfig(
            name="em",
            pretty_name="Exact Match",
            huggingface_id="squad_v2",
            results_key="exact",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2f}%"),
        ),
    ],
)


SENT = Task(
    name="sentiment-classification",
    task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
    metrics=[
        MetricConfig(
            name="mcc",
            pretty_name="Matthew's Correlation Coefficient",
            huggingface_id="matthews_correlation",
            results_key="matthews_correlation",
        ),
        MetricConfig(
            name="macro_f1",
            pretty_name="Macro-average F1-score",
            huggingface_id="f1",
            results_key="f1",
            compute_kwargs=dict(average="macro"),
        ),
    ],
)


SUMM = Task(
    name="summarization",
    task_group=TaskGroup.TEXT_TO_TEXT,
    metrics=[
        MetricConfig(
            name="bertscore",
            pretty_name="BERTScore",
            huggingface_id="bertscore",
            results_key="f1",
            compute_kwargs=dict(
                model_type="microsoft/mdeberta-v3-base", device="auto", batch_size=32
            ),
        ),
        MetricConfig(
            name="rouge_l",
            pretty_name="ROUGE-L",
            huggingface_id="rouge",
            results_key="rougeL",
        ),
    ],
)


KNOW = Task(
    name="knowledge",
    task_group=TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
    metrics=[
        MetricConfig(
            name="mcc",
            pretty_name="Matthew's Correlation Coefficient",
            huggingface_id="matthews_correlation",
            results_key="matthews_correlation",
        ),
        MetricConfig(
            name="accuracy",
            pretty_name="Accuracy",
            huggingface_id="accuracy",
            results_key="accuracy",
        ),
    ],
)


MCRC = Task(
    name="multiple-choice-reading-comprehension",
    task_group=TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
    metrics=[
        MetricConfig(
            name="mcc",
            pretty_name="Matthew's Correlation Coefficient",
            huggingface_id="matthews_correlation",
            results_key="matthews_correlation",
        ),
        MetricConfig(
            name="accuracy",
            pretty_name="Accuracy",
            huggingface_id="accuracy",
            results_key="accuracy",
        ),
    ],
)


COMMON_SENSE = Task(
    name="common-sense-reasoning",
    task_group=TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
    metrics=[
        MetricConfig(
            name="mcc",
            pretty_name="Matthew's Correlation Coefficient",
            huggingface_id="matthews_correlation",
            results_key="matthews_correlation",
        ),
        MetricConfig(
            name="accuracy",
            pretty_name="Accuracy",
            huggingface_id="accuracy",
            results_key="accuracy",
        ),
    ],
)


SPEED = Task(
    name="speed",
    task_group=TaskGroup.SPEED,
    metrics=[
        MetricConfig(
            name="speed",
            pretty_name="Tokens per second",
            huggingface_id="",
            results_key="speed",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.0f}"),
        ),
        MetricConfig(
            name="speed_short",
            pretty_name="Tokens per second on short documents",
            huggingface_id="",
            results_key="speed",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.0f}"),
        ),
    ],
)
