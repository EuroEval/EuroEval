"""All benchmarks tasks used in EuroEval."""

from .data_models import MetricConfig, Task
from .enums import TaskGroup
from .prompt_templates import (
    LA_TEMPLATES,
    MULTIPLE_CHOICE_TEMPLATES,
    NER_TEMPLATES,
    RC_TEMPLATES,
    SENT_TEMPLATES,
    SUMM_TEMPLATES,
)


def get_all_tasks() -> dict[str, Task]:
    """Get a list of all the dataset tasks.

    Returns:
        A mapping between names of dataset tasks and their configurations.
    """
    return {cfg.name: cfg for cfg in globals().values() if isinstance(cfg, Task)}


LA = Task(
    name="linguistic-acceptability",
    task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
    template_dict=LA_TEMPLATES,
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
    default_num_few_shot_examples=12,
    default_max_generated_tokens=5,
    default_labels=["correct", "incorrect"],
)


NER = Task(
    name="named-entity-recognition",
    task_group=TaskGroup.TOKEN_CLASSIFICATION,
    template_dict=NER_TEMPLATES,
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
    default_num_few_shot_examples=8,
    default_max_generated_tokens=128,
    default_labels=[
        "o",
        "b-loc",
        "i-loc",
        "b-org",
        "i-org",
        "b-per",
        "i-per",
        "b-misc",
        "i-misc",
    ],
)


RC = Task(
    name="reading-comprehension",
    task_group=TaskGroup.QUESTION_ANSWERING,
    template_dict=RC_TEMPLATES,
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
    default_num_few_shot_examples=4,
    default_max_generated_tokens=32,
    default_labels=["start_positions", "end_positions"],
)


SENT = Task(
    name="sentiment-classification",
    task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
    template_dict=SENT_TEMPLATES,
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
    default_num_few_shot_examples=12,
    default_max_generated_tokens=5,
    default_labels=["positive", "neutral", "negative"],
)


SUMM = Task(
    name="summarization",
    task_group=TaskGroup.TEXT_TO_TEXT,
    template_dict=SUMM_TEMPLATES,
    metrics=[
        MetricConfig(
            name="bertscore",
            pretty_name="BERTScore",
            huggingface_id="bertscore",
            results_key="f1",
            compute_kwargs=dict(
                model_type="microsoft/mdeberta-v3-base", device="auto", batch_size=1
            ),
        ),
        MetricConfig(
            name="rouge_l",
            pretty_name="ROUGE-L",
            huggingface_id="rouge",
            results_key="rougeL",
        ),
    ],
    default_num_few_shot_examples=1,
    default_max_generated_tokens=256,
    default_labels=[],
)


KNOW = Task(
    name="knowledge",
    task_group=TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
    template_dict=MULTIPLE_CHOICE_TEMPLATES,
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
    default_num_few_shot_examples=5,
    default_max_generated_tokens=5,
    default_labels=["a", "b", "c", "d"],
)


MCRC = Task(
    name="multiple-choice-reading-comprehension",
    task_group=TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
    template_dict=MULTIPLE_CHOICE_TEMPLATES,
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
    default_num_few_shot_examples=5,
    default_max_generated_tokens=5,
    default_labels=["a", "b", "c", "d"],
)


COMMON_SENSE = Task(
    name="common-sense-reasoning",
    task_group=TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
    template_dict=MULTIPLE_CHOICE_TEMPLATES,
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
    default_num_few_shot_examples=5,
    default_max_generated_tokens=5,
    default_labels=["a", "b", "c", "d"],
)


SPEED = Task(
    name="speed",
    task_group=TaskGroup.SPEED,
    template_dict={},
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
    default_num_few_shot_examples=0,
    default_max_generated_tokens=5,
    default_labels=[],
)
