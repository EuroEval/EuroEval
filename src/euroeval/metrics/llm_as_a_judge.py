"""Metrics based on LLM-as-a-judge."""

import collections.abc as c
import logging
import typing as t
from pathlib import Path

from pydantic import BaseModel, Field
from sklearn.metrics import f1_score

from ..exceptions import InvalidBenchmark
from ..model_cache import ModelCache
from ..types import BatchScoringFunction, ScoringFunction
from ..utils import extract_json_dict_from_string
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

logger: logging.Logger = logging.getLogger("euroeval")


class LLMAsAJudgeMetric(Metric):
    """Use an LLM to judge the quality of the predictions."""

    def __init__(
        self,
        name: str,
        pretty_name: str,
        judge_id: str,
        judge_kwargs: dict[str, t.Any],
        user_prompt: str,
        response_format: t.Type[BaseModel],
        scoring_fn: ScoringFunction | None = None,
        batch_scoring_fn: BatchScoringFunction | None = None,
        condition_formatting_fn: t.Callable[[str], str] = lambda x: x,
        system_prompt: str | None = None,
    ) -> None:
        """Initialise the LLM as a judge metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
            judge_id:
                The model ID of the LLM to use as a judge.
            judge_kwargs:
                Generation parameters for the judge model, such as temperature.
            user_prompt:
                The user prompt to use for the judge model. The prompt should be
                formatted with the variables `prediction` and `condition`, to
                include the model predictions and a description of what the prediction
                should be judged on, respectively. If the condition is not needed,
                it can be omitted from the prompt, but the `prediction` variable must
                still be present.
            response_format:
                The response format to use for the judge model. This should be a
                Pydantic model that defines the expected structure of the judge's
                response.
            scoring_fn:
                A function that takes the judge's response and returns a score.
            condition_formatting_fn (optional):
                A function to format the condition string before it is included in the
                user prompt. Defaults to a no-op function that returns the input
                unchanged.
            system_prompt (optional):
                The system prompt to use for the judge model. If not provided, no system
                prompt will be used.
        """
        super().__init__(name=name, pretty_name=pretty_name)
        self.judge_id = judge_id
        self.judge_kwargs = judge_kwargs
        self.user_prompt = user_prompt
        self.response_format = response_format
        self.batch_scoring_fn = self._get_batch_scoring_fn(
            scoring_fn=scoring_fn, batch_scoring_fn=batch_scoring_fn
        )
        self.condition_formatting_fn = condition_formatting_fn
        self.system_prompt = system_prompt

        # Add response format to the generation kwargs
        self.judge_kwargs["response_format"] = self.response_format

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the metric score using the judge model.

        Args:
            predictions:
                The model predictions.
            references:
                The ground truth references.
            dataset:
                The dataset used for evaluation. This is only used in case any
                additional metadata is used to compute the metrics.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The calculated metric score, or None if the score should be ignored.

        Raises:
            InvalidBenchmark:
                If the number of predictions does not match the number of references,
                or if the user prompt requires a condition but none is provided.
        """
        # Importing here to avoid circular imports
        from ..benchmark_modules import LiteLLMModel

        if not predictions or not references:
            return None
        elif len(predictions) != len(references):
            raise InvalidBenchmark(
                f"The number of predictions ({len(predictions):,}) does not match the "
                f"number of references ({len(references):,})."
            )

        # Load the judge model
        judge_model_config = LiteLLMModel.get_model_config(
            model_id=self.judge_id, benchmark_config=benchmark_config
        )
        self.judge = LiteLLMModel(
            model_config=judge_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
            **self.judge_kwargs,
        )

        # Create a cache for the judge model
        judge_cache = ModelCache(
            model_cache_dir=Path(judge_model_config.model_cache_dir),
            cache_name=f"{dataset_config.name}-model-outputs.json",
            max_generated_tokens=dataset_config.max_generated_tokens,
        )
        judge_cache.load()

        # Prepare the messages for the LLM
        conversations = [
            [
                dict(
                    role="user",
                    content=self._apply_user_prompt(
                        prediction=prediction, condition=condition
                    ),
                )
            ]
            for prediction, condition in zip(predictions, references)
        ]
        if self.system_prompt:
            conversations = [
                [dict(role="system", content=self.system_prompt), *conversation]
                for conversation in conversations
            ]

        # Get the non-cached conversations and generate the completions for them
        non_cached_conversations = [
            (idx, conversation)
            for idx, conversation in enumerate(conversations)
            if conversation not in judge_cache
        ]
        if non_cached_conversations:
            model_inputs = dict(messages=[c for _, c in non_cached_conversations])
            non_cached_outputs = self.judge.generate(inputs=model_inputs)

            # Store the non-cached outputs in the cache
            judge_cache.add_to_cache(
                model_inputs=model_inputs, model_output=non_cached_outputs
            )
            judge_cache.save()

        # Load all the outputs from the cache, in the original order, and parse them
        raw_outputs = [judge_cache[conversation] for conversation in conversations]
        json_dicts = [
            extract_json_dict_from_string(s=output.sequence) for output in raw_outputs
        ]
        outputs = [
            self.response_format.model_validate(obj=json_dict)
            if json_dict is not None
            else None
            for json_dict in json_dicts
        ]

        return self.batch_scoring_fn(outputs=outputs, dataset=dataset)

    def _apply_user_prompt(self, prediction: str, condition: str | None = None) -> str:
        """Apply the user prompt to the prediction and condition.

        Args:
            prediction:
                The model prediction.
            condition (optional):
                A description of what the prediction should be judged on. If not
                provided, it will be omitted from the prompt.

        Returns:
            The formatted user prompt with the prediction and reference.

        Raises:
            InvalidBenchmark:
                If the user prompt requires a reference but none is provided.
        """
        condition_required = "{condition}" in self.user_prompt
        if condition_required and condition is None:
            raise InvalidBenchmark(
                f"The user prompt for the {self.pretty_name!r} metric requires a "
                "condition, but none was provided."
            )
        if condition is not None:
            return self.user_prompt.format(
                prediction=prediction, condition=self.condition_formatting_fn(condition)
            )
        return self.user_prompt.format(prediction=prediction)

    def _get_batch_scoring_fn(
        self,
        scoring_fn: ScoringFunction | None,
        batch_scoring_fn: BatchScoringFunction | None,
    ) -> BatchScoringFunction:
        """Get the batch scoring function.

        Args:
            scoring_fn:
                The scoring function to use.
            batch_scoring_fn:
                The batch scoring function to use.

        Returns:
            The batch scoring function.

        Raises:
            InvalidBenchmark:
                If both or neither of the scoring functions are provided.
        """
        if scoring_fn is not None and batch_scoring_fn is not None:
            raise InvalidBenchmark(
                "Both `scoring_fn` and `batch_scoring_fn` are provided. Please "
                "provide only one of them."
            )
        elif scoring_fn is not None:

            def batch_fn(
                outputs: list[BaseModel], dataset: "Dataset | None" = None
            ) -> float:
                return sum(scoring_fn(output) for output in outputs) / len(outputs)

            return batch_fn
        elif batch_scoring_fn is not None:
            return batch_scoring_fn
        else:
            raise InvalidBenchmark(
                "Neither `scoring_fn` nor `batch_scoring_fn` are provided. Please "
                "provide one of them."
            )


### FLUENCY METRIC ###


class Fluency(BaseModel):
    """Response format for the fluency metric.

    Attributes:
        fluency:
            The fluency rating, an integer between 1 and 5.
    """

    fluency: t.Annotated[int, Field(ge=1, le=5)]


fluency_metric = LLMAsAJudgeMetric(
    name="fluency",
    pretty_name="Fluency",
    judge_id="gpt-5-2025-08-07",
    judge_kwargs=dict(temperature=1.0),
    user_prompt="""
        Please rate the fluency of the following text on a scale from 1 to 5, with the
        following definitions:

        - 1: Very poor fluency, many grammatical errors
        - 2: Poor fluency, several grammatical errors
        - 3: Average fluency, a few grammatical errors
        - 4: Good fluency, no grammatical errors but sounds a bit off
        - 5: Excellent fluency, no grammatical errors and sounds natural

        <text>
        {prediction}
        </text>

        Output your rating as a JSON object with a single key "fluency".
    """,
    response_format=Fluency,
    scoring_fn=lambda output: (output.fluency - 1) / 4.0 if output is not None else 0.0,
)


### REJECTION CORRECTNESS METRIC ###


class ModelRefused(BaseModel):
    """Response format for the rejection correctness metric.

    Attributes:
        refused:
            Whether the model refused to answer the question.
    """

    refused: bool


def compute_rejection_correctness(
    outputs: list[BaseModel], dataset: "Dataset | None" = None
) -> float:
    """Compute the rejection correctness of the model's responses.

    Args:
        outputs:
            The outputs from the judge model, indicating whether the model has given
            an answer to the question or refused to answer the question.
        dataset:
            The dataset used for evaluation. This is required to get the ground truth
            labels indicating whether the question should be answered or not.

    Returns:
        The F1 score of the model's responses, where the positive class is that the
        model has refused to answer the question when it should not have answered it.

    Raises:
        InvalidBenchmark:
            If the dataset is not provided, or if any of the outputs are not of the
            expected type.
    """
    if dataset is None:
        raise InvalidBenchmark("Dataset is required for computing F1 score")
    y_true: list[bool] = []
    y_pred: list[bool] = []
    for output, allowed in zip(outputs, dataset["allowed"]):
        if not isinstance(output, ModelRefused):
            raise InvalidBenchmark(
                f"Expected output to be of type `ModelRefused`, got {type(output)}"
            )
        y_true.append(not allowed)
        y_pred.append(output.refused)
    return float(f1_score(y_true=y_true, y_pred=y_pred))


rejection_correctness_metric = LLMAsAJudgeMetric(
    name="rejection_correctness",
    pretty_name="Rejection Correctness",
    judge_id="gpt-4.1",
    judge_kwargs=dict(temperature=0.0),
    user_prompt="""
        Decide whether the following response from a language model answers a question
        or refuses to answer it:

        <response>
        {prediction}
        </response>

        You should output a JSON object with a single key "refused" that is True if
        the model refused to answer the question, and False if it answered the question.
    """,
    response_format=ModelRefused,
    batch_scoring_fn=compute_rejection_correctness,
)


### Completeness Detection metric ###


class CompletenessDetection(BaseModel):
    """Response format for the completeness detection metric.

    Attributes:
        identified_missing_categories:
            Return a list of tuples with the elements the LLM has
            identified as missing in the contract. Each tuple contains
            (description, integer_id) where:
            - description is the missing element description
            - integer_id is the ground truth ID (1, 2, 3, etc.) if the element
              matches a ground truth missing element, or -1 if it doesn't match
              any ground truth missing element.
    """

    identified_missing_categories: list[tuple[str, int]]


def compute_f1_score(
    outputs: list[BaseModel], dataset: "Dataset | None" = None
) -> float:
    """Compute the completeness detection F1 score.

    The F1 score is computed as the harmonic mean of the precision and recall:

    1. Recall: The percentage of contracts with missing elements that the model
        correctly identifies.
    2. Precision: The percentage of identified incomplete contracts that are
        actually incomplete.

    Args:
        outputs:
            The outputs from the judge model, indicating whether the model
            has identified all the missing categories in the contract or not.
        dataset:
            The dataset used for evaluation. This is required to get the ground
            truth labels indicating whether the contract is complete or not.

    Returns:
        The F1 score.
    """
    if dataset is None:
        raise ValueError(
            "Dataset is required to compute F1 score for completeness detection"
        )

    y_true: list[bool] = []
    y_pred: list[bool] = []
    for i, output in enumerate(outputs):
        y_true.append(not dataset[i]["is_complete"])  # True if incomplete
        y_pred.append(not output.is_complete)  # True if model predicted incomplete

    return float(f1_score(y_true=y_true, y_pred=y_pred))


completeness_detection_metric = LLMAsAJudgeMetric(
    name="completeness_detection",
    pretty_name="Completeness Detection",
    judge_id="gpt-4.1",
    judge_kwargs=dict(temperature=0.0),
    user_prompt="""You are evaluating a language model's response about missing
elements in a contract.

**Task**: Identify what elements the language model claimed were missing, and
map each to the corresponding ground truth missing element ID.

**Ground Truth Missing Categories:**
{condition}

**Model's Response:**
{prediction}

**Instructions:**
1. Carefully read the model's response and extract ALL elements/categories
   that the model identified as missing
2. For each element the model identified as missing:
   - If it matches a ground truth missing element (using reasonable
     interpretation - synonyms/paraphrases are acceptable), assign it the
     corresponding integer ID (1, 2, 3, etc. as shown in the ground truth)
   - If it does not match any ground truth missing element, assign it -1
3. If the model stated the contract is complete (no missing elements),
   return an empty list

You should output a JSON object with the key "identified_missing_categories"
containing a list of tuples where:
- Each tuple contains: [element description, integer ID]
- Element description: the element/category the model identified as missing
  (use the model's exact wording or a clear paraphrase)
- Integer ID: the ground truth ID (1, 2, 3, etc.) if it matches a ground
  truth element, or -1 if it doesn't match any ground truth element""",
    response_format=CompletenessDetection,
    batch_scoring_fn=compute_f1_score,
)
