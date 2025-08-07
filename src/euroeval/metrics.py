"""All the metrics used in EuroEval."""

import abc
import logging
import typing as t

import evaluate
import litellm
from litellm.types.utils import Choices, ModelResponse
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from .exceptions import InvalidBenchmark
from .utils import HiddenPrints

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from evaluate import EvaluationModule

logger = logging.getLogger(__name__)


class Metric(abc.ABC):
    """Abstract base class for all metrics."""

    def __init__(
        self,
        name: str,
        pretty_name: str,
        postprocessing_fn: t.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
        """Initialise the metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
            postprocessing_fn:
                A function to apply to the metric scores after they are computed,
                taking the score to the postprocessed score along with its string
                representation. Defaults to x -> (100 * x, f"{x:.2%}").
        """
        self.name = name
        self.pretty_name = pretty_name
        self.postprocessing_fn = (
            postprocessing_fn
            if postprocessing_fn is not None
            else lambda x: (100 * x, f"{x:.2%}")
        )

    @abc.abstractmethod
    def __call__(
        self, predictions: t.Sequence, references: t.Sequence, dataset: "Dataset | None"
    ) -> float | None:
        """Calculate the metric score.

        Args:
            predictions:
                The model predictions.
            references:
                The ground truth references.
            dataset:
                The dataset used for evaluation. This is only used in case any
                additional metadata is used to compute the metrics.

        Returns:
            The calculated metric score, or None if the score should be ignored.
        """
        ...

    def __hash__(self) -> int:
        """Return a hash of the metric configuration."""
        return hash(self.name)


class HuggingFaceMetric(Metric):
    """A metric which is implemented in the `evaluate` package.

    Attributes:
        name:
            The name of the metric in snake_case.
        pretty_name:
            The pretty name of the metric, used for display purposes.
        huggingface_id:
            The Hugging Face ID of the metric.
        results_key:
            The name of the key used to extract the metric scores from the results
            dictionary.
        compute_kwargs:
            Keyword arguments to pass to the metric's compute function. Defaults to
            an empty dictionary.
    """

    def __init__(
        self,
        name: str,
        pretty_name: str,
        huggingface_id: str,
        results_key: str,
        compute_kwargs: dict[str, t.Any] | None = None,
        postprocessing_fn: t.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
        """Initialise the Hugging Face metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
            huggingface_id:
                The Hugging Face ID of the metric.
            results_key:
                The name of the key used to extract the metric scores from the results
                dictionary.
            compute_kwargs:
                Keyword arguments to pass to the metric's compute function. Defaults to
                an empty dictionary.
            postprocessing_fn:
                A function to apply to the metric scores after they are computed, taking
                the score to the postprocessed score along with its string
                representation. Defaults to x -> (100 * x, f"{x:.2%}").
        """
        super().__init__(
            name=name, pretty_name=pretty_name, postprocessing_fn=postprocessing_fn
        )
        self.huggingface_id = huggingface_id
        self.results_key = results_key
        self.compute_kwargs: dict[str, t.Any] = (
            dict() if compute_kwargs is None else compute_kwargs
        )
        self.metric: "EvaluationModule | None" = None

    def __call__(
        self, predictions: t.Sequence, references: t.Sequence, dataset: "Dataset | None"
    ) -> float | None:
        """Calculate the metric score.

        Args:
            predictions:
                The model predictions.
            references:
                The ground truth references.
            dataset:
                The dataset used for evaluation. This is only used in case any
                additional metadata is used to compute the metrics.

        Returns:
            The calculated metric score, or None if the score should be ignored.
        """
        if self.metric is None:
            self.metric = evaluate.load(path=self.huggingface_id)

        with HiddenPrints():
            results = self.metric.compute(
                predictions=predictions, references=references, **self.compute_kwargs
            )

        # The metric returns None if we are running on multi-GPU and the current
        # process is not the main process
        if results is None:
            return None

        score = results[self.results_key]
        if isinstance(score, list):
            score = sum(score) / len(score)

        return score


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
        scoring_fn: t.Callable[[BaseModel], float],
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
        self.scoring_fn = scoring_fn
        self.condition_formatting_fn = condition_formatting_fn
        self.system_prompt = system_prompt

    def __call__(
        self, predictions: t.Sequence, references: t.Sequence, dataset: "Dataset | None"
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

        Returns:
            The calculated metric score, or None if the score should be ignored.

        Raises:
            InvalidBenchmark:
                If the number of predictions does not match the number of references,
                or if the user prompt requires a condition but none is provided.
        """
        if not predictions or not references:
            return None
        elif len(predictions) != len(references):
            raise InvalidBenchmark(
                f"The number of predictions ({len(predictions):,}) does not match the "
                f"number of references ({len(references):,})."
            )

        # Prepare the messages for the LLM
        conversations: list[list[dict[str, str]]] = [
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

        # Get the judge generations
        generations = [
            litellm.completion(
                model=self.judge_id,
                messages=conversation,
                response_format=self.response_format,
                **self.judge_kwargs,
            )
            for conversation in tqdm(
                iterable=conversations,
                desc=f"Computing {self.pretty_name} scores",
                unit="sample",
            )
        ]

        # Extract the outputs from the generations
        outputs: list[BaseModel] = list()
        for generation in generations:
            assert isinstance(generation, ModelResponse), (
                f"The judge model did not return a valid response: {generation!r}"
            )
            choice = generation.choices[0]
            assert isinstance(choice, Choices), (
                f"The judge model did not return a valid choice: {choice!r}"
            )
            json_content = choice.message.content
            assert json_content is not None, (
                "The judge model returned a None content in the response message."
            )
            output = self.response_format.model_validate_json(json_data=json_content)
            outputs.append(output)

        # Calculate the scores using the scoring function
        scores = [self.scoring_fn(output) for output in outputs]
        if not scores:
            logger.warning(f"No scores were calculated for {self.pretty_name}.")
            return None
        return sum(scores) / len(scores)

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


class SpeedMetric(Metric):
    """Speed metric."""

    def __init__(self, name: str, pretty_name: str) -> None:
        """Initialise the speed metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
        """
        super().__init__(
            name=name,
            pretty_name=pretty_name,
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.0f}"),
        )

    def __call__(
        self, _: t.Sequence, __: t.Sequence, ___: "Dataset | None"
    ) -> float | None:
        """Not used with the speed metric, but required for consistency."""
        raise NotImplementedError


mcc_metric = HuggingFaceMetric(
    name="mcc",
    pretty_name="Matthew's Correlation Coefficient",
    huggingface_id="matthews_correlation",
    results_key="matthews_correlation",
)

macro_f1_metric = HuggingFaceMetric(
    name="macro_f1",
    pretty_name="Macro-average F1-score",
    huggingface_id="f1",
    results_key="f1",
    compute_kwargs=dict(average="macro"),
)

micro_f1_metric = HuggingFaceMetric(
    name="micro_f1",
    pretty_name="Micro-average F1-score with MISC tags",
    huggingface_id="seqeval",
    results_key="overall_f1",
)

micro_f1_no_misc_metric = HuggingFaceMetric(
    name="micro_f1_no_misc",
    pretty_name="Micro-average F1-score without MISC tags",
    huggingface_id="seqeval",
    results_key="overall_f1",
)

f1_metric = HuggingFaceMetric(
    name="f1",
    pretty_name="F1-score",
    huggingface_id="squad_v2",
    results_key="f1",
    postprocessing_fn=lambda x: (x, f"{x:.2f}%"),
)

em_metric = HuggingFaceMetric(
    name="em",
    pretty_name="Exact Match",
    huggingface_id="squad_v2",
    results_key="exact",
    postprocessing_fn=lambda x: (x, f"{x:.2f}%"),
)

bert_score_metric = HuggingFaceMetric(
    name="bertscore",
    pretty_name="BERTScore",
    huggingface_id="bertscore",
    results_key="f1",
    compute_kwargs=dict(
        model_type="microsoft/mdeberta-v3-base", device="auto", batch_size=1
    ),
)

rouge_l_metric = HuggingFaceMetric(
    name="rouge_l", pretty_name="ROUGE-L", huggingface_id="rouge", results_key="rougeL"
)

accuracy_metric = HuggingFaceMetric(
    name="accuracy",
    pretty_name="Accuracy",
    huggingface_id="accuracy",
    results_key="accuracy",
)


class Fluency(BaseModel):
    """Response format for the fluency metric.

    Attributes:
        fluency:
            The fluency rating, an integer between 1 and 5.
    """

    fluency: t.Annotated[int, Field(ge=1, le=5)]


# Example LLM-as-a-judge metric, to measure the fluency of the LLM output
fluency_metric = LLMAsAJudgeMetric(
    name="fluency",
    pretty_name="Fluency",
    judge_id="gpt-4o-mini",
    judge_kwargs=dict(temperature=0.0),
    user_prompt="Please rate the fluency of the following text on a scale from 1 to 5, "
    "with the following definitions:\n"
    "- 1: Very poor fluency, many grammatical errors\n"
    "- 2: Poor fluency, several grammatical errors\n"
    "- 3: Average fluency, a few grammatical errors\n"
    "- 4: Good fluency, no grammatical errors but sounds a bit off\n"
    "- 5: Excellent fluency, no grammatical errors and sounds natural\n\n"
    "Text: {prediction!r}\n\n"
    "Output your rating as a JSON object with a single key 'fluency'.",
    response_format=Fluency,
    scoring_fn=lambda output: (output.fluency - 1) / 4.0,
)

speed_metric = SpeedMetric(name="speed", pretty_name="Tokens per second")

speed_short_metric = SpeedMetric(
    name="speed_short", pretty_name="Tokens per second on short documents"
)
