"""Metrics based on LLM-as-a-judge."""

import collections.abc as c
import logging
import typing as t
from pathlib import Path

from pydantic import BaseModel, Field

from ..exceptions import InvalidBenchmark
from ..logging_utils import log
from ..utils import extract_json_dict_from_string
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig


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
        scoring_fn: t.Callable[..., float],
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
        from ..model_cache import ModelCache

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
            progress_bar=benchmark_config.progress_bar,
            hash_inputs=not benchmark_config.debug,
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

        # Calculate the scores using the scoring function
        scores = [self.scoring_fn(output) for output in outputs]
        if not scores:
            log(
                f"No scores were calculated for {self.pretty_name}.",
                level=logging.WARNING,
            )
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


### Fluency metric ###


class GEvalOutput(BaseModel):
    """Response format for the GEval metric.

    Attributes:
        coherency:
            Coherency score between 1 and 5.
        consistency:
            Consistency score between 1 and 5.
        fluency:
            Fluency score between 1 and 3.
        relevance:
            Relevance score between 1 and 5.
    """

    coherency: t.Annotated[int, Field(ge=1, le=5)]
    consistency: t.Annotated[int, Field(ge=1, le=5)]
    fluency: t.Annotated[int, Field(ge=1, le=3)]
    relevance: t.Annotated[int, Field(ge=1, le=5)]


def geval_scoring_fn(output: GEvalOutput | None) -> float:
    """Scoring function for the GEval metric.

    Args:
        output:
            The GEval output object, containing all the raw scores, or None if the
            output could not be parsed.

    Returns:
        The overall GEval score between 0.0 and 1.0.
    """
    if output is None:
        return 0.0
    normalised_coherency = (output.coherency - 1) / 4
    normalised_consistency = (output.consistency - 1) / 4
    normalised_fluency = (output.fluency - 1) / 2
    normalised_relevance = (output.relevance - 1) / 4
    overall_score = (
        normalised_coherency
        + normalised_consistency
        + normalised_fluency
        + normalised_relevance
    ) / 4
    return overall_score


geval_metric = LLMAsAJudgeMetric(
    name="geval",
    pretty_name="GEval",
    judge_id="gpt-4.1-2025-04-14",
    judge_kwargs=dict(temperature=0.0),
    response_format=GEvalOutput,
    scoring_fn=geval_scoring_fn,
    system_prompt="""
<instruction>
    You will be given one summary written for an article. Your task is to rate the
    summary on one metric.
</instruction>
<list_of_evaluation_criteria>
    <evaluation_criterion>
        <name>
            Coherency
        </name>
        <scale>
            1-5
        </scale>
        <description>
            The collective quality of all sentences. We align this dimension with the
            DUC quality question of structure and coherence whereby "the summary should
            be well-structured and well-organized. The summary should not just be a heap
            of related information, but should build from sentence to a coherent body of
            information about a topic.

            - 1. Incoherent: The summary is disorganized and hard to follow. Sentences
              do not logically connect, making it difficult to understand the overall
              message.
            - 2. Decent Coherency: The summary has some organization, but there are
              noticeable gaps in the flow of ideas. Some sentences may feel out of place
              or disconnected from the main topic.
            - 3. Good Coherency: The summary is generally well-organized, with
              sentences that mostly connect logically. There may be minor issues with
              transitions or flow, but the overall message is clear. This is the level
              expected of a good human-written summary.
            - 4. Very Good Coherency: The summary is well-structured and easy to follow.
              Sentences connect logically, and transitions between ideas are smooth.
              This level is expected of very highly skilled human summarizers.
            - 5. Excellent Coherency: The summary is exceptionally well-organized and
              coherent. Ideas flow seamlessly from one to the next, creating a clear and
              compelling narrative. This level is rarely achieved, even by expert human
              summarizers.
        </description>
    </evaluation_criterion>
    <evaluation_criterion>
        <name>
            Consistency
        </name>
        <scale>
            1-5
        </scale>
        <description>
            The factual alignment between the summary and the summarized source. A
            factually consistent summary contains only statements that are entailed by
            the source document. Annotators were also asked to penalize summaries that
            contained hallucinated facts.

            - 1. Inconsistent: The summary contains multiple factual inaccuracies or
                hallucinations that contradict the source document.
            - 2. Somewhat Consistent: The summary has some factual alignment with the
                source, but there are noticeable inaccuracies or hallucinations present.
            - 3. Mostly Consistent: The summary is generally factually aligned with the
                source, with only minor inaccuracies or hallucinations that do not
                significantly affect the overall understanding. This is the level
                expected of a good human-written summary.
            - 4. Very Consistent: The summary is factually accurate and aligns well with
                the source document, with only rare and minor inaccuracies. This level
                is expected of very highly skilled human summarizers.
            - 5. Perfectly Consistent: The summary is entirely factually accurate and
                fully aligns with the source document, containing no inaccuracies or
                hallucinations. This level is rarely achieved, even by expert human
                summarizers.
        </description>
    </evaluation_criterion>
    <evaluation_criterion>
        <name>
            Fluency
        </name>
        <scale>
            1-5
        </scale>
        <description>
            The quality of the summary in terms of grammar, spelling, punctuation, word
            choice, and sentence structure.

            - 1. Poor Fluency: The summary contains numerous grammatical errors,
                spelling mistakes, and awkward sentence structures that significantly
                hinder readability.
            - 2. Fair Fluency: The summary has several grammatical errors and awkward
                phrasings, but the overall meaning is still understandable.
            - 3. Good Fluency: The summary is generally well-written, with only minor
                grammatical errors or awkward phrasings that do not significantly affect
                readability. This is the level expected of a good human-written summary.
            - 4. Very Good Fluency: The summary is well-written and easy to read
                overall, with very few grammatical errors or awkward phrasings. This
                level is expected of very highly skilled human summarizers.
            - 5. Excellent Fluency: The summary is exceptionally well-written, with
                flawless grammar, spelling, and sentence structure that make it a
                pleasure to read. This level is rarely achieved, even by expert human
                summarizers.
        </description>
    </evaluation_criterion>
    <evaluation_criterion>
        <name>
            Relevance
        </name>
        <scale>
            1-5
        </scale>
        <description>
            Selection of important content from the source. The summary should include
            only important information from the source document. Annotators were
            instructed to penalize summaries which contained redundancies and excess
            information.

            - 1. Irrelevant: The summary includes little to no important information
                from the source document, focusing instead on trivial or unrelated
                details.
            - 2. Somewhat Relevant: The summary includes some important information
                from the source, but also contains a significant amount of trivial or
                unrelated details.
            - 3. Mostly Relevant: The summary generally captures the important
                information from the source, with only minor inclusion of trivial or
                unrelated details that do not significantly detract from the overall
                understanding. This is the level expected of a good human-written
                summary.
            - 4. Very Relevant: The summary effectively captures the important
                information from the source document, with minimal inclusion of trivial
                or unrelated details. This level is expected of very highly skilled
                human summarizers.
            - 5. Perfectly Relevant: The summary exclusively includes important
                information from the source document, with no trivial or unrelated
                details. This level is rarely achieved, even by expert human
                summarizers.
        </description>
    </evaluation_criterion>
</list_of_evaluation_criteria>
<output_format>
    You will output a single JSON object with the keys "coherency", "consistency",
    "fluency", and "relevance", each with an integer as the associated value, which is
    within the specified scale as the value.
</output_format>
""",
    user_prompt="""
<text_to_evaluate>
{prediction}
</text_to_evaluate>
    """,
)
