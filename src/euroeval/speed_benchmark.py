"""Benchmarking model inference speed."""

import logging

import pyinfer
from tqdm.auto import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from .benchmark_modules import (
    BenchmarkModule,
    HuggingFaceEncoderModel,
    LiteLLMModel,
    VLLMModel,
)
from .data_models import BenchmarkConfig
from .exceptions import InvalidBenchmark
from .utils import clear_memory

logger = logging.getLogger("euroeval")


def benchmark_speed(
    model: "BenchmarkModule", benchmark_config: "BenchmarkConfig"
) -> list[dict[str, float]]:
    """Benchmark model inference speed.

    Args:
        model:
            Model to use.
        benchmark_config:
            Configuration for the benchmark.

    Returns:
        Dictionary of scores.
    """
    scores: list[dict[str, float]] = list()
    for idx in tqdm(
        iterable=range(benchmark_config.num_iterations),
        desc="Benchmarking",
        disable=not benchmark_config.progress_bar,
    ):
        itr_scores = benchmark_speed_single_iteration(model=model, itr_idx=idx)
        clear_memory()
        scores.append(itr_scores)
        logger.debug(f"Scores for iteration {idx}: {itr_scores}")
    return scores


def benchmark_speed_single_iteration(
    model: "BenchmarkModule", itr_idx: int
) -> dict[str, float]:
    """Run a single iteration of the speed benchmark.

    Args:
        model:
            The model to use in the benchmark.
        itr_idx:
            The index of the iteration.

    Returns:
        A dictionary containing the scores for the current iteration.
    """
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)

    base_doc = "Document which contains roughly 10 tokens. "
    multiplier = 10 * (1 + itr_idx)
    doc = base_doc * multiplier
    short_multiplier = 1.25 * (1 + itr_idx)
    short_doc = base_doc * round(short_multiplier)

    def generate_messages_predict(doc: str) -> None:
        model.generate(inputs=dict(messages=[[dict(role="user", content=doc)]]))

    def generate_prompt_predict(doc: str) -> None:
        model.generate(inputs=dict(text=[doc]))

    def encoder_predict(doc: str) -> None:
        tokenizer = model.get_tokenizer()
        pytorch_model = model.get_pytorch_module()
        inputs = {
            key: tensor.to(pytorch_model.device)
            for key, tensor in tokenizer(
                text=[doc], truncation=True, return_tensors="pt"
            ).items()
        }
        pytorch_model(**inputs)

    if isinstance(model, VLLMModel):
        predict = generate_prompt_predict
    elif isinstance(model, LiteLLMModel):
        predict = generate_messages_predict
    elif isinstance(model, HuggingFaceEncoderModel):
        predict = encoder_predict
    else:
        raise ValueError(f"Model type {model} not supported for speed benchmark")

    try:
        # Do a warmup run, as the first run is always slower
        pyinfer.InferenceReport(model=predict, inputs=base_doc, n_seconds=1).run(
            print_report=False
        )

        speed_scores = pyinfer.InferenceReport(
            model=predict, inputs=doc, n_seconds=3
        ).run(print_report=False)
        num_gpt2_tokens = len(gpt2_tokenizer([doc], truncation=True)["input_ids"][0])
        gpt2_tokens_per_second = speed_scores["Infer(p/sec)"] * num_gpt2_tokens

        speed_scores_short = pyinfer.InferenceReport(
            model=predict, inputs=short_doc, n_seconds=3
        ).run(print_report=False)
        num_gpt2_tokens_short = len(
            gpt2_tokenizer([short_doc], truncation=True)["input_ids"][0]
        )
        gpt2_tokens_per_second_short = (
            speed_scores_short["Infer(p/sec)"] * num_gpt2_tokens_short
        )

    except (RuntimeError, ValueError, IndexError) as e:
        raise InvalidBenchmark(f"Speed benchmark failed with error: {e!r}")

    return dict(
        test_speed=gpt2_tokens_per_second, test_speed_short=gpt2_tokens_per_second_short
    )
