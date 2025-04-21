"""Ollama module for benchmarking language models using the Ollama API."""

import asyncio

from ollama import AsyncClient, Client

from ..data_models import BenchmarkConfig, InferenceBackend, ModelConfig, ModelType
from ..types import GenerativeModelOutput
from ..utils import create_model_cache_dir
from .base import BatchingPreference, BenchmarkModule


class OllamaModule(BenchmarkModule):
    """A benchmark module for running language models through the Ollama API."""

    fresh_model = False
    batching_preference = BatchingPreference.ALL_AT_ONCE
    high_priority = True

    @classmethod
    def model_exists(cls, model_id: str, benchmark_config: BenchmarkConfig) -> bool:
        """Check if the specified model exists in Ollama.

        Args:
            model_id: The ID of the model to check.
            benchmark_config: The benchmark configuration.

        Returns:
            True if the model exists, False otherwise.
        """
        # use Client().show to verify
        return Client().show(model_id).modelinfo is not None

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Get the model configuration for the specified model.

        Args:
            model_id: The ID of the model to get the configuration for.
            benchmark_config: The benchmark configuration.

        Returns:
            The model configuration.
        """
        return ModelConfig(
            model_id=model_id,
            revision="",
            task="text-generation",
            languages=[],
            inference_backend=InferenceBackend.OLLAMA,
            model_type=ModelType.GENERATIVE,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=model_id
            ),
            merge=False,
            adapter_base_model_id=None,
        )

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate text from the model.

        Args:
            inputs: The input data for generation.

        Returns:
            The generated text.
        """
        client = AsyncClient()
        outputs = asyncio.run(
            client.chat(
                model=self.model_config.model_id,
                messages=inputs["messages"],
                stream=False,
            )
        )
        texts = [out.message["content"].strip() for out in outputs]
        return GenerativeModelOutput(sequences=texts)
