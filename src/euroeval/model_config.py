"""Functions related to getting the model configuration."""

import logging
import typing as t

from . import benchmark_modules
from .exceptions import InvalidModel, NeedsEnvironmentVariable, NeedsExtraInstalled

if t.TYPE_CHECKING:
    from .data_models import BenchmarkConfig, ModelConfig


logger = logging.getLogger("euroeval")


def get_model_config(
    model_id: str, benchmark_config: "BenchmarkConfig"
) -> "ModelConfig":
    """Fetches configuration for a model.

    Args:
        model_id:
            The model ID.
        benchmark_config:
            The configuration of the benchmark.

    Returns:
        The model configuration.

    Raises:
        InvalidModel:
            If all model setups can handle the model, but the model does not exist.
    """
    all_benchmark_modules = [
        cls
        for cls in benchmark_modules.__dict__.values()
        if isinstance(cls, type)
        and issubclass(cls, benchmark_modules.BenchmarkModule)
        and cls is not benchmark_modules.BenchmarkModule
    ]
    all_benchmark_modules.sort(key=lambda cls: cls.high_priority, reverse=True)

    needs_extras: list[str] = list()
    needs_env_vars: list[str] = list()
    for benchmark_module in all_benchmark_modules:
        exists_or_err = benchmark_module.model_exists(
            model_id=model_id, benchmark_config=benchmark_config
        )
        if isinstance(exists_or_err, NeedsExtraInstalled):
            needs_extras.append(exists_or_err.extra)
        elif isinstance(exists_or_err, NeedsEnvironmentVariable):
            needs_env_vars.append(exists_or_err.env_var)
        elif exists_or_err is True:
            logger.debug(
                f"The model {model_id!r} was identified by the "
                f"{benchmark_module.__name__} benchmark module."
            )
            model_config = benchmark_module.get_model_config(
                model_id=model_id, benchmark_config=benchmark_config
            )
            return model_config
    else:
        msg = f"Model {model_id} not found."
        if needs_extras:
            msg += (
                " However, it is possible that the model exists, but a package "
                "needs to be installed to check if it exists. Please try running "
                f"`pip install euroeval[{','.join(needs_extras)}]` or `pip install "
                "euroeval[all]`, and try again."
            )
        elif needs_env_vars:
            msg += (
                " However, it is possible that the model exists, but an environment "
                "variable needs to be set to check if it exists. Please set the "
                f"environment variables {','.join(needs_env_vars)} and try again."
            )
        raise InvalidModel(msg)
