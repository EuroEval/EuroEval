"""Command-line entry point for ``euroeval-worker``."""

import argparse
import json
import logging
import typing as t
from pathlib import Path

from . import __version__
from .external import ExternalServerConfig, external_server_config

DEFAULT_SERVER = "https://euroeval.com/api/worker"
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the volunteer worker CLI.

    Args:
            argv (optional):
                Argv.

    Returns:
            Process exit status.
    """
    parser = _parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    external_config = _external_config(parser=parser, arguments=arguments)
    if arguments.gpu_health_check and not arguments.inspect_state:
        return _run_gpu_health_check()
    from .broker import BrokerClient, BrokerError  # noqa: PLC0415
    from .hardware import NoGpuError  # noqa: PLC0415
    from .runtime import Worker  # noqa: PLC0415
    from .state import StateStore, default_state_dir  # noqa: PLC0415

    try:
        state_dir = (
            Path(arguments.state_dir)
            if arguments.state_dir
            else default_state_dir(readonly=arguments.inspect_state)
        )
        state = StateStore(directory=state_dir)
        if arguments.inspect_state:
            active = state.load_active()
            print(
                json.dumps(
                    {
                        "schema_version": "volunteer-worker-state/v1",
                        "active": active is not None,
                        "finalised": bool(active and active.finalised_submission_id),
                        "managed_image_digest": getattr(
                            active, "managed_image_digest", None
                        ),
                        "lease_image_digest": (
                            active.lease.image_digest if active else None
                        ),
                        "execution_fingerprint": getattr(
                            active, "execution_fingerprint", None
                        ),
                    }
                )
            )
            return 0
        return (
            Worker(
                client=BrokerClient(server=arguments.server),
                state=state,
                gpu_memory_utilisation=arguments.gpu_memory_utilisation,
                image_digest=arguments.image_digest,
                external=external_config,
            ).run(once=arguments.once, launcher_managed=arguments.launcher_managed)
            or 0
        )
    except KeyboardInterrupt:
        logger.info("Worker stopped")
        if arguments.launcher_managed:
            return 130
    except BrokerError as error:
        if (
            arguments.launcher_managed
            and error.status == 426
            and error.code == "worker_update_required"
        ):
            return 75
        logger.error("%s", error)
        return 1
    except NoGpuError as error:
        logger.error("%s", error)
        return 1 if arguments.launcher_managed else 2
    except Exception:
        logger.exception("Worker stopped after an error")
        return 1
    return 0


_EXTERNAL_ARGUMENT_NAMES = (
    "api_base",
    "served_model",
    "gguf_file",
    "gguf_binding",
    "max_context_length",
    "generative_type",
)
_EXTERNAL_FLAG_CONFLICTS = (
    "gpu_health_check",
    "image_digest",
    "launcher_managed",
    "inspect_state",
)


def _external_config(
    *, parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> ExternalServerConfig | None:
    """Validate external-server arguments and build the frozen config.

    Supplying any external argument requires all of them, and external input
    cannot be combined with managed-image or GPU-health flags.

    Args:
        parser:
            The argument parser, used to report usage errors.
        arguments:
            Parsed command-line arguments.

    Returns:
        The frozen external-server configuration, or ``None`` in CUDA mode.
    """
    values = {name: getattr(arguments, name) for name in _EXTERNAL_ARGUMENT_NAMES}
    if not any(value is not None for value in values.values()):
        if arguments.api_key_env is not None:
            parser.error("--api-key-env requires the external-server arguments")
        return None
    missing = [name for name, value in values.items() if value is None]
    if missing:
        parser.error(
            "external-server mode requires all of "
            + ", ".join(
                f"--{name.replace('_', '-')}" for name in _EXTERNAL_ARGUMENT_NAMES
            )
            + f"; missing: {', '.join(missing)}"
        )
    for flag in _EXTERNAL_FLAG_CONFLICTS:
        if getattr(arguments, flag):
            parser.error(
                f"--{flag.replace('_', '-')} cannot be combined with "
                "external-server arguments"
            )
    try:
        return external_server_config(
            api_base=t.cast(str, values["api_base"]),
            served_model=t.cast(str, values["served_model"]),
            gguf_file=t.cast(Path, values["gguf_file"]),
            gguf_binding=t.cast(Path, values["gguf_binding"]),
            max_context_length=t.cast(int, values["max_context_length"]),
            generative_type=t.cast(str, values["generative_type"]),
            api_key_env=arguments.api_key_env,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
        return None


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser for worker execution and GPU health checks.

    Returns:
        Parser configured with broker, state, run-mode, memory, health-check, and
        version options.
    """
    parser = argparse.ArgumentParser(description="Run a EuroEval volunteer GPU worker.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Worker broker URL.")
    parser.add_argument(
        "--state-dir", type=Path, help="Directory for private worker state."
    )
    parser.add_argument(
        "--once", action="store_true", help="Claim at most one piece of work."
    )
    parser.add_argument("--launcher-managed", action="store_true")
    parser.add_argument("--image-digest")
    parser.add_argument("--inspect-state", action="store_true")
    parser.add_argument(
        "--gpu-memory-utilisation",
        "--gpu-memory-utilization",
        dest="gpu_memory_utilisation",
        type=float,
        default=0.8,
        help="Fraction of GPU memory offered to the evaluator (default: 0.8).",
    )
    parser.add_argument(
        "--gpu-health-check",
        action="store_true",
        help="Check the pinned CUDA and vLLM stack without broker access.",
    )
    parser.add_argument(
        "--api-base",
        help="Base URL of an external OpenAI-compatible inference server "
        "(external-server mode).",
    )
    parser.add_argument(
        "--served-model", help="Model alias served by the external inference server."
    )
    parser.add_argument(
        "--gguf-file",
        type=Path,
        help="Local GGUF file loaded by the external inference server.",
    )
    parser.add_argument(
        "--gguf-binding",
        type=Path,
        help="JSON file with one six-field artifact descriptor or the "
        "version-1 artifacts binding envelope.",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        help="Context-length override for the served model.",
    )
    parser.add_argument(
        "--generative-type",
        choices=["instruction_tuned", "reasoning"],
        help="Generative type of the served model; base is not in the pilot.",
    )
    parser.add_argument(
        "--api-key-env",
        help="Name of an environment variable holding the optional endpoint "
        "API key; the key is never persisted or logged.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _run_gpu_health_check() -> int:
    """Run the credential-free physical-canary health check.

    Returns:
        Process exit status.
    """
    from .hardware import NoGpuError  # noqa: PLC0415
    from .health import GpuHealthError, run_gpu_health_check  # noqa: PLC0415

    try:
        report = run_gpu_health_check()
    except (GpuHealthError, NoGpuError) as error:
        logger.error("GPU health check failed: %s", error)
        return 2
    logger.info("GPU health check passed: %s", report.summary())
    return 0
