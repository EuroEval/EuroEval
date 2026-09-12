"""Command-line entry point for ``euroeval-worker``."""

import argparse
import logging
from pathlib import Path

from . import __version__

DEFAULT_SERVER = "https://euroeval.com/api/worker"
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the volunteer worker CLI.

    Returns:
        Process exit status.
    """
    parser = _parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if arguments.gpu_health_check:
        return _run_gpu_health_check()
    from .broker import BrokerClient  # noqa: PLC0415
    from .hardware import NoGpuError  # noqa: PLC0415
    from .runtime import Worker  # noqa: PLC0415
    from .state import StateStore, default_state_dir  # noqa: PLC0415

    try:
        state_dir = (
            Path(arguments.state_dir) if arguments.state_dir else default_state_dir()
        )
        Worker(
            client=BrokerClient(server=arguments.server),
            state=StateStore(directory=state_dir),
            gpu_memory_utilisation=arguments.gpu_memory_utilisation,
        ).run(once=arguments.once)
    except KeyboardInterrupt:
        logger.info("Worker stopped")
    except NoGpuError as error:
        logger.error("%s", error)
        return 2
    except Exception:
        logger.exception("Worker stopped after an error")
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a EuroEval volunteer GPU worker.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Worker broker URL.")
    parser.add_argument(
        "--state-dir", type=Path, help="Directory for private worker state."
    )
    parser.add_argument(
        "--once", action="store_true", help="Claim at most one piece of work."
    )
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
