"""Command-line entry point for ``euroeval-worker``."""

import argparse
import logging
from pathlib import Path

from . import __version__
from .broker import BrokerClient
from .hardware import NoGpuError
from .runtime import Worker
from .state import StateStore, default_state_dir

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
    parser.add_argument("--version", action="version", version=__version__)
    return parser
