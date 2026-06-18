"""Process priority and GPU thermal management for the queue processor.

The queue processor runs unattended on a shared box. Two small concerns
keep it a polite tenant:

* Lower its niceness so heavy euroeval subprocesses don't crowd out sshd.
* Pause between issues whenever ``nvidia-smi`` reports the GPU near its
  thermal limit, resuming once it has cooled.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Set to False once we've logged that nvidia-smi isn't available, so we
# don't spam the log every cooldown.
_nvidia_smi_available: bool = True


@dataclass
class ThermalConfig:
    """Knobs controlling the inter-issue thermal cooldown.

    Attributes:
        inter_issue_sleep_seconds:
            Minimum time to wait between issues, regardless of GPU temp.
        pause_temp_c:
            GPU temperature (in degrees C) at or above which to enter a
            cooldown loop.
        resume_temp_c:
            GPU temperature (in degrees C) the GPU must cool to before
            resuming.
        poll_interval_seconds:
            How often to re-check the GPU temperature during cooldown.
    """

    inter_issue_sleep_seconds: float = 30.0
    pause_temp_c: float = 80.0
    resume_temp_c: float = 70.0
    poll_interval_seconds: float = 15.0


def lower_process_priority() -> None:
    """Nice the process so SSH and other interactive work stay responsive.

    A niceness bump of 10 keeps the heavy euroeval subprocess out of the
    way of sshd, the shell, and similar low-CPU interactive workloads
    without starving the evaluation itself when the box is otherwise idle.
    """
    try:
        os.nice(10)
    except OSError as e:
        logger.warning(f"Could not lower process priority: {e}")


def cool_down_between_issues(config: ThermalConfig) -> None:
    """Pause between issues, extending the pause until the GPU has cooled.

    Always waits at least ``config.inter_issue_sleep_seconds``. If
    ``nvidia-smi`` reports a GPU at or above ``config.pause_temp_c``,
    keeps sleeping in ``config.poll_interval_seconds`` chunks until the
    hottest GPU has dropped below ``config.resume_temp_c``.

    Args:
        config:
            The thermal configuration controlling the pause behaviour.
    """
    time.sleep(config.inter_issue_sleep_seconds)
    wait_for_gpu_to_cool(config=config)


def wait_for_gpu_to_cool(config: ThermalConfig) -> None:
    """Block until the GPU is below ``resume_temp_c``; return immediately if cool.

    Unlike :func:`cool_down_between_issues`, this does not impose a
    minimum sleep -- if the GPU is already cool (or ``nvidia-smi`` is
    unavailable), it returns without sleeping at all.

    Args:
        config:
            The thermal configuration controlling the pause behaviour.
    """
    temp = read_gpu_temperature_c()
    if temp is None or temp < config.pause_temp_c:
        return
    logger.info(
        f"GPU at {temp:.0f}C (>= {config.pause_temp_c:.0f}C); pausing until "
        f"it cools below {config.resume_temp_c:.0f}C."
    )
    while True:
        time.sleep(config.poll_interval_seconds)
        temp = read_gpu_temperature_c()
        if temp is None:
            return
        if temp < config.resume_temp_c:
            logger.info(f"GPU cooled to {temp:.0f}C; resuming.")
            return


def read_gpu_temperature_c() -> float | None:
    """Return the hottest GPU temperature reported by ``nvidia-smi``, in C.

    Returns:
        The hottest temperature in degrees Celsius, or None if
        ``nvidia-smi`` is missing or fails (callers treat that as
        "thermal check unavailable" and skip the cooldown).
    """
    global _nvidia_smi_available
    if not _nvidia_smi_available:
        return None
    try:
        # Safe: a fixed argument list, no shell and no user-supplied input.
        result = subprocess.run(  # noqa: S603, S607
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.info(f"nvidia-smi unavailable for thermal check ({e}); disabling.")
        _nvidia_smi_available = False
        return None
    if result.returncode != 0:
        return None
    temps: list[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            temps.append(float(line))
        except ValueError:
            continue
    return max(temps) if temps else None
