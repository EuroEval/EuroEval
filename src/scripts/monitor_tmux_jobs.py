# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1.7",
#   "python-dotenv>=1.1.2",
# ]
# ///

"""Monitor active tmux jobs on remote servers and notify on completion.

This script is intentionally local-only and gitignored. It reads tmux pane state over
SSH, never attaches to sessions, and never sends keys or mutates remote state.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import click
from dotenv import load_dotenv

REMOTE_STATUS_SCRIPT = r"""
while IFS='|' read -r sess target dead pid cmd; do
  [ -z "$sess" ] && continue
  queue="$pid"; desc=0
  if [ "$dead" != "1" ]; then
    while [ -n "$queue" ]; do
      next=""
      for p in $queue; do
        kids=$(pgrep -P "$p" 2>/dev/null || true)
        for k in $kids; do desc=$((desc+1)); next="$next $k"; done
      done
      queue="$next"
    done
  fi
  echo "PANE|$sess|$target|$dead|$pid|$cmd|$desc"
done <<EOF
$(
format='#{session_name}|#{session_name}:#{window_index}.#{pane_index}'
format="$format"'|#{pane_dead}|#{pane_pid}|#{pane_current_command}'
tmux list-panes -a -F "$format" 2>/dev/null
)
EOF
"""

REMOTE_TAIL_SCRIPT = r"""
sess=$1
echo "TAIL_BEGIN"
tmux capture-pane -t "$sess" -p 2>/dev/null | tail -"$2" || true
echo "TAIL_END"
"""

load_dotenv()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--server",
    "servers",
    multiple=True,
    required=True,
    help=(
        "SSH alias/host to inspect, optionally with a tmux session as "
        "HOST::SESSION. Pass multiple times to monitor many targets."
    ),
)
@click.option(
    "--interval-seconds",
    default=300,
    show_default=True,
    type=click.IntRange(min=10),
    help="Seconds between checks.",
)
@click.option(
    "--output-regex",
    "output_regexes",
    multiple=True,
    help=(
        "Regex to search for in captured tmux output. If it matches, monitoring "
        "stops and sends an alert. Pass multiple times for multiple patterns."
    ),
)
@click.option(
    "--tail-lines",
    default=20,
    show_default=True,
    type=click.IntRange(min=1, max=500),
    help="Number of tmux pane lines to include in alerts and regex checks.",
)
@click.option(
    "--connect-timeout-seconds",
    default=15,
    show_default=True,
    type=click.IntRange(min=1),
    help="SSH connection timeout.",
)
@click.option(
    "--command-timeout-seconds",
    default=60,
    show_default=True,
    type=click.IntRange(min=5),
    help="Per-server SSH command timeout.",
)
@click.option(
    "--telegram-token-env",
    default="TELEGRAM_BOT_TOKEN",
    show_default=True,
    help="Environment variable containing the Telegram bot token.",
)
@click.option(
    "--telegram-chat-id-env",
    default="TELEGRAM_CHAT_ID",
    show_default=True,
    help="Environment variable containing the Telegram chat ID.",
)
@click.option(
    "--no-telegram",
    is_flag=True,
    help="Only write alerts to the shell; do not send Telegram messages.",
)
def main(
    servers: tuple[str, ...],
    interval_seconds: int,
    output_regexes: tuple[str, ...],
    tail_lines: int,
    connect_timeout_seconds: int,
    command_timeout_seconds: int,
    telegram_token_env: str,
    telegram_chat_id_env: str,
    no_telegram: bool,
) -> None:
    """Monitor remote tmux sessions on SSH servers."""
    patterns = _compile_patterns(patterns=output_regexes)
    target_specs = _parse_target_specs(values=servers)
    monitored = _discover_initial_sessions(
        target_specs=target_specs,
        connect_timeout_seconds=connect_timeout_seconds,
        command_timeout_seconds=command_timeout_seconds,
    )

    if not monitored:
        click.echo("No active tmux sessions found; exiting.")
        return

    click.echo(
        f"Monitor started at {dt.datetime.now().isoformat(timespec='seconds')}; "
        f"polling every {interval_seconds}s"
    )
    click.echo("Monitoring: " + ", ".join(_format_target(target=t) for t in monitored))

    while True:
        alert = _poll_once(
            target_specs=target_specs,
            monitored=monitored,
            patterns=patterns,
            tail_lines=tail_lines,
            connect_timeout_seconds=connect_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
        )
        if alert is not None:
            _emit_alert(
                alert=alert,
                no_telegram=no_telegram,
                telegram_token_env=telegram_token_env,
                telegram_chat_id_env=telegram_chat_id_env,
                command_timeout_seconds=command_timeout_seconds,
            )
            return
        time.sleep(interval_seconds)


def _compile_patterns(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    compiled_patterns: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled_patterns.append(re.compile(pattern=pattern, flags=re.MULTILINE))
        except re.error as error:
            raise click.BadParameter(
                message=f"Invalid --output-regex {pattern!r}: {error}",
                param_hint="--output-regex",
            ) from error
    return compiled_patterns


def _format_ssh_error(server: str, result: subprocess.CompletedProcess[str]) -> str:
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    return (
        f"{server}: ssh/tmux check failed with code {result.returncode}\n"
        f"STDERR:\n{stderr}\nSTDOUT:\n{stdout}"
    )


@dataclasses.dataclass(frozen=True)
class PaneStatus:
    """Current state of one tmux pane."""

    server: str
    session: str
    target: str
    dead: bool
    pid: str
    command: str
    descendants: int

    @property
    def is_active(self) -> bool:
        """Whether the pane appears to have a running child process."""
        return self.descendants > 0

    def summary(self) -> str:
        """Return a compact, shell-friendly summary."""
        dead = "1" if self.dead else "0"
        return (
            f"{self.server}:{self.target} dead={dead} cmd={self.command} "
            f"pid={self.pid} descendants={self.descendants}"
        )


def _discover_statuses(
    servers: tuple[str, ...], connect_timeout_seconds: int, command_timeout_seconds: int
) -> list[PaneStatus]:
    statuses: list[PaneStatus] = []
    errors: list[str] = []
    for server in servers:
        result = _ssh_run(
            server=server,
            command="bash -s",
            stdin=REMOTE_STATUS_SCRIPT,
            connect_timeout_seconds=connect_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
        )
        if result.returncode != 0:
            errors.append(_format_ssh_error(server=server, result=result))
            continue
        statuses.extend(_parse_statuses(server=server, output=result.stdout))
    if errors:
        raise RuntimeError("\n".join(errors))
    return statuses


def _parse_statuses(server: str, output: str) -> list[PaneStatus]:
    statuses: list[PaneStatus] = []
    for line in output.splitlines():
        parts = line.split("|", maxsplit=6)
        if len(parts) != 7 or parts[0] != "PANE":
            continue
        _, session, target, dead, pid, command, descendants = parts
        statuses.append(
            PaneStatus(
                server=server,
                session=session,
                target=target,
                dead=dead == "1",
                pid=pid,
                command=command,
                descendants=_parse_descendants(value=descendants),
            )
        )
    return statuses


def _parse_descendants(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def _ssh_run(
    server: str,
    command: str,
    stdin: str,
    connect_timeout_seconds: int,
    command_timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    ssh_command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout_seconds}",
        server,
        command,
    ]
    return subprocess.run(
        args=ssh_command,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=command_timeout_seconds,
        check=False,
    )


def _send_telegram_message(
    message: str, token_env: str, chat_id_env: str, timeout_seconds: int
) -> None:
    token = os.environ.get(token_env)
    chat_id = os.environ.get(chat_id_env)
    if not token or not chat_id:
        click.echo(
            f"Telegram not sent: set {token_env} and {chat_id_env}, "
            "or pass --no-telegram."
        )
        return

    body = urllib.parse.urlencode(
        query={
            "chat_id": chat_id,
            "text": message[:4096],
            "disable_web_page_preview": "true",
        }
    ).encode()
    request = urllib.request.Request(
        url=f"https://api.telegram.org/bot{token}/sendMessage", data=body, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response.read()
    except OSError as error:
        click.echo(f"Telegram send failed: {error}", err=True)


def _format_target(target: tuple[str, str]) -> str:
    server, session = target
    return f"{server}:{session}"


@dataclasses.dataclass(frozen=True, order=True)
class TargetSpec:
    """A server-wide or session-specific monitoring target."""

    server: str
    session: str | None = None

    @property
    def is_session_specific(self) -> bool:
        """Whether this target names a specific tmux session."""
        return self.session is not None

    def matches(self, status: "PaneStatus") -> bool:
        """Return whether a pane status belongs to this target."""
        return status.server == self.server and (
            self.session is None or status.session == self.session
        )

    def monitored_session(self) -> tuple[str, str] | None:
        """Return the exact session target, if one was configured."""
        if self.session is None:
            return None
        return (self.server, self.session)

    def summary(self) -> str:
        """Return a compact, shell-friendly target summary."""
        if self.session is None:
            return self.server
        return f"{self.server}::{self.session}"


def _discover_initial_sessions(
    target_specs: tuple[TargetSpec, ...],
    connect_timeout_seconds: int,
    command_timeout_seconds: int,
) -> set[tuple[str, str]]:
    statuses = _discover_statuses(
        servers=_target_servers(target_specs=target_specs),
        connect_timeout_seconds=connect_timeout_seconds,
        command_timeout_seconds=command_timeout_seconds,
    )
    active_sessions = _matching_active_sessions(
        statuses=statuses, target_specs=target_specs
    )
    configured_sessions = {
        session
        for spec in target_specs
        for session in [spec.monitored_session()]
        if session is not None
    }
    click.echo("Initial discovered panes:")
    for status in statuses:
        click.echo(f"  {status.summary()}")
    return active_sessions | configured_sessions


def _matching_active_sessions(
    statuses: list[PaneStatus], target_specs: tuple[TargetSpec, ...]
) -> set[tuple[str, str]]:
    return {
        (status.server, status.session)
        for status in statuses
        if status.is_active
        and any(spec.matches(status=status) for spec in target_specs)
    }


def _target_servers(target_specs: tuple[TargetSpec, ...]) -> tuple[str, ...]:
    servers = dict.fromkeys(spec.server for spec in target_specs)
    return tuple(servers)


def _parse_target_specs(values: tuple[str, ...]) -> tuple[TargetSpec, ...]:
    specs: list[TargetSpec] = []
    for value in values:
        if "::" not in value:
            server = value.strip()
            session = None
        else:
            server, session = (part.strip() for part in value.split("::", maxsplit=1))
        if not server:
            raise click.BadParameter(
                message=f"Invalid --server target {value!r}: missing server name",
                param_hint="--server",
            )
        if session == "":
            raise click.BadParameter(
                message=f"Invalid --server target {value!r}: missing session name",
                param_hint="--server",
            )
        specs.append(TargetSpec(server=server, session=session))
    return tuple(dict.fromkeys(specs))


@dataclasses.dataclass(frozen=True)
class Alert:
    """An event that should stop monitoring and notify the user."""

    kind: str
    target: str
    detail: str
    tail: str = ""

    def message(self) -> str:
        """Return the notification body."""
        timestamp = dt.datetime.now().isoformat(timespec="seconds")
        parts = [
            "EuroEval tmux monitor alert",
            f"time: {timestamp}",
            f"target: {self.target}",
            f"state: {self.kind}",
            "detail:",
            self.detail,
        ]
        if self.tail:
            parts.extend(["tail:", self.tail])
        return "\n".join(parts)


def _emit_alert(
    alert: Alert,
    no_telegram: bool,
    telegram_token_env: str,
    telegram_chat_id_env: str,
    command_timeout_seconds: int,
) -> None:
    message = alert.message()
    click.echo("\n=== MONITOR ALERT ===")
    click.echo(message)
    if no_telegram:
        return
    _send_telegram_message(
        message=message,
        token_env=telegram_token_env,
        chat_id_env=telegram_chat_id_env,
        timeout_seconds=command_timeout_seconds,
    )


def _poll_once(
    target_specs: tuple[TargetSpec, ...],
    monitored: set[tuple[str, str]],
    patterns: list[re.Pattern[str]],
    tail_lines: int,
    connect_timeout_seconds: int,
    command_timeout_seconds: int,
) -> Alert | None:
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    try:
        statuses = _discover_statuses(
            servers=_target_servers(target_specs=target_specs),
            connect_timeout_seconds=connect_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
        )
    except RuntimeError as error:
        return Alert(kind="ERROR", target="ssh/tmux discovery", detail=str(error))

    active = _matching_active_sessions(statuses=statuses, target_specs=target_specs)
    by_session = _group_by_session(statuses=statuses)

    finished = sorted(monitored - active)
    if finished:
        server, session = finished[0]
        tail = _capture_tail(
            server=server,
            session=session,
            tail_lines=tail_lines,
            connect_timeout_seconds=connect_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
        )
        detail_lines = [
            status.summary() for status in by_session.get((server, session), [])
        ] or ["session missing or no panes returned"]
        return Alert(
            kind="FINISHED",
            target=_format_target(target=(server, session)),
            detail="\n".join(detail_lines),
            tail=tail,
        )

    new_sessions = sorted(active - monitored)
    for target in new_sessions:
        monitored.add(target)
        click.echo(f"  NEW active session added: {_format_target(target=target)}")

    click.echo(f"[{timestamp}] discovered {len(active)} active sessions")
    for server, session in sorted(monitored):
        first_status = by_session.get((server, session), [None])[0]
        if first_status is not None:
            click.echo(
                f"  {_format_target(target=(server, session))} RUNNING; "
                f"{first_status.summary()}"
            )

    regex_alert = _check_output_regexes(
        monitored=monitored,
        patterns=patterns,
        tail_lines=tail_lines,
        connect_timeout_seconds=connect_timeout_seconds,
        command_timeout_seconds=command_timeout_seconds,
    )
    if regex_alert is not None:
        return regex_alert

    return None


def _capture_tail(
    server: str,
    session: str,
    tail_lines: int,
    connect_timeout_seconds: int,
    command_timeout_seconds: int,
) -> str:
    result = _ssh_run(
        server=server,
        command=f"bash -s -- {shlex.quote(session)} {tail_lines}",
        stdin=REMOTE_TAIL_SCRIPT,
        connect_timeout_seconds=connect_timeout_seconds,
        command_timeout_seconds=command_timeout_seconds,
    )
    tail = result.stdout.strip() or "<no output>"
    if result.stderr.strip():
        tail = f"{tail}\nSTDERR:\n{result.stderr.strip()}"
    return tail


def _check_output_regexes(
    monitored: set[tuple[str, str]],
    patterns: list[re.Pattern[str]],
    tail_lines: int,
    connect_timeout_seconds: int,
    command_timeout_seconds: int,
) -> Alert | None:
    if not patterns:
        return None

    for server, session in sorted(monitored):
        tail = _capture_tail(
            server=server,
            session=session,
            tail_lines=tail_lines,
            connect_timeout_seconds=connect_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
        )
        for pattern in patterns:
            match = pattern.search(string=tail)
            if match is not None:
                return Alert(
                    kind="OUTPUT_REGEX_MATCH",
                    target=_format_target(target=(server, session)),
                    detail=f"Matched --output-regex {pattern.pattern!r}",
                    tail=tail,
                )
    return None


def _group_by_session(
    statuses: list[PaneStatus],
) -> dict[tuple[str, str], list[PaneStatus]]:
    grouped: dict[tuple[str, str], list[PaneStatus]] = {}
    for status in statuses:
        grouped.setdefault((status.server, status.session), []).append(status)
    return grouped


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        click.echo("Interrupted; exiting.", err=True)
        sys.exit(130)
