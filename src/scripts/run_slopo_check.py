"""Run Slopo stages safely and reject failed or duplicate analyses."""

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ANALYSIS_PATTERN = re.compile(
    r"^Similarity ratio \(including exact copies\):\s*"
    r"([0-9]+(?:\.[0-9]+)?%) \(([0-9]+)\/([0-9]+) units flagged as similar\)$",
    re.MULTILINE,
)


def main() -> int:
    """Run the Slopo check from the current repository.

    Returns:
        The process exit status.
    """
    return run_slopo_check()


def run_slopo_check(root: Path | None = None) -> int:
    """Run Slopo index, embed, and analyze without touching shared caches.

    Args:
        root (optional): Repository root. Defaults to the current working directory.

    Returns:
        Zero for a successful duplicate-free analysis, otherwise one.

    Raises:
        OSError:
            If an isolated cache path is a symlink or cannot be written.
    """
    repository = (root or Path.cwd()).resolve()
    config_path = repository / "slopo.conf.yaml"

    try:
        config = yaml.safe_load(config_path.read_text())
        config["source_dir"] = str((repository / config["source_dir"]).resolve())
        if config.get("ignore_file"):
            config["ignore_file"] = str((repository / config["ignore_file"]).resolve())
        work_dir = _cache_directory(repository=repository)
        if work_dir.is_symlink():
            raise OSError(f"Refusing to use linked Slopo cache directory: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)
        isolated_config = work_dir / "slopo.conf.yaml"
        if isolated_config.is_symlink():
            raise OSError(
                f"Refusing to overwrite linked Slopo config: {isolated_config}"
            )
        config["db_file"] = str(work_dir / "slopo.db")
        config["report_dir"] = str(work_dir / "report")
        for path in (work_dir / "slopo.db", work_dir / "report"):
            if path.is_symlink():
                raise OSError(f"Refusing to use linked Slopo cache path: {path}")
        isolated_config.write_text(yaml.safe_dump(config))
        environment = os.environ.copy()
        environment["LITELLM_DROP_PARAMS"] = "true"

        for stage in ("index", "embed"):
            if (
                _run_stage(
                    stage=stage,
                    config=isolated_config,
                    repository=repository,
                    environment=environment,
                )
                is None
            ):
                print(f"Slopo report: {config['report_dir']}")
                return 1

        analysis = _run_stage(
            stage="analyze",
            config=isolated_config,
            repository=repository,
            environment=environment,
        )
        print(f"Slopo report: {config['report_dir']}")
        if analysis is None:
            return 1
        match = ANALYSIS_PATTERN.search(analysis)
        if match is None:
            if _reports_no_similar_code(analysis):
                print("Slopo passed: no similar code units detected.")
                return 0
            print("Slopo analyze failed: missing or malformed summary", file=sys.stderr)
            return 1
        if len(ANALYSIS_PATTERN.findall(analysis)) != 1:
            print("Slopo analyze failed: ambiguous summary", file=sys.stderr)
            return 1
        duplicates = int(match.group(2))
        total = int(match.group(3))
        if duplicates > total:
            print("Slopo analyze failed: invalid duplicate count", file=sys.stderr)
            return 1
        if duplicates:
            print("Slopo failed: duplicate code detected", file=sys.stderr)
            print(f"Duplicate units: {duplicates}/{total} ({match.group(1)})")
            print(
                "Review "
                f"{config.get('ignore_file', 'Slopo ignore file')} "
                "for reviewed exceptions."
            )
            return 1
        print(f"Slopo passed: {duplicates}/{total} duplicate units ({match.group(1)}).")
        return 0
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"Slopo setup failed: {error}", file=sys.stderr)
        return 1


def _cache_directory(*, repository: Path) -> Path:
    """Return a stable cache directory isolated to this repository checkout.

    Args:
        repository: Resolved repository root.

    Returns:
        A worktree-local Slopo cache directory that does not traverse a linked
        ``.slopo`` directory.
    """
    digest = hashlib.sha256(str(repository).encode()).hexdigest()[:12]
    slopo_directory = repository / ".slopo"
    if slopo_directory.is_symlink():
        return repository / f".slopo-check-{digest}"
    return slopo_directory / f"check-{digest}"


def _reports_no_similar_code(output: str) -> bool:
    """Return whether Slopo reported a clean-analysis outcome.

    Args:
        output: Successful analyze output.

    Returns:
        Whether the output explicitly confirms no unignored similar code remains.
    """
    clean_outcomes = {
        "No similar pairs found.",
        "No similar code units found for configured similarity and rerank thresholds.",
        "All similar code clusters are in the ignore list.",
    }
    return any(line.strip() in clean_outcomes for line in output.splitlines())


def _run_stage(
    *, stage: str, config: Path, repository: Path, environment: dict[str, str]
) -> str | None:
    """Run one Slopo stage and return its output or None on failure.

    Args:
        stage: Slopo command name.
        config: Isolated configuration file.
        repository: Repository root used as the process working directory.
        environment: Process environment for Slopo.

    Returns:
        Combined output on success, otherwise None.
    """
    try:
        result = subprocess.run(
            ["uv", "run", "slopo", "--config", str(config), stage],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        print(f"Slopo {stage} failed to start: {error}", file=sys.stderr)
        return None
    output = result.stdout + result.stderr
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if result.returncode:
        print(
            f"Slopo {stage} failed with exit status {result.returncode}.",
            file=sys.stderr,
        )
        return None
    return output


if __name__ == "__main__":
    raise SystemExit(main())
