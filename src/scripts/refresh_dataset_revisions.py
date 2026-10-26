"""Refresh pinned Hugging Face revisions in built-in dataset configurations."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from huggingface_hub import HfApi

CONFIG_DIR = Path(__file__).parents[1] / "euroeval" / "dataset_configs"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CONFIG_CALLS = {"DatasetConfig", "TranslationDatasetConfig"}


def main() -> None:
    """Resolve all Hub sources before updating any built-in configuration."""
    parsed, sources = _parse_configs()
    revisions = _resolve_revisions(sources=sources)
    generated = [
        (
            path,
            text,
            _render_config(path=path, text=text, tree=tree, revisions=revisions),
        )
        for path, text, tree in parsed
    ]
    updates = sum(original != updated for _, original, updated in generated)
    for path, original, updated in generated:
        if original != updated:
            path.write_text(updated)
    print(f"Pinned {updates} config files across {len(sources)} Hub sources.")


def _parse_configs() -> tuple[list[tuple[Path, str, ast.Module]], set[str]]:
    """Parse built-in configs and collect their Hub sources before network access.

    Returns:
        Parsed files and distinct Hub repository IDs.

    Raises:
        SystemExit:
            If a config cannot be parsed.
    """
    parsed: list[tuple[Path, str, ast.Module]] = []
    sources: set[str] = set()
    for path in sorted(CONFIG_DIR.glob("*.py")):
        text = path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            raise SystemExit(f"Invalid config syntax in {path}: {exc}") from exc
        parsed.append((path, text, tree))
        for node in ast.walk(tree):
            source = _get_source(node=node)
            if source is not None:
                sources.add(source)
    return parsed, sources


def _get_source(*, node: ast.AST) -> str | None:
    """Extract a literal Hub source from a built-in dataset config call.

    Args:
        node:
            An AST node in a dataset configuration module.

    Returns:
        Hub repository ID, or None for a non-config or local source.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id not in CONFIG_CALLS:
        return None
    source_node = next(
        (keyword.value for keyword in node.keywords if keyword.arg == "source"), None
    )
    if not isinstance(source_node, ast.Constant) or not isinstance(
        source_node.value, str
    ):
        return None
    return source_node.value.split("::", maxsplit=1)[0] or None


def _render_config(
    *, path: Path, text: str, tree: ast.Module, revisions: dict[str, str]
) -> str:
    """Prepare validated, idempotent revision edits for one config module.

    Args:
        path:
            Config file path used in errors.
        text:
            Original module source.
        tree:
            Parsed module syntax tree.
        revisions:
            Current Hub commit SHA for each source.

    Returns:
        Complete updated module source.

    Raises:
        SystemExit:
            If an existing revision is not literal or a generated edit is invalid.
    """
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        source = _get_source(node=node)
        if source is not None and isinstance(node, ast.Call):
            edit = _revision_edit(
                path=path, text=text, node=node, revision=revisions[source]
            )
            if edit is not None:
                edits.append(edit)
    updated = text
    for start, end, replacement in sorted(edits, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    try:
        ast.parse(updated)
    except SyntaxError as exc:
        raise SystemExit(f"Generated invalid config in {path}: {exc}") from exc
    return updated


def _revision_edit(
    *, path: Path, text: str, node: ast.Call, revision: str
) -> tuple[int, int, str] | None:
    """Find the edit that inserts or updates a literal revision.

    Args:
        path:
            Config file path used in errors.
        text:
            Original module source.
        node:
            Dataset config call to edit.
        revision:
            Hub commit SHA to pin.

    Returns:
        Replacement range and text, or None if already up to date.

    Raises:
        SystemExit:
            If the config cannot be edited safely.
    """
    existing = next(
        (keyword.value for keyword in node.keywords if keyword.arg == "revision"), None
    )
    if existing is not None:
        if not isinstance(existing, ast.Constant) or not isinstance(
            existing.value, str
        ):
            raise SystemExit(f"Non-literal revision in {path}")
        if existing.value == revision:
            return None
        if existing.end_lineno is None or existing.end_col_offset is None:
            raise SystemExit(f"Missing revision location in {path}")
        start = _offset(text=text, line=existing.lineno, column=existing.col_offset)
        end = _offset(
            text=text, line=existing.end_lineno, column=existing.end_col_offset
        )
        return start, end, repr(revision)
    if node.end_lineno is None or node.end_col_offset is None:
        raise SystemExit(f"Missing source location in {path}")
    close = _offset(text=text, line=node.end_lineno, column=node.end_col_offset) - 1
    if text[close : close + 1] != ")":
        raise SystemExit(f"Unexpected config syntax at {path}:{node.end_lineno}")
    separator = "" if text[:close].rstrip().endswith(",") else ","
    return close, close, f'{separator} revision="{revision}"'


def _offset(*, text: str, line: int, column: int) -> int:
    """Convert an AST UTF-8 byte location into a string offset.

    Args:
        text:
            Python source.
        line:
            One-based AST line number.
        column:
            UTF-8 byte offset within the line.

    Returns:
        Character offset within the source text.
    """
    lines = text.splitlines(keepends=True)
    return sum(len(value) for value in lines[: line - 1]) + len(
        lines[line - 1].encode()[:column].decode()
    )


def _resolve_revisions(*, sources: set[str]) -> dict[str, str]:
    """Fetch SHA pins and fail closed if any source is unavailable.

    Args:
        sources:
            Repository IDs to resolve.

    Returns:
        Mapping from repository ID to Hub commit SHA.

    Raises:
        SystemExit:
            If any repository is missing or has an invalid SHA.
    """
    api = HfApi()
    euroeval_datasets = {
        dataset.id: dataset.sha
        for dataset in api.list_datasets(author="EuroEval", expand=["sha"])
    }
    revisions: dict[str, str] = {}
    missing: list[str] = []
    for source in sorted(sources):
        try:
            sha = (
                euroeval_datasets[source]
                if source in euroeval_datasets
                else api.dataset_info(source, files_metadata=False).sha
            )
        except Exception as exc:
            missing.append(f"{source}: {type(exc).__name__}")
            continue
        if not isinstance(sha, str) or not SHA_PATTERN.fullmatch(sha):
            missing.append(f"{source}: invalid revision")
        else:
            revisions[source] = sha
    if missing:
        raise SystemExit(
            "Could not validate every configured dataset source:\n" + "\n".join(missing)
        )
    return revisions


if __name__ == "__main__":
    main()
