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
    """Resolve and pin datasets after validating every Hub source.

    Raises:
        SystemExit: If a source cannot be resolved or a config cannot be updated.
    """
    files = sorted(CONFIG_DIR.glob("*.py"))
    parsed: list[tuple[Path, str, ast.Module]] = []
    sources: set[str] = set()
    for path in files:
        text = path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            raise SystemExit(f"Invalid config syntax in {path}: {exc}") from exc
        parsed.append((path, text, tree))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in CONFIG_CALLS:
                continue
            source_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "source"),
                None,
            )
            if (
                isinstance(source_node, ast.Constant)
                and isinstance(source_node.value, str)
                and source_node.value
            ):
                sources.add(source_node.value.split("::", maxsplit=1)[0])

    api = HfApi()
    euroeval_datasets = {
        dataset.id: dataset.sha
        for dataset in api.list_datasets(author="EuroEval", expand=["sha"])
    }
    revisions: dict[str, str] = {}
    missing = []
    for source in sorted(sources):
        if source in euroeval_datasets:
            sha = euroeval_datasets[source]
        else:
            try:
                sha = api.dataset_info(source, files_metadata=False).sha
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

    generated: list[tuple[Path, str, str]] = []
    for path, text, tree in parsed:
        edits: list[tuple[int, int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in CONFIG_CALLS:
                continue
            source_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "source"),
                None,
            )
            if (
                not isinstance(source_node, ast.Constant)
                or not isinstance(source_node.value, str)
                or not source_node.value
            ):
                continue
            repo = source_node.value.split("::", maxsplit=1)[0]
            revision = revisions[repo]
            existing = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "revision"
                ),
                None,
            )
            if existing is not None:
                if not isinstance(existing, ast.Constant) or not isinstance(
                    existing.value, str
                ):
                    raise SystemExit(f"Non-literal revision in {path}")
                if existing.value == revision:
                    continue
                start = _offset(
                    text=text, line=existing.lineno, column=existing.col_offset
                )
                end = _offset(
                    text=text, line=existing.end_lineno, column=existing.end_col_offset
                )
                edits.append((start, end, repr(revision)))
            else:
                if node.end_lineno is None or node.end_col_offset is None:
                    raise SystemExit(f"Missing source location in {path}")
                close = (
                    _offset(text=text, line=node.end_lineno, column=node.end_col_offset)
                    - 1
                )
                if text[close : close + 1] != ")":
                    raise SystemExit(
                        f"Unexpected config syntax at {path}:{node.end_lineno}"
                    )
                before = text[:close].rstrip()
                separator = "" if before.endswith(",") else ","
                edits.append((close, close, f'{separator} revision="{revision}"'))

        updated = text
        for start, end, replacement in sorted(edits, reverse=True):
            updated = updated[:start] + replacement + updated[end:]
        try:
            ast.parse(updated)
        except SyntaxError as exc:
            raise SystemExit(f"Generated invalid config in {path}: {exc}") from exc
        generated.append((path, text, updated))

    updates = sum(original != updated for _, original, updated in generated)
    for path, original, updated in generated:
        if original != updated:
            path.write_text(updated)
    print(
        f"Pinned {updates} built-in dataset configs across {len(sources)} Hub sources."
    )


def _offset(*, text: str, line: int, column: int) -> int:
    """Convert an AST UTF-8 byte location into a string offset."""
    lines = text.splitlines(keepends=True)
    return sum(len(value) for value in lines[: line - 1]) + len(
        lines[line - 1].encode()[:column].decode()
    )


if __name__ == "__main__":
    main()
