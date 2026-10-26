"""Refresh pinned Hugging Face revisions in built-in dataset configurations."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from huggingface_hub import HfApi

CONFIG_DIR = Path(__file__).parents[1] / "euroeval" / "dataset_configs"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def main() -> None:
    """Resolve and pin datasets after validating every Hub source.

    Raises:
        SystemExit: If any source cannot be resolved or a config cannot be updated.
    """
    api = HfApi()
    euroeval_datasets = {
        dataset.id: dataset.sha
        for dataset in api.list_datasets(author="EuroEval", expand=["sha"])
    }
    files = sorted(CONFIG_DIR.glob("*.py"))
    parsed: list[tuple[Path, str, list[tuple[int, int, str]]]] = []
    sources: set[str] = set()
    for path in files:
        text = path.read_text()
        tree = ast.parse(text)
        edits = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in {"DatasetConfig", "TranslationDatasetConfig"}:
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
                repo = source_node.value.split("::", maxsplit=1)[0]
                sources.add(repo)
                if node.end_lineno is None or node.end_col_offset is None:
                    raise SystemExit(f"Missing source location in {path}")
                edits.append((node.end_lineno, node.end_col_offset, repo))
        parsed.append((path, text, edits))

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

    updates = 0
    for path, text, edits in parsed:
        lines = text.splitlines(keepends=True)
        for line_number, column, source in sorted(edits, reverse=True):
            line_index = line_number - 1
            line = lines[line_index]
            # The AST end column points just after the closing parenthesis.
            close = column - 1
            if line[close : close + 1] != ")":
                raise SystemExit(f"Unexpected config syntax at {path}:{line_number}")
            prefix = line[:close]
            separator = "" if prefix.rstrip().endswith(",") else ","
            lines[line_index] = (
                prefix + separator + f' revision="{revisions[source]}"' + line[close:]
            )
            updates += 1
        updated = "".join(lines)
        ast.parse(updated)
        if updated != text:
            path.write_text(updated)
    print(
        f"Pinned {updates} built-in dataset configs across {len(sources)} Hub sources."
    )


if __name__ == "__main__":
    main()
