#!/usr/bin/env python3
"""Generate the API reference markdown for the EuroEval Python package.

Walks `src/euroeval` recursively, extracts module docstrings plus the public
classes and functions with their signatures and docstrings, and writes a
single markdown file at `src/frontend/md/api.md` that the Vue docs site
renders via the existing markdown pipeline.

Run directly (`python3 src/scripts/build_api_reference.py`) or via the
sibling Vite plugin, which calls this on dev-server start, on build, and
whenever a Python source file changes.
"""

from __future__ import annotations

import ast
import html
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO / "src" / "euroeval"
OUTPUT = REPO / "src" / "frontend" / "md" / "api.md"
GITHUB_SRC = "https://github.com/EuroEval/EuroEval/blob/main/src/euroeval"


def gh_url(rel: Path, lineno: int | None = None) -> str:
    """Build a GitHub source URL for a file (optionally pointing at a line).

    Args:
        rel: Path relative to the package root.
        lineno: 1-based line number, or None to link to the file itself.

    Returns:
        An absolute URL on github.com.
    """
    base = f"{GITHUB_SRC}/{rel.as_posix()}"
    return f"{base}#L{lineno}" if lineno else base


def source_link(url: str) -> str:
    """Render a small "source" anchor pointing at a GitHub URL.

    Args:
        url: The GitHub URL to link to.

    Returns:
        An HTML `<a>` snippet ready to be appended to a heading.
    """
    return (
        f' <a class="api-source-link" href="{url}" '
        f'target="_blank" rel="noopener">source</a>'
    )


def heading_html(level: int, anchor: str, code: str, url: str) -> str:
    """Emit a raw HTML heading with a `<code>` label and a GitHub source link.

    We bypass markdown's heading syntax so we can attach the source link
    without polluting the auto-generated heading slug — and so the TOC
    extraction can recognise the source-link span and skip it.

    Args:
        level: Heading level (2–6).
        anchor: The HTML id for the heading.
        code: Label text rendered inside `<code>` (e.g. a function signature).
        url: GitHub URL appended as a `source` link.

    Returns:
        A single-line `<hN>…</hN>` snippet for the markdown buffer.
    """
    return (
        f'<h{level} id="{anchor}" class="api-symbol">'
        f"<code>{html.escape(code, quote=False)}</code>"
        f"{source_link(url)}</h{level}>"
    )


def symbol_anchor(rel: Path, *parts: str) -> str:
    """Build a stable HTML id for a symbol declared in a module.

    Args:
        rel: Path relative to the package root.
        *parts: Symbol path within the file (e.g. `Class`, `Class.method`).

    Returns:
        A unique anchor like `api-metrics-Foo-bar`.
    """
    mod = rel.with_suffix("").as_posix().replace("/", "-")
    if mod.endswith("-__init__"):
        mod = mod[: -len("-__init__")]
    suffix = "-" + "-".join(parts) if parts else ""
    return f"api-{mod}{suffix}"


def is_public(name: str) -> bool:
    """Check whether a name is part of the public API.

    Args:
        name: The identifier to test.

    Returns:
        True iff the name does not start with an underscore.
    """
    return not name.startswith("_")


def rel_to_module(rel: Path) -> str:
    """Convert a relative path to a dotted module label.

    The leading `euroeval.` prefix is dropped — every entry on the page is
    implicitly under the `euroeval` package, so repeating it just adds noise.

    Args:
        rel: Path relative to the package root, e.g. `metrics/__init__.py`.

    Returns:
        The dotted module label, e.g. `metrics` or `metrics.ifeval.compute`.
        Returns the empty string for the package root `__init__.py`.
    """
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def render_signature(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a function's argument list and return type as Python source.

    Args:
        func: The function or method AST node to render.

    Returns:
        The signature as a string starting with `(` — suitable to be appended
        directly after the function name in markdown output.
    """
    args = func.args
    parts: list[str] = []

    posonly = list(args.posonlyargs)
    positional = list(args.args)
    flat_pos = posonly + positional
    defaults = list(args.defaults)
    # Defaults align to the right edge of (posonly + positional).
    n_pos = len(flat_pos)
    pos_defaults: dict[int, ast.expr] = {
        idx: defaults[i] for i, idx in enumerate(range(n_pos - len(defaults), n_pos))
    }

    for idx, a in enumerate(flat_pos):
        s = a.arg
        if a.annotation is not None:
            s += f": {ast.unparse(a.annotation)}"
        if idx in pos_defaults:
            s += f" = {ast.unparse(pos_defaults[idx])}"
        parts.append(s)
        if posonly and idx == len(posonly) - 1:
            parts.append("/")

    if args.vararg is not None:
        v = f"*{args.vararg.arg}"
        if args.vararg.annotation is not None:
            v += f": {ast.unparse(args.vararg.annotation)}"
        parts.append(v)
    elif args.kwonlyargs:
        parts.append("*")

    for kw, kd in zip(args.kwonlyargs, args.kw_defaults):
        s = kw.arg
        if kw.annotation is not None:
            s += f": {ast.unparse(kw.annotation)}"
        if kd is not None:
            s += f" = {ast.unparse(kd)}"
        parts.append(s)

    if args.kwarg is not None:
        k = f"**{args.kwarg.arg}"
        if args.kwarg.annotation is not None:
            k += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append(k)

    sig = "(" + ", ".join(parts) + ")"
    if func.returns is not None:
        sig += f" -> {ast.unparse(func.returns)}"
    return sig


def indent_docstring(doc: str) -> str:
    """Normalise a docstring for inclusion in markdown.

    The Google-style sections (`Args:`, `Returns:`, …) are kept as-is so they
    appear as normal paragraphs — good enough for an at-a-glance reference.

    Args:
        doc: The raw docstring text returned by `ast.get_docstring`.

    Returns:
        The trimmed docstring, ready to be appended to the output buffer.
    """
    return doc.strip()


def render_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    out: list[str],
    *,
    level: int,
    rel: Path,
    parent_class: str | None = None,
) -> None:
    """Append a function or method entry to the output buffer.

    Args:
        node: The function/method AST node.
        out: The output buffer (mutated in place).
        level: Heading level (2–6).
        rel: Path relative to the package root, used to build the GitHub link.
        parent_class: Optional enclosing class name; used both as a display
            prefix (`Class.method(…)`) and in the anchor id.
    """
    sig = render_signature(node)
    label = f"{parent_class}.{node.name}{sig}" if parent_class else f"{node.name}{sig}"
    parts = [parent_class, node.name] if parent_class else [node.name]
    anchor = symbol_anchor(rel, *parts)
    out.append(heading_html(level, anchor, label, gh_url(rel, node.lineno)))
    out.append("")
    doc = ast.get_docstring(node, clean=True)
    if doc:
        out.append(indent_docstring(doc))
        out.append("")


def render_class(
    cls: ast.ClassDef, out: list[str], *, rel: Path, level: int = 3
) -> None:
    """Append a class entry — including any public methods — to the output.

    Args:
        cls: The class AST node.
        out: The output buffer (mutated in place).
        rel: Path relative to the package root.
        level: Heading level for the class entry (methods go one level deeper).
    """
    bases = ", ".join(ast.unparse(b) for b in cls.bases)
    header = f"class {cls.name}" + (f"({bases})" if bases else "")
    anchor = symbol_anchor(rel, cls.name)
    out.append(heading_html(level, anchor, header, gh_url(rel, cls.lineno)))
    out.append("")
    doc = ast.get_docstring(cls, clean=True)
    if doc:
        out.append(indent_docstring(doc))
        out.append("")
    methods = [
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (is_public(n.name) or n.name == "__init__")
    ]
    for m in methods:
        render_function(m, out, level=min(level + 1, 6), rel=rel, parent_class=cls.name)


def render_module(rel: Path, source: str, out: list[str]) -> None:
    """Append a module section (heading, docstring, classes, functions) to out.

    Args:
        rel: Path relative to the package root.
        source: The source code of the module.
        out: The output buffer (mutated in place).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        out.append(f"## `{rel_to_module(rel)}`")
        out.append("")
        out.append(f"_failed to parse: {e}_")
        out.append("")
        return

    classes = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and is_public(n.name)
    ]
    funcs = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(n.name)
    ]
    module_doc = ast.get_docstring(tree, clean=True)

    # Skip modules with no public API and no module-level docstring — they're
    # almost always internal glue.
    if not classes and not funcs and not module_doc:
        return

    mod_name = rel_to_module(rel)
    anchor = "api-" + mod_name.replace(".", "-")
    out.append(f'<details class="api-module" id="{anchor}-wrap">')
    out.append(
        f'<summary class="api-module-summary">'
        f'<h2 id="{anchor}" class="api-module-heading">'
        f"<code>{mod_name}</code>{source_link(gh_url(rel))}</h2></summary>"
    )
    out.append("")
    if module_doc:
        out.append(indent_docstring(module_doc))
        out.append("")
    for cls in classes:
        render_class(cls, out, rel=rel, level=3)
    for f in funcs:
        render_function(f, out, level=3, rel=rel)
    out.append("</details>")
    out.append("")


def render_submodule_inline(rel: Path, source: str, out: list[str]) -> None:
    """Render a submodule as an h4 entry inside its parent's collapsible.

    Only the module docstring and any classes / functions are emitted; the
    heading is an h4 so it doesn't appear in the right-hand table of
    contents.

    Args:
        rel: Path relative to the package root.
        source: The source code of the module.
        out: The output buffer (mutated in place).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        out.append(f"#### `{rel_to_module(rel)}`")
        out.append("")
        out.append(f"_failed to parse: {e}_")
        out.append("")
        return

    classes = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and is_public(n.name)
    ]
    funcs = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(n.name)
    ]
    module_doc = ast.get_docstring(tree, clean=True)
    if not classes and not funcs and not module_doc:
        return

    mod_name = rel_to_module(rel)
    anchor = "api-" + mod_name.replace(".", "-")
    out.append(f'<details class="api-submodule" id="{anchor}-wrap">')
    out.append(
        f'<summary class="api-submodule-summary">'
        f"<code>{mod_name}</code>{source_link(gh_url(rel))}</summary>"
    )
    out.append("")
    if module_doc:
        out.append(indent_docstring(module_doc))
        out.append("")
    for cls in classes:
        # Bump class headings down a level so they remain below the submodule
        # heading and out of the TOC.
        render_class(cls, out, rel=rel, level=5)
    for f in funcs:
        render_function(f, out, level=5, rel=rel)
    out.append("</details>")
    out.append("")


def render_nested_subpackage(
    parent_rel: Path, parent_source: str, submodule_files: list[Path], out: list[str]
) -> None:
    """Render a subpackage with its submodules folded inside.

    Args:
        parent_rel: Path of the subpackage's `__init__.py` relative to the
            package root.
        parent_source: Source code of the `__init__.py`.
        submodule_files: Absolute paths of submodule `.py` files to nest
            inside the subpackage's collapsible.
        out: The output buffer (mutated in place).
    """
    try:
        tree = ast.parse(parent_source)
    except SyntaxError:
        tree = ast.parse("")
    classes = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and is_public(n.name)
    ]
    funcs = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(n.name)
    ]
    module_doc = ast.get_docstring(tree, clean=True)

    mod_name = rel_to_module(parent_rel)
    anchor = "api-" + mod_name.replace(".", "-")
    out.append(f'<details class="api-module" id="{anchor}-wrap">')
    out.append(
        f'<summary class="api-module-summary">'
        f'<h2 id="{anchor}" class="api-module-heading">'
        f"<code>{mod_name}</code>{source_link(gh_url(parent_rel))}</h2></summary>"
    )
    out.append("")
    if module_doc:
        out.append(indent_docstring(module_doc))
        out.append("")
    for cls in classes:
        render_class(cls, out, rel=parent_rel, level=3)
    for f in funcs:
        render_function(f, out, level=3, rel=parent_rel)
    for sub in submodule_files:
        rel = sub.relative_to(PKG_ROOT)
        render_submodule_inline(rel, sub.read_text(encoding="utf-8"), out)
    out.append("</details>")
    out.append("")


def should_skip(rel: Path) -> bool:
    """Decide whether a Python file should be excluded from the API reference.

    Private modules and packages (any path segment starting with `_`, except
    `__init__.py` itself) are skipped, as are `__pycache__` artifacts.

    Args:
        rel: Path relative to the package root.

    Returns:
        True iff the file should be skipped.
    """
    for part in rel.parts[:-1]:
        if part.startswith("_") and part != "__pycache__":
            return True
        if part == "__pycache__":
            return True
    name = rel.name
    if name == "__init__.py":
        return False
    return name.startswith("_")


def main() -> None:
    """Walk the EuroEval package and write the API reference markdown file."""
    files = sorted(PKG_ROOT.rglob("*.py"))
    out: list[str] = [
        "---",
        "hide:",
        "    - navigation",
        "---",
        "# API Reference",
        "",
        "Auto-generated from the EuroEval Python package source. Lists every "
        "public module along with its classes, functions, and signatures. "
        "All entries are under the `euroeval` package — the prefix is implied "
        "and omitted for brevity.",
        "",
        "_Regenerated on every dev-server start, build, and Python source "
        "change — so this page is always in sync with the installed package._",
        "",
    ]
    rendered: set[Path] = set()
    for f in files:
        rel = f.relative_to(PKG_ROOT)
        if should_skip(rel):
            continue
        # Skip the root package `__init__.py` — every entry on the page is
        # already implicitly under `euroeval`, no need for a bare top-level
        # entry.
        if rel == Path("__init__.py"):
            continue
        if f in rendered:
            continue
        # Subpackage handling: render the `__init__.py` together with all of
        # its descendant `.py` files in a single collapsible so the TOC stays
        # focused on top-level modules.
        if len(rel.parts) == 2 and rel.name == "__init__.py":
            children = sorted(
                g
                for g in files
                if g != f
                and not should_skip(g.relative_to(PKG_ROOT))
                and g.relative_to(PKG_ROOT).parts[0] == rel.parts[0]
            )
            render_nested_subpackage(rel, f.read_text(encoding="utf-8"), children, out)
            rendered.add(f)
            rendered.update(children)
            continue
        # Skip submodules of any subpackage — they're rendered inside their
        # parent above.
        if len(rel.parts) > 1:
            continue
        render_module(rel, f.read_text(encoding="utf-8"), out)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    n_lines = len(out)
    print(f"[api-reference] wrote {OUTPUT.relative_to(REPO)} ({n_lines} lines)")


if __name__ == "__main__":
    main()
