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
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO / "src" / "euroeval"
OUTPUT = REPO / "src" / "frontend" / "md" / "api.md"


def is_public(name: str) -> bool:
    """Check whether a name is part of the public API.

    Args:
        name: The identifier to test.

    Returns:
        True iff the name does not start with an underscore.
    """
    return not name.startswith("_")


def rel_to_module(rel: Path) -> str:
    """Convert a relative path to a dotted module path.

    Args:
        rel: Path relative to the package root, e.g. `metrics/__init__.py`.

    Returns:
        The dotted module name, e.g. `euroeval.metrics`.
    """
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(["euroeval", *parts]) if parts else "euroeval"


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
    heading: str,
    prefix: str = "",
) -> None:
    """Append a function or method entry to the output buffer.

    Args:
        node: The function/method AST node.
        out: The output buffer (mutated in place).
        heading: Markdown heading prefix (e.g. `###` for functions, `####`
            for methods).
        prefix: Optional symbol prefix prepended before the function name,
            typically `ClassName.` for methods.
    """
    sig = render_signature(node)
    out.append(f"{heading} `{prefix}{node.name}{sig}`")
    out.append("")
    doc = ast.get_docstring(node, clean=True)
    if doc:
        out.append(indent_docstring(doc))
        out.append("")


def render_class(cls: ast.ClassDef, out: list[str]) -> None:
    """Append a class entry — including any public methods — to the output.

    Args:
        cls: The class AST node.
        out: The output buffer (mutated in place).
    """
    bases = ", ".join(ast.unparse(b) for b in cls.bases)
    header = f"class {cls.name}" + (f"({bases})" if bases else "")
    out.append(f"### `{header}`")
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
        render_function(m, out, heading="####", prefix=f"{cls.name}.")


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
        f"<code>{mod_name}</code></h2></summary>"
    )
    out.append("")
    if module_doc:
        out.append(indent_docstring(module_doc))
        out.append("")
    for cls in classes:
        render_class(cls, out)
    for f in funcs:
        render_function(f, out, heading="###")
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

    out.append(f"#### <code>{rel_to_module(rel)}</code>")
    out.append("")
    if module_doc:
        out.append(indent_docstring(module_doc))
        out.append("")
    for cls in classes:
        # Bump class headings down a level so they remain below the submodule
        # heading and out of the TOC.
        render_class_inline(cls, out)
    for f in funcs:
        render_function(f, out, heading="#####")


def render_class_inline(cls: ast.ClassDef, out: list[str]) -> None:
    """Like `render_class` but with h5/h6 headings to stay out of the TOC.

    Args:
        cls: The class AST node.
        out: The output buffer (mutated in place).
    """
    bases = ", ".join(ast.unparse(b) for b in cls.bases)
    header = f"class {cls.name}" + (f"({bases})" if bases else "")
    out.append(f"##### `{header}`")
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
        render_function(m, out, heading="######", prefix=f"{cls.name}.")


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
        f"<code>{mod_name}</code></h2></summary>"
    )
    out.append("")
    if module_doc:
        out.append(indent_docstring(module_doc))
        out.append("")
    for cls in classes:
        render_class(cls, out)
    for f in funcs:
        render_function(f, out, heading="###")
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
        "public module along with its classes, functions, and signatures.",
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
