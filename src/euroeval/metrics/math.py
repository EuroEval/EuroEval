r"""Scoring for mathematical answers.

The candidate ladder prefers the last boxed answer, then a connected answer
marker, delimited mathematics, the whole short text, the last line, and the
last number. The first candidate wins; opaque prose may fall back to later
numeric candidates. Plain numbers use exact equality after percent values are
divided by 100, expressions are evaluated and compared as values, and anything
left over uses normalised case-folded text equality.

Evaluating an expression needs no LaTeX grammar, only a rewrite: the forms
benchmark answers use -- fractions, roots, powers, products, thousands
grouping, percentages, implicit products like ``2\pi`` -- are turned into plain
arithmetic, parsed with :mod:`ast` under limits that keep SymPy honest, and
compared with the rules Inspect AI uses. SymPy itself arrives through torch, so
this costs no dependency, whereas the ANTLR grammar behind Inspect's own LaTeX
parsing would. Anything the rewrite cannot render faithfully reports itself
unknown and falls back to text equality, so matrices, integrals and prose are
scored as text rather than guessed at.

Values SymPy compares exactly, such as rationals and ``\sqrt{2}``, are never
compared approximately, which is why ``99999999999`` is not ``100000000000``.

Diverges from Inspect AI: answers built from structure this rewrite does not
cover -- matrices, integrals, piecewise braces -- are compared as text, and a
boxed word such as ``\text{Yes}`` is accepted where Inspect rejects it.
"""

from __future__ import annotations

import ast
import collections.abc as c
import decimal
import re
import typing as t
from math import isfinite

import sympy

from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

_NUMBER = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?\s*%?"
)
_PERCENT_SUFFIX = re.compile(
    r"\s*(?:\\?%|\\percent|\\text\s*\{\s*(?:percent(?:age)?|pct)\s*\}|"
    r"\s+(?:percent(?:age)?|pct))\s*$",
    re.IGNORECASE,
)
_MARKER = re.compile(
    r"(?:final\s+answer|answer|result)\s*(?:is\b|[:=])\s*", re.IGNORECASE
)
_ASSIGNMENT = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*=\s*(?P<value>.+)$", re.IGNORECASE
)
_BOX = re.compile(
    r"(?:\\(?:beginboxed|boxed|fbox)|(?<![A-Za-z\\])(?:boxed|fbox|oxed))\s*\{"
)
_DELIMITERS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
_WORD = re.compile(r"[A-Za-z]{2,}")


class MathAccuracy(Metric):
    r"""Score answers with the Inspect-style candidate ladder.

    The first candidate wins. Only an unmatched opaque prose candidate may
    fall back to later candidates that are plain numbers. Plain numbers are
    compared exactly after percent values are divided by 100, expressions are
    evaluated and compared as values, and anything else uses normalised
    case-folded text equality.

    Diverges from Inspect AI: LaTeX is rewritten rather than parsed by a
    grammar, so structure beyond fractions, roots, powers and products --
    matrices, integrals, piecewise braces -- is compared as text.
    """

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the mean per-item mathematical exact-match score.

        Args:
            predictions: Model answers whose first matching candidate wins.
            references: Expected answers.
            dataset: Unused benchmark dataset.
            dataset_config: Unused dataset configuration.
            benchmark_config: Unused benchmark configuration.

        Returns:
            The mean score, or None when there are no paired inputs.
        """
        scores = [
            float(_answer_matches(prediction=str(prediction), reference=str(reference)))
            for prediction, reference in zip(predictions, references)
        ]
        return sum(scores) / len(scores) if scores else None


def _answer_matches(prediction: str, reference: str) -> bool:
    """Compare the first candidate, falling back only from opaque prose.

    Args:
        prediction: Model completion.
        reference: Expected answer.

    Returns:
        Whether the winning candidate matches a reference candidate.
    """
    reference_candidates = _reference_candidates(reference)
    prediction_candidates = _answer_candidates(prediction)
    if not prediction_candidates:
        return False
    primary = prediction_candidates[0]
    if any(_equivalent(primary, target) for target in reference_candidates):
        return True
    if _number_value(primary) is not None or not _looks_like_prose(primary):
        return False
    return any(
        any(_equivalent(candidate, target) for target in reference_candidates)
        for candidate in prediction_candidates[1:]
        if _number_value(candidate) is not None
    )


def _answer_candidates(text: str) -> list[str]:
    """Build the Inspect-style answer candidate ladder, without expression parsing.

    Returns:
        Candidate answer strings in preference order.
    """
    text = _replace_unicode(text)
    candidates: list[str] = []
    boxes = _boxed_candidates(text)
    if boxes:
        _append_candidate(candidates, boxes[-1])
    markers = list(_MARKER.finditer(text))
    if markers:
        suffix = text[markers[-1].end() :].splitlines()
        _append_candidate(candidates, suffix[0] if suffix else None)
    _append_candidate(candidates, _last_delimited_math(text))
    if len(text) <= _MAX_CANDIDATE_CHARS:
        _append_candidate(candidates, text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    _append_candidate(candidates, lines[-1] if lines else None)
    numbers = _NUMBER.findall(text)
    _append_candidate(candidates, numbers[-1] if numbers else None)
    return candidates


def _append_candidate(candidates: list[str], candidate: str | None) -> None:
    """Add a non-empty, normalised candidate once."""
    if candidate:
        candidate = _strip_delimiters(candidate)
        if candidate and candidate not in candidates:
            candidates.append(candidate)


def _strip_delimiters(text: str) -> str:
    """Remove display, Markdown, TeX spacing, and punctuation wrappers.

    Returns:
        The unwrapped text.
    """
    text = text.strip()
    while text.startswith("**") and text.endswith("**") and len(text) >= 4:
        text = text[2:-2].strip()
    for opening, closing in _DELIMITERS:
        if (
            text.startswith(opening)
            and text.endswith(closing)
            and len(text) > len(opening) + len(closing)
        ):
            text = text[len(opening) : -len(closing)].strip()
            break
    text = re.sub(r"^(?:\\[,;:]|\\quad|\\qquad|\\;|\\,)\s*", "", text)
    text = re.sub(r"(?:\\[,;:]|\\quad|\\qquad|\\;|\\,)\s*$", "", text)
    text = text.rstrip(" .,;:")
    # A boxed equation naming the answer, such as `x = 5`, is the value it names, as
    # Inspect AI's symbolic comparison also reduces it to that
    assignment = _ASSIGNMENT.match(text)
    if assignment is not None and _number_value(assignment.group("value")) is not None:
        text = assignment.group("value").strip()
    return text


def _number_value(text: str) -> tuple[decimal.Decimal, bool] | None:
    """Parse a plain number and return its exact value and percent status.

    Returns:
        An exact numeric value and percent flag, or None for non-numeric text.
    """
    normalised = _normalize_text(text)
    percent_match = _PERCENT_SUFFIX.search(normalised)
    percent = percent_match is not None
    if percent_match is not None:
        normalised = normalised[: percent_match.start()].strip()
    normalised = normalised.replace(r"\,", ",")
    if "," in normalised and not re.fullmatch(
        r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][-+]?\d+)?", normalised
    ):
        return None
    normalised = normalised.replace(",", "")
    if not re.fullmatch(
        r"[-+]?(?:(?:\d{1,3}(?:\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        normalised,
    ):
        return None
    try:
        return decimal.Decimal(normalised), percent
    except decimal.InvalidOperation:
        return None


def _normalize_text(text: str) -> str:
    """Apply the shared normalisation used for both predictions and references.

    Returns:
        The case-folded normalised text.
    """
    text = _replace_unicode(_strip_delimiters(text))
    text = re.sub(r"\\(?:text|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"^[£€$]\s*", "", text)
    text = text.replace(r"\ ", " ").replace(r"\%", "%")
    text = re.sub(r"\s+", " ", text).strip(" .,;:")
    return text.casefold()


def _replace_unicode(text: str) -> str:
    """Replace mathematical Unicode symbols using Inspect AI's table.

    Returns:
        Text with mathematical Unicode symbols replaced.
    """
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    replacements = {
        "\u23a7": r"\boxed{",
        "\u23ab": "}",
        "\n\u2502": r"\boxed{",
        "\u2502": "}",
        "\n\u2503": r"\boxed{",
        "\u2503": "}",
        "\n\uf8f0": r"\boxed{",
        "\uf8fb": "}",
        "√": r"\sqrt",
        "×": r"\cdot",
        "÷": "/",
        "\u202f": " ",
        "−": "-",
        "–": "-",
        "π": r"\pi",
        "°": r"^\circ",
        "∞": r"\infty",
        "≤": r"\le",
        "≥": r"\ge",
        "≠": r"\ne",
        "∪": r"\cup",
        "∩": r"\cap",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    # A line-oriented box has an opening glyph on the first line and a closing
    # glyph on the last; the table's overlapping newline and single-glyph forms
    # can otherwise turn the latter into a second opener.
    text = re.sub(r"\\boxed\{([^{}]*)\\boxed\{", r"\\boxed{\1}", text)
    return text


def _boxed_candidates(text: str) -> list[str]:
    """Return brace-balanced contents of boxes, in source order."""
    matches: list[str] = []
    position = 0
    while match := _BOX.search(text, position):
        content = _balanced_content(text, match.end() - 1)
        if content is not None:
            matches.append(content[0])
            position = content[1]
        else:
            position = match.end()
    return matches


def _balanced_content(text: str, opening: int) -> tuple[str, int] | None:
    """Return content and end position for a balanced opening brace."""
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index], index + 1
    return None


def _last_delimited_math(text: str) -> str | None:
    """Return the delimited mathematical span ending furthest to the right."""
    spans: list[tuple[int, str]] = []
    for opening, closing in _DELIMITERS:
        end = text.rfind(closing)
        start = text.rfind(opening, 0, end)
        if start >= 0 and end > start:
            spans.append((end, text[start + len(opening) : end]))
    return max(spans, default=(0, None))[1]


def _equivalent(left: str, right: str) -> bool:
    r"""Compare plain numbers exactly, then values, then normalised text.

    Plain numbers are compared without any mathematics, so every value is exact and
    ``==`` is intentional. Everything else is compared as a value by
    ``_symbolically_equivalent``, which is where ``\frac{1}{2}`` meets ``0.5`` and
    ``x + y`` meets ``y + x``; a string it cannot translate, and an answer made of
    several words, is compared as text instead.

    Returns:
        Whether the values are equivalent.
    """
    left_boxes = _boxed_candidates(left)
    right_boxes = _boxed_candidates(right)
    if left_boxes:
        left = left_boxes[-1]
    if right_boxes:
        right = right_boxes[-1]
    left_number = _number_value(left)
    right_number = _number_value(right)
    if left_number is not None and right_number is not None:
        left_value = _percent_value(left_number)
        right_value = _percent_value(right_number)
        return left_value == right_value
    symbolic = _symbolically_equivalent(left, right)
    if symbolic is not None:
        return symbolic
    return _normalize_text(left) == _normalize_text(right)


def _percent_value(number: tuple[decimal.Decimal, bool]) -> decimal.Decimal:
    """Return the value a parsed number denotes, dividing a percentage by 100.

    Returns:
        The exact value, with no rounding at any length of digits.
    """
    value, percent = number
    if not percent:
        return value
    with decimal.localcontext() as context:
        # Dividing would round at the context's 28 digits; widening it to the digits in
        # hand keeps 99...9% distinct from its nearest neighbour.
        context.prec = len(value.as_tuple().digits) + 2
        return value.scaleb(-2)


def _looks_like_prose(candidate: str) -> bool:
    """Return whether a candidate is opaque prose rather than an expression."""
    if "\\" in candidate or any(char in candidate for char in "=+-*/^<>[]{}()"):
        return False
    return len(_WORD.findall(candidate)) >= 2


def _reference_candidates(text: str) -> list[str]:
    """Extract the boxed reference followed by its complete text.

    Returns:
        Candidate reference strings in preference order.
    """
    text = _replace_unicode(text)
    boxes = _boxed_candidates(text)
    values = [_strip_delimiters(boxes[-1])] if boxes else []
    values.append(text)
    return _unique(values)


def _unique(values: list[str]) -> list[str]:
    """Return values with duplicates removed while preserving order."""
    return list(dict.fromkeys(value for value in values if value))


math_accuracy_metric = MathAccuracy(name="math_accuracy", pretty_name="Math Accuracy")


_MAX_CANDIDATE_CHARS = 4_096

_MAX_INTEGER_BITS = 512

_MAX_NODES = 512

_MAX_DEPTH = 64

_MAX_ARGUMENTS = 128

_MAX_SYMBOL_CHARS = 64

_MAX_SYMBOLS = 64

_MAX_POWER_EXPONENT = 512

_MAX_RECURSION = 16


_GROUPED_NUMBER = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][-+]?\d+)?")

_COMMAND = re.compile(r"\\([A-Za-z]+)")

_TOKEN = re.compile(r"[-+]?\w+")

_POWER = re.compile(r"\s*\^")

_BARE_POWER = re.compile(r"\s*([-+]?\w+)")

_HEADS = r"(?:pi|E|oo|sqrt|abs|exp|log|ln|sin|cos|tan)"

_UNSUPPORTED = re.compile(
    r"\\(?:int|oint|iint|sum|prod|lim|limits|begin|end|matrix|array|pmatrix|bmatrix|"
    r"vmatrix|cases|align|stackrel|overset|underset|binom|vec|hat|bar|dot|mathbf|"
    r"mathit|operatorname|det|gcd|mod|pmod|to|rightarrow|Rightarrow|cup|cap|in|"
    r"subseteq|forall|exists|neg|land|lor|alpha|beta|gamma|theta|lambda|mu|sigma|"
    r"Delta|Omega|partial|nabla|approx|neq|leq|geq|lt|gt|ne)\b"
)

_LAYOUT = re.compile(
    r"\\(?:left|right|middle|quad|qquad|displaystyle)\b"
    r"|\\(?:Bigg?|bigg?)[lrm]?"
    r"|\\[,:!;]"
    r"|\\ "
)

_TEXTISH = re.compile(r"\\(?:text|textrm|mathrm|mbox|mathsf|mathbb|mathcal)\s*")

_PLAIN_IN_TEXT = re.compile(r"^[\d\s.,+\-*/=:%]+$")


def _has_free_symbols(value: sympy.Expr) -> bool:
    """Return whether a value names something rather than denoting a value."""
    return bool(_unwrap(value).free_symbols)


def _unwrap(expression: sympy.Expr) -> sympy.Expr:
    """Reduce `x = 5` to the value it names, whichever side carries it.

    Returns:
        The expression, or the value an equality names.
    """
    if not isinstance(expression, sympy.Equality):
        return expression
    left_is_symbol = bool(expression.lhs.is_Symbol)
    right_is_symbol = bool(expression.rhs.is_Symbol)
    if right_is_symbol and not left_is_symbol:
        return expression.lhs
    return expression.rhs


def _symbolically_equivalent(left: str, right: str) -> bool | None:
    """Compare two answers as mathematical values.

    Args:
        left: Answer string, LaTeX or plain arithmetic.
        right: Answer string, LaTeX or plain arithmetic.

    Returns:
        Whether the two denote the same value, or None when either string is outside
        what this module can parse and the caller should compare them as text instead.
    """
    if _looks_like_prose(left) or _looks_like_prose(right):
        # An answer of several words is text rather than a product of names, and
        # Inspect AI reads it that way too; anything else is committed to SymPy,
        # including a name like `CO2`, which is where it scores `co2` as wrong.
        return None
    left_value = _parse(left)
    right_value = _parse(right)
    if left_value is None or right_value is None:
        return None
    return _expressions_equivalent(left_value, right_value)


def _expressions_equivalent(left: sympy.Expr, right: sympy.Expr) -> bool:
    """Compare two values the way Inspect AI's math scorer does.

    Returns:
        Whether the two denote the same value.
    """
    left, right = _unwrap(left), _unwrap(right)
    try:
        if bool(left == right):
            return True
    except Exception:
        pass
    try:
        if bool(left.equals(right)):
            return True
    except Exception:
        pass
    return _close_enough(left, right)


def _close_enough(left: sympy.Expr, right: sympy.Expr) -> bool:
    """Apply the floating-point tolerance, but never to two exact values.

    Returns:
        Whether the two values are close enough to count as the same one.
    """
    if not all(_is_numeric(value) for value in (left, right)):
        return False
    if all(_is_exact(value) for value in (left, right)):
        return False
    try:
        values = [complex(sympy.N(value, 30)) for value in (left, right)]
    except (TypeError, ValueError, ArithmeticError):
        return False
    if not all(isfinite(value.real) and isfinite(value.imag) for value in values):
        return False
    error = abs(values[0] - values[1])
    scale = max(abs(values[0]), abs(values[1]), 1e-10)
    return error < 1e-10 or error / scale < 1e-10


def _is_exact(expression: sympy.Expr) -> bool:
    """Return whether SymPy can compare the value without approximating it."""
    return bool(expression.is_number) and bool(expression.is_algebraic)


def _is_numeric(expression: sympy.Expr) -> bool:
    """Return whether the expression is a number rather than a formula."""
    return isinstance(expression, sympy.Expr) and not expression.free_symbols


def _parse(text: str) -> sympy.Expr | None:
    """Rewrite a candidate and build it as a SymPy expression.

    Returns:
        The expression, or None if the candidate does not translate or is out of bounds.
    """
    try:
        plain = _to_plain(text)
        if plain is None:
            return None
        tree = ast.parse(plain, mode="eval")
        expression = _build(tree)
        _check(expression)
    except (
        MathParseError,
        SyntaxError,
        ValueError,
        RecursionError,
        MemoryError,
        OverflowError,
    ):
        return None
    except ZeroDivisionError:
        return None
    return expression


def _build(tree: ast.Expression) -> sympy.Expr:
    """Turn a parsed plain expression into SymPy values.

    Returns:
        The value the expression denotes.
    """

    def build(node: ast.AST) -> sympy.Expr:
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant):
            return _constant(node)
        if isinstance(node, ast.Name):
            if len(node.id) > _MAX_SYMBOL_CHARS:
                raise MathParseError("symbol name is too long")
            return _CONSTANTS.get(node.id, sympy.Symbol(node.id))
        if isinstance(node, ast.UnaryOp):
            operand = build(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise MathParseError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            return _binary(node, build)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise MathParseError("unsupported function call")
            if len(node.args) != 1 or node.keywords:
                raise MathParseError("function takes exactly one argument")
            return _FUNCTIONS[node.func.id](build(node.args[0]))
        return _structure(node, build)

    return build(tree)


class MathParseError(ValueError):
    """Raised when a candidate cannot be turned into a value."""


def _binary(node: ast.BinOp, build: t.Callable[[ast.AST], sympy.Expr]) -> sympy.Expr:
    """Apply an arithmetic operator to two built operands.

    Returns:
        The result of the operation.

    Raises:
        MathParseError: If the operator is not arithmetic.
    """
    left, right = build(node.left), build(node.right)
    if isinstance(left, sympy.Tuple) or isinstance(right, sympy.Tuple):
        raise MathParseError("a sequence is not an arithmetic operand")
    if isinstance(node.op, ast.Add):
        return sympy.Add(left, right)
    if isinstance(node.op, ast.Sub):
        return sympy.Add(left, -right)
    if isinstance(node.op, ast.Mult):
        return sympy.Mul(left, right)
    if isinstance(node.op, ast.Div):
        return sympy.Mul(left, sympy.Pow(right, -1))
    if isinstance(node.op, ast.Pow):
        _guard_power(node.right)
        return sympy.Pow(left, right)
    if isinstance(node.op, ast.FloorDiv):
        return sympy.floor(sympy.Mul(left, sympy.Pow(right, -1)))
    if isinstance(node.op, ast.Mod):
        return sympy.Mod(left, right)
    raise MathParseError("unsupported operator")


def _constant(node: ast.Constant) -> sympy.Expr:
    """Build a number, keeping integers exact and decimals as written.

    Returns:
        The value of the literal.

    Raises:
        MathParseError: If the literal is not a number.
    """
    if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
        raise MathParseError(f"unsupported constant {node.value!r}")
    if isinstance(node.value, int):
        return sympy.Integer(node.value)
    return sympy.Float(repr(node.value))


def _structure(node: ast.AST, build: t.Callable[[ast.AST], sympy.Expr]) -> sympy.Expr:
    """Build a sequence or an equation, both of which are values but not expressions.

    Returns:
        The tuple or equality the node denotes.

    Raises:
        MathParseError: If the node is neither, or is too large.
    """
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        if len(node.elts) > _MAX_ARGUMENTS:
            raise MathParseError("sequence is too long")
        # A structured answer, such as `(1, 2)`, is not an Expr, but it is still
        # comparable as a value, so it is passed through as one.
        return t.cast("sympy.Expr", sympy.Tuple(*[build(e) for e in node.elts]))
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            raise MathParseError("unsupported comparison")
        # An equation is a Boolean rather than an Expr; `_unwrap` reduces it to the
        # value it names before anything is compared with it.
        return t.cast(
            "sympy.Expr",
            sympy.Eq(build(node.left), build(node.comparators[0]), evaluate=False),
        )
    raise MathParseError("expression is not arithmetic")


_CONSTANTS = {
    "pi": sympy.pi,
    "E": sympy.E,
    "e": sympy.E,
    "oo": sympy.oo,
    "inf": sympy.oo,
    "infinity": sympy.oo,
}

_FUNCTIONS = {
    "sqrt": sympy.sqrt,
    "abs": sympy.Abs,
    "exp": sympy.exp,
    "log": sympy.log,
    "ln": sympy.log,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
}


def _check(expression: sympy.Expr) -> None:
    """Refuse expressions whose size would make comparison unbounded.

    Raises:
        MathParseError: If the expression is too large, deep, or expensive.
    """
    roots = list(expression) if isinstance(expression, sympy.Tuple) else [expression]
    nodes = 0
    symbols: set[str] = set()
    stack: list[tuple[sympy.Basic, int]] = [(root, 1) for root in roots]
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_NODES:
            raise MathParseError("expression has too many nodes")
        if depth > _MAX_DEPTH:
            raise MathParseError("expression is nested too deeply")
        if isinstance(node, (sympy.Integer, sympy.Rational)):
            numerators = [node] if isinstance(node, sympy.Integer) else [node.p, node.q]
            for value in numerators:
                if _bit_length(value) > _MAX_INTEGER_BITS:
                    raise MathParseError("integer is too large")
        if isinstance(node, sympy.Symbol):
            symbols.add(node.name)
            if len(symbols) > _MAX_SYMBOLS:
                raise MathParseError("expression has too many symbols")
        if isinstance(node, sympy.Pow) and _is_expensive_power(node):
            raise MathParseError("exponentiation is too expensive")
        arguments = getattr(node, "args", ())
        if len(arguments) > _MAX_ARGUMENTS:
            raise MathParseError("expression has too many arguments")
        stack.extend(
            (argument, depth + 1)
            for argument in arguments
            if isinstance(argument, sympy.Basic)
        )


def _bit_length(value: object) -> int:
    """Return the bit length of an integer-valued SymPy object."""
    try:
        return abs(int(t.cast("int", value))).bit_length()
    except (TypeError, ValueError, OverflowError):
        return _MAX_INTEGER_BITS + 1


def _is_expensive_power(node: sympy.Pow) -> bool:
    """Return whether a power is an exponent tower or has a huge exponent."""
    exponent = node.exp
    if isinstance(exponent, sympy.Pow):
        return True
    return isinstance(exponent, sympy.Integer) and _bit_length(exponent) > 8


def _guard_power(exponent: ast.expr) -> None:
    """Reject exponentiation that SymPy would evaluate into something unbounded.

    SymPy evaluates a power as it is built, so `9^{9^{9}}` has to be refused here rather
    than after parsing: by the time the tree exists the integer has been computed. The
    exponent is therefore measured before it is built, including when it is written as
    `-1000000000` or `1000000000 - 1`.

    Raises:
        MathParseError: If the exponent is a tower or too large to expand.
    """
    if any(isinstance(node, ast.Pow) for node in ast.walk(exponent)):
        raise MathParseError("exponentiation is nested")
    size = _exponent_size(exponent)
    if size is not None and size > _MAX_POWER_EXPONENT:
        raise MathParseError("exponent is too large")


def _exponent_size(node: ast.expr) -> float | None:
    """Estimate how big a power's exponent is, without evaluating it.

    Returns:
        An upper bound on the magnitude, or None when the exponent is symbolic or is
        something SymPy leaves unevaluated, in which case nothing is expanded.

    Raises:
        MathParseError: If the exponent is not a number at all.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise MathParseError("exponent is not a number")
        return abs(float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _exponent_size(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    ):
        left = _exponent_size(node.left)
        right = _exponent_size(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return left + right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Mod):
            return left
        if right < 1.0:
            return float("inf") if right == 0.0 else left / right
        return left
    if isinstance(node, (ast.Name, ast.Call)):
        return None
    raise MathParseError("exponent is not a number")


def _to_plain(text: str, depth: int = 0) -> str | None:
    """Rewrite a LaTeX answer into an arithmetic expression Python can parse.

    Returns:
        The plain expression, or None where the rewrite would change the meaning.
    """
    if depth > _MAX_RECURSION or len(text) > _MAX_CANDIDATE_CHARS:
        return None
    text = text.strip()
    for opening, closing in _DELIMITERS:
        if (
            text.startswith(opening)
            and text.endswith(closing)
            and len(text) > len(opening) + len(closing)
        ):
            text = text[len(opening) : -len(closing)].strip()
            break
    if not text:
        return None
    text = (
        text.replace("\u00d7", "*")
        .replace("\u00f7", "/")
        .replace("\u2212", "-")
        .replace("\u2215", "/")
        .replace("\u2219", "*")
        .replace("\u00b0", "")
    )
    if _UNSUPPORTED.search(text):
        return None
    # Thin spaces used as a thousands separator read as the comma form, so that
    # `1\,234` is one number rather than two tokens.
    text = re.sub(r"(?<=\d)\\,(?=\d{3}(?:\D|$))", ",", text)
    text = _LAYOUT.sub(" ", text)
    percent = _PERCENT_SUFFIX.search(text) is not None
    if percent:
        text = _PERCENT_SUFFIX.sub("", text).strip()
    if "%" in text:
        # A percent sign that is not the trailing one would otherwise reach the parser
        # as Python's modulo operator, scoring `50% + 10%` as a remainder.
        return None
    rewritten = _rewrite(text, depth)
    if rewritten is None:
        return None
    plain = _powers(rewritten, depth)
    if plain is None:
        return None
    plain = _products(plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    if not plain:
        return None
    if _GROUPED_NUMBER.fullmatch(plain):
        plain = plain.replace(",", "")
    elif "," in plain and not _keeps_commas(plain):
        return None
    if "\\" in plain:
        return None
    if percent:
        plain = f"({plain})/100"
    return plain


def _keeps_commas(plain: str) -> bool:
    """Return whether commas in a plain expression are structure rather than digits."""
    return bool(re.fullmatch(r"[(\[].*[)\]]|\{.*\}", plain.strip()))


def _powers(text: str, depth: int) -> str | None:
    """Rewrite ``x^{...}`` and ``x^n`` as Python exponentiation.

    Returns:
        The expression with ``**`` powers, or None where an exponent is unreadable.
    """
    if depth > _MAX_RECURSION or len(text) > _MAX_CANDIDATE_CHARS:
        return None
    out: list[str] = []
    position = 0
    while match := _POWER.search(text, position):
        out.append(text[position : match.start()])
        after = match.end()
        if after < len(text) and text[after] == "{":
            content, after = _balanced(text, after)
            if content is None:
                return None
            inner = _powers(content, depth + 1)
            if inner is None:
                return None
            out.append(f"**({inner})")
        else:
            token = _BARE_POWER.match(text, after)
            if token is None:
                return None
            out.append(f"**({token.group(1)})")
            after = token.end()
        position = after
    out.append(text[position:])
    return "".join(out)


def _balanced(text: str, opening: int) -> tuple[str | None, int]:
    """Return the contents of a balanced brace group and the index after it."""
    found = _balanced_content(text, opening)
    return (None, len(text)) if found is None else found


def _products(text: str) -> str:
    r"""Insert the multiplication that LaTeX leaves implicit, as in ``2\\pi``.

    Returns:
        The expression with explicit products where a coefficient runs into a value.
    """
    text = re.sub(r"([\d.])\s*(" + _HEADS + r")\b", r"\1*\2", text)
    return re.sub(r"([\d.)])\s*\(", r"\1*(", text)


def _rewrite(text: str, depth: int) -> str | None:
    """Replace value-bearing LaTeX commands with arithmetic, recursively.

    Returns:
        The rewritten text, or None for a command with no arithmetic meaning.
    """
    if depth > _MAX_RECURSION or len(text) > _MAX_CANDIDATE_CHARS:
        return None
    out: list[str] = []
    position = 0
    while match := _COMMAND.search(text, position):
        out.append(text[position : match.start()])
        name = match.group(1)
        after = match.end()
        if name in ("frac", "dfrac", "tfrac", "cfrac"):
            numerator, after = _argument(text, after)
            denominator, after = _argument(text, after)
            if numerator is None or denominator is None:
                return None
            left, right = (
                _rewrite(numerator, depth + 1),
                _rewrite(denominator, depth + 1),
            )
            if left is None or right is None:
                return None
            out.append(f"(({left})/({right}))")
        elif name == "sqrt":
            index, after = _optional_argument(text, after)
            radicand, after = _argument(text, after)
            if radicand is None or (index is not None and index.isdigit() is False):
                return None
            root = _rewrite(radicand, depth + 1)
            if root is None:
                return None
            # Written as a call or an exact rational power, never as `**0.5`, which
            # SymPy would evaluate to a float and lose the exactness of the root.
            out.append(f"sqrt({root})" if not index else f"(({root})**(1/{index}))")
        elif name in ("cdot", "ast", "times"):
            out.append("*")
        elif name == "div":
            out.append("/")
        elif name == "pi":
            out.append(" pi ")
        elif name == "euler":
            out.append(" E ")
        elif name in ("infty", "infinity"):
            out.append(" oo ")
        elif name in ("text", "textrm", "mathrm", "mbox", "mathsf") or _TEXTISH.match(
            text, match.start()
        ):
            content, after = _argument(text, after)
            if content is None or not _PLAIN_IN_TEXT.fullmatch(content):
                return None
            out.append(f" ({content}) " if content.strip() else " ")
        else:
            return None
        position = after
    out.append(text[position:])
    rewritten = "".join(out)
    return re.sub(r"\s+", " ", rewritten) if depth == 0 else rewritten


def _argument(text: str, position: int) -> tuple[str | None, int]:
    """Read one LaTeX argument, either braced or a single token.

    Returns:
        The argument and the position after it, or None and where reading stopped.
    """
    while position < len(text) and text[position] in " \t":
        position += 1
    if position >= len(text):
        return None, position
    if text[position] == "{":
        content, end = _balanced(text, position)
        return content, end
    match = _TOKEN.match(text, position)
    if match is None:
        return None, position
    if match.group(0)[:1] in "+-" and len(match.group(0)) > 2:
        return match.group(0)[:2], position + 2
    return match.group(0), match.end()


def _optional_argument(text: str, position: int) -> tuple[str | None, int]:
    """Read a bracketed argument, such as the degree of a root.

    Returns:
        The argument, None when absent, and the position after it.
    """
    while position < len(text) and text[position] in " \t":
        position += 1
    if position < len(text) and text[position] == "[":
        end = text.find("]", position)
        if end == -1:
            return None, position
        return text[position + 1 : end].strip(), end + 1
    return None, position
