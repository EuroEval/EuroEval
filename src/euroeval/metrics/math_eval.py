r"""Symbolic comparison of mathematical answers, without a LaTeX parser.

Inspect AI parses LaTeX with an ANTLR grammar shipped by ``latex2sympy2-extended``. That
grammar is not needed for benchmark answers, which are single expressions: this module
rewrites the LaTeX forms such answers actually use into plain arithmetic, parses that
with :mod:`ast`, and compares the result with SymPy, which is already installed as a
dependency of torch. So ``\frac{1}{2}`` equals ``0.5``, ``-0.25`` equals ``-1/4``, and
``2\pi`` equals ``6.283185307179586``.

Anything the rewrite cannot render faithfully -- matrices, integrals, piecewise
braces, words inside ``\text{}`` -- is reported as unknown rather than guessed at, and
the caller falls back to text equality. Unknown is never wrong in the direction of
silently resymboling prose.

Values that SymPy can compare exactly, such as rationals and ``\sqrt{2}``, are compared
exactly only; the floating-point tolerance is reserved for values that are not exact, so
that a nearby-but-different number is never accepted.
"""

from __future__ import annotations

import ast
import re
import typing as t

import sympy

# Limits mirror the guardrails Inspect AI puts in front of SymPy, whose simplifier will
# otherwise happily spend forever on `9^{9^{9}}`.
_MAX_CANDIDATE_CHARS = 4_096
_MAX_INTEGER_BITS = 512
_MAX_NODES = 512
_MAX_DEPTH = 64
_MAX_ARGUMENTS = 128
_MAX_SYMBOL_CHARS = 64
_MAX_SYMBOLS = 64
_MAX_POWER_EXPONENT = 64
_MAX_RECURSION = 16

_PERCENT_SUFFIX = re.compile(
    r"(?:\\%|\\percent|\\text\s*\{\s*(?:percent|pct)\s*\}|\s*%|\s+(?:percent|pct))\s*$",
    re.IGNORECASE,
)
_GROUPED_NUMBER = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_COMMAND = re.compile(r"\\([A-Za-z]+)")
_TOKEN = re.compile(r"[-+]?\w+")
_POWER = re.compile(r"\s*\^")
_BARE_POWER = re.compile(r"\s*([-+]?\w+)")
# Functions and constants that LaTeX writes next to a coefficient, as in `2\pi`.
_HEADS = r"(?:pi|E|oo|sqrt|abs|exp|log|ln|sin|cos|tan)"

# Structure this module does not attempt: bail out instead of misreading it.
_UNSUPPORTED = re.compile(
    r"\\(?:int|oint|iint|sum|prod|lim|limits|begin|end|matrix|array|pmatrix|bmatrix|"
    r"vmatrix|cases|align|stackrel|overset|underset|binom|vec|hat|bar|dot|mathbf|"
    r"mathit|operatorname|det|gcd|mod|pmod|to|rightarrow|Rightarrow|cup|cap|in|"
    r"subseteq|forall|exists|neg|land|lor|alpha|beta|gamma|theta|lambda|mu|sigma|"
    r"Delta|Omega|partial|nabla|approx|neq|leq|geq|lt|gt|ne)\b"
)
# Sizing and spacing that carries no value at all.
_LAYOUT = re.compile(
    r"\\(?:left|right|middle|quad|qquad|displaystyle)\b"
    r"|\\(?:Bigg?|bigg?)[lrm]?"
    r"|\\[,:!;]"
    r"|\\ "
)
_TEXTISH = re.compile(r"\\(?:text|textrm|mathrm|mbox|mathsf|mathbb|mathcal)\s*")
_PLAIN_IN_TEXT = re.compile(r"^[\d\s.,+\-*/=:%]+$")


def is_symbolically_equivalent(left: str, right: str) -> bool | None:
    """Compare two answers as mathematical values.

    Args:
        left: Answer string, LaTeX or plain arithmetic.
        right: Answer string, LaTeX or plain arithmetic.

    Returns:
        Whether the two denote the same value, or None when either string is outside
        what this module can parse and the caller should compare them as text instead.
    """
    left_value = _parse(left)
    right_value = _parse(right)
    if left_value is None or right_value is None:
        return None
    return _equivalent(left_value, right_value)


def _equivalent(left: sympy.Expr, right: sympy.Expr) -> bool:
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
    if not all(
        value.real == value.real and value.imag == value.imag for value in values
    ):
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


def _parse(text: str) -> sympy.Expr | None:
    """Rewrite a candidate and build it as a SymPy expression.

    Returns:
        The expression, or None if the candidate does not translate or is out of bounds.
    """
    plain = _to_plain(text)
    if plain is None:
        return None
    try:
        tree = ast.parse(plain, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    try:
        expression = _build(tree)
        _check(expression)
    except (MathParseError, RecursionError, MemoryError, OverflowError):
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
    than after parsing: by the time the tree exists the integer has been computed.

    Raises:
        MathParseError: If the exponent is a tower or a literal too large to expand.
    """
    if any(isinstance(node, ast.Pow) for node in ast.walk(exponent)):
        raise MathParseError("exponentiation is nested")
    value = exponent.value if isinstance(exponent, ast.Constant) else None
    if isinstance(value, int) and abs(value) > _MAX_POWER_EXPONENT:
        raise MathParseError("exponent is too large")


def _to_plain(text: str, depth: int = 0) -> str | None:
    """Rewrite a LaTeX answer into an arithmetic expression Python can parse.

    Returns:
        The plain expression, or None where the rewrite would change the meaning.
    """
    if depth > _MAX_RECURSION or len(text) > _MAX_CANDIDATE_CHARS:
        return None
    text = text.strip()
    for opening, closing in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
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
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index], index + 1
    return None, len(text)


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
