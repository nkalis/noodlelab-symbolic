"""Reading expressions and equations typed into widgets, safely.

SymPy's own ``sympify`` and ``parse_expr`` call ``eval``, so text from the
editor would run as Python. Before SymPy sees it, the text is tokenized and
only numbers, names and arithmetic operators are let through: no strings, no
attribute access, no names starting with ``_``, no keywords. Names are then
either one of the functions and constants in :data:`FUNCTIONS` or a symbol.

The notation is SymPy's, with a few conveniences for engineers:

* ``^`` is a power, like ``**``, and ``2 x`` is ``2*x``;
* ``E`` and ``I`` are symbols (Young's modulus, second moment of area), not
  Euler's number and the imaginary unit: write ``e`` or ``exp(1)``, and ``j``;
* ``lambda`` is a symbol too (printed λ);
* primes are derivatives: ``y''(x)`` is the second derivative of ``y`` with
  respect to ``x``, and ``y'(0)`` its first derivative at ``x = 0`` (when the
  variable is known, as in Solve ODE's conditions);
* ``{a}`` … ``{d}`` stand for linked expressions, in parentheses.
"""

from __future__ import annotations

import ast
import io
import keyword
import re
import tokenize
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

from .._expr import pow_problem

MAX_LENGTH = 4000

FUNCTIONS: dict[str, Any] = {
    # constants
    "pi": sp.pi,
    "e": sp.E,
    "j": sp.I,
    "oo": sp.oo,
    "inf": sp.oo,
    # elementary functions
    **{
        name: getattr(sp, name)
        for name in [
            "sin",
            "cos",
            "tan",
            "cot",
            "sec",
            "csc",
            "asin",
            "acos",
            "atan",
            "atan2",
            "acot",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "exp",
            "log",
            "sqrt",
            "cbrt",
            "root",
            "Abs",
            "sign",
            "floor",
            "ceiling",
            "Min",
            "Max",
            "re",
            "im",
            "conjugate",
            "factorial",
            "erf",
            "erfc",
            "Heaviside",
            "DiracDelta",
            "SingularityFunction",
            "Piecewise",
        ]
    },
    "arg": sp.arg,  # the phase of a complex number
    "ln": sp.log,
    "abs": sp.Abs,
    "min": sp.Min,
    "max": sp.Max,
    # calculus, left unevaluated or evaluated
    "diff": sp.diff,
    "integrate": sp.integrate,
    "limit": sp.limit,
    "Derivative": sp.Derivative,
    "Integral": sp.Integral,
    "Subs": sp.Subs,
    "Rational": sp.Rational,
}
# what parse_expr's transformations produce: Symbol('x'), Integer(2), Function('y')...
_MACHINERY = {
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Function": sp.Function,
}
_TRANSFORMS = (*standard_transformations, convert_xor)
_OPERATORS = frozenset(["+", "-", "*", "/", "**", "^", "(", ")", ",", "<", ">", "<=", ">="])
# Python's == and != compare SymPy objects structurally: x == 1 would read as False
_COMPARISONS = frozenset(["==", "!="])
_RENAMED = {"lambda": "lamda"}  # a keyword in Python; SymPy prints lamda as λ
_PLACEHOLDER = re.compile(r"\{([a-d])\}")
_PRIMES = re.compile(r"\b([A-Za-z]\w*)('+)\(([^()]*)\)")
# an "=" that is not part of ==, <=, >= or !=
_EQUALS = re.compile(r"(?<![<>=!])=(?!=)")


class ParseError(ValueError):
    """Text that is not a valid expression; the message is shown on the node."""


def _derivatives(text: str, variable: str | None) -> str:
    """``y''(x)`` as ``Derivative(y(x), x, 2)`` and ``y'(0)`` as
    ``Subs(Derivative(y(x), x, 1), x, 0)``."""

    def repl(m: re.Match[str]) -> str:
        name, order, arg = m.group(1), len(m.group(2)), m.group(3).strip()
        if not arg:
            raise ParseError(f"{name}{m.group(2)}() needs an argument, such as {name}'(x)")
        is_name = arg.isidentifier()
        if variable is None or (is_name and arg == variable):
            if not is_name:
                raise ParseError(
                    f"In {m.group(0)}, the variable is unknown: write it as a derivative "
                    f"with respect to a symbol, such as {name}{m.group(2)}(x)"
                )
            return f"Derivative({name}({arg}), {arg}, {order})"
        v = variable
        return f"Subs(Derivative({name}({v}), {v}, {order}), {v}, ({arg}))"

    return _PRIMES.sub(repl, text)


def _check_tokens(text: str, allowed_private: frozenset[str]) -> str:
    """The text with keywords renamed, or ParseError if it has anything but
    numbers, names and arithmetic."""
    out: list[tuple[int, str]] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise ParseError(f"Unbalanced brackets or an incomplete expression ({exc})") from None
    for tok in tokens:
        kind, s = tok.type, tok.string
        if kind in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.INDENT):
            pass
        elif kind == tokenize.DEDENT:
            continue
        elif kind == tokenize.NUMBER:
            if s[-1] in "jJ":
                raise ParseError(f"'{s}': write imaginary numbers as {s[:-1]}*j")
        elif kind == tokenize.NAME:
            s = _RENAMED.get(s, s)
            if s.startswith("_") and s not in allowed_private:
                raise ParseError(f"Names may not start with an underscore: '{s}'")
            if keyword.iskeyword(s):
                raise ParseError(f"'{s}' is a reserved word; choose another symbol name")
        elif kind == tokenize.OP:
            if s in _COMPARISONS:
                raise ParseError(f"'{s}' is not allowed: write an equation as lhs = rhs")
            if s not in _OPERATORS:
                raise ParseError(f"'{s}' is not allowed in an expression")
        else:
            raise ParseError(f"'{s}' is not allowed in an expression")
        if out and _juxtaposed(out[-1], (kind, s)):
            out.append((tokenize.OP, "*"))  # implicit multiplication: 2 x, E I, (a)(b)
        out.append((kind, s))
    return tokenize.untokenize(out).strip()


def _check_size(code: str) -> None:
    """ParseError when an integer power or factorial is too large to compute.

    SymPy evaluates ``9^9^9`` exactly, which would hold the server for hours
    (and the probe, on every keystroke). The vetted token stream is plain
    Python arithmetic once ``^`` is ``**``, so its tree can be sized by
    :func:`noodlelab.nodes._expr.pow_problem` before SymPy sees it. Text that
    Python cannot parse but SymPy can is left for SymPy, as before. Values
    linked in later (Substitute's ``.subs``) are not seen here and can still
    build a large power; the editor's probe timeout covers that.
    """
    try:
        tree = ast.parse(code.replace("^", "**"), mode="eval")
    except SyntaxError:
        return
    found = pow_problem(tree)
    if found:
        raise ParseError(found)


def _juxtaposed(prev: tuple[int, str], tok: tuple[int, str]) -> bool:
    """Two tokens side by side that Python cannot read, but mean a product.
    A name followed by ``(`` stays a function call."""
    ends_value = prev[0] in (tokenize.NAME, tokenize.NUMBER) or prev[1] == ")"
    starts_value = tok[0] in (tokenize.NAME, tokenize.NUMBER) or (
        tok[1] == "(" and prev[0] != tokenize.NAME
    )
    return ends_value and starts_value


def parse_expression(
    text: str,
    linked: dict[str, Any] | None = None,
    variable: str | None = None,
) -> sp.Expr:
    """An expression from text. ``linked`` maps ``a``…``d`` to expressions
    that ``{a}``…``{d}`` stand for; ``variable`` is the independent variable
    that ``y'(0)`` refers to."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("Enter an expression, such as w*L^2/12")
    if len(text) > MAX_LENGTH:
        raise ParseError(f"The expression is longer than {MAX_LENGTH} characters")
    linked = {k: v for k, v in (linked or {}).items() if v is not None}
    local: dict[str, Any] = {}

    def placeholder(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in linked:
            raise ParseError(f"{{{name}}} is used but nothing is linked to {name}")
        local[f"_ph_{name}"] = linked[name]
        return f"(_ph_{name})"

    text = _PLACEHOLDER.sub(placeholder, text)
    text = _derivatives(text, variable)
    code = _check_tokens(text, frozenset(local))
    if not code:
        raise ParseError("Enter an expression, such as w*L^2/12")
    _check_size(code)
    # E and I are symbols here, so they are not in the namespace at all
    namespace: dict[str, Any] = {"__builtins__": {}, **_MACHINERY, **FUNCTIONS}
    try:
        result = parse_expr(
            code, local_dict=local, global_dict=namespace, transformations=_TRANSFORMS
        )
    # the text has only been vetted token by token: SymPy may still raise anything
    # (TypeError for sin(1, 2), ValueError, RecursionError...), all shown on the node
    except Exception as exc:
        raise ParseError(f"Cannot read '{text.strip()}': {exc}") from None
    if isinstance(result, bool | int | float):
        result = sp.sympify(result)
    if not isinstance(result, sp.Basic) or isinstance(result, sp.FunctionClass):
        raise ParseError(f"'{text.strip()}' is not an expression")
    return result


def parse_equation(
    text: str,
    linked: dict[str, Any] | None = None,
    variable: str | None = None,
) -> sp.Eq:
    """An equation ``lhs = rhs`` from text; an expression alone means ``= 0``."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("Enter an equation, such as E*I*y''''(x) = -w")
    sides = _EQUALS.split(text)
    if len(sides) > 2:
        raise ParseError("An equation has one '=' sign")
    lhs = parse_expression(sides[0], linked, variable)
    rhs = parse_expression(sides[1], linked, variable) if len(sides) == 2 else sp.Integer(0)
    return sp.Eq(lhs, rhs, evaluate=False)


def split_assignments(text: str, bare: bool = False) -> list[tuple[str, str]]:
    """``"E = 200 GPa; L = 6 m"`` (or one per line) as ``[("E", "200 GPa"), ...]``.
    Blank lines and ``#`` comments are skipped. ``bare``: a line that is just
    a name stands for ``name = name`` (a constant, in Values)."""
    pairs = []
    for raw in re.split(r"[;\n]", text or ""):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, value = line.partition("=")
        if bare and not sep and line.isidentifier():
            sep, value = "=", line
        name = _RENAMED.get(name.strip(), name.strip())
        if not sep or not value.strip():
            raise ParseError(f"'{line}' is not 'name = value'")
        if not name.isidentifier() or name.startswith("_") or keyword.iskeyword(name):
            raise ParseError(f"'{name}' is not a valid symbol name")
        pairs.append((name, value.strip()))
    return pairs
