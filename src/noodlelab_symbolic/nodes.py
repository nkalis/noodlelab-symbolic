"""Symbolic nodes: expressions, equations, calculus, solvers, and evaluation
with units."""

from __future__ import annotations

import keyword
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
import pint
import sympy as sp
from numpy.typing import NDArray

from noodlelab import Param, Quantity, RunContext, node, warning
from noodlelab.core.units import dims_or_none, is_quantity, parse, ureg
from noodlelab.reports.math import TypstMath

from .parse import _RENAMED, ParseError, parse_equation, parse_expression, split_assignments
from .types import Equation, Expression, SymbolValues, functions_of, symbols_of
from .typst import typst_math

EXPRESSION_HELP = (
    "SymPy notation: w*L^2/12, sqrt(x), sin(2 pi f t). Juxtaposition multiplies (2 x, E I). "
    "E and I are symbols; e is Euler's number. {a} … {d} insert linked expressions."
)
Text = Annotated[str, Param(multiline=True, description=EXPRESSION_HELP)]
Variable = Annotated[str, Param(options_from="expression.symbols")]
Linked = Expression | None


def _problem(fn: Any, *args: Any, **kwargs: Any) -> str | None:
    try:
        fn(*args, **kwargs)
    except ParseError as exc:
        return str(exc)
    return None


def _split_top_level(text: str) -> list[str]:
    """Split at commas, semicolons and newlines outside brackets."""
    parts, depth, current = [], 0, ""
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch in ",;\n" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    return [p.strip() for p in parts if p.strip()]


# --- building expressions --------------------------------------------------------------------


@node(category="Symbolic", title="Expression", fold=True)
def expression(
    text: Text = "w*L^2/12",
    a: Linked = None,
    b: Linked = None,
    c: Linked = None,
    d: Linked = None,
) -> Expression:
    """A symbolic expression typed as text. Names that are not functions
    become symbols; ``{a}`` … ``{d}`` insert the linked expressions, so
    ``-E*I*diff({a}, x, 2)`` differentiates whatever is linked to a."""
    result = parse_expression(text, {"a": a, "b": b, "c": c, "d": d})
    if isinstance(result, sp.Rel):
        raise ParseError("This is an equation: use the Equation node")
    return result


@expression.check
def _check_expression(
    text: str = "", a: Any = None, b: Any = None, c: Any = None, d: Any = None
) -> str | None:
    return _problem(expression, text, a, b, c, d)


@node(category="Symbolic", title="Equation", fold=True)
def equation(
    text: Text = "E*I*y''''(x) = -w",
    a: Linked = None,
    b: Linked = None,
    c: Linked = None,
    d: Linked = None,
) -> Equation:
    """An equation, ``lhs = rhs``, for Solve and Solve ODE. Primes are
    derivatives: ``y''(x)`` is d²y/dx². Without ``=``, the expression equals zero."""
    return parse_equation(text, {"a": a, "b": b, "c": c, "d": d})


@equation.check
def _check_equation(
    text: str = "", a: Any = None, b: Any = None, c: Any = None, d: Any = None
) -> str | None:
    return _problem(equation, text, a, b, c, d)


SimplifyMethod = Literal["simplify", "factor", "expand", "cancel", "together", "trigsimp"]


@node(category="Symbolic", title="Simplify")
def simplify(expression: Expression, method: SimplifyMethod = "simplify") -> Expression:
    """Rewrite an expression: simplify (SymPy's heuristics), factor, expand
    products and powers, cancel common factors, put over one denominator
    (together), or simplify trigonometric functions."""
    fn = {
        "simplify": sp.simplify,
        "factor": sp.factor,
        "expand": sp.expand,
        "cancel": sp.cancel,
        "together": sp.together,
        "trigsimp": sp.trigsimp,
    }[method]
    return fn(expression)


@node(category="Symbolic", title="Substitute")
def substitute(
    expression: Expression,
    substitutions: Annotated[
        str, Param(multiline=True, description="name = expression; one per line or ; separated")
    ] = "x = L/2",
) -> Expression:
    """Replace symbols by expressions: ``x = L/2`` gives the value at mid-span,
    still in symbols. To put in numbers with units, use Evaluate."""
    pairs = {
        sp.Symbol(name): parse_expression(value) for name, value in split_assignments(substitutions)
    }
    return expression.subs(pairs)


@substitute.check
def _check_substitute(substitutions: str = "") -> str | None:
    try:
        for _, value in split_assignments(substitutions):
            parse_expression(value)
    except ParseError as exc:
        return str(exc)
    return None


# --- calculus --------------------------------------------------------------------------------


@node(category="Symbolic", title="Differentiate")
def differentiate(
    expression: Expression,
    variable: Variable = "x",
    order: Annotated[int, Param(min=1, max=10)] = 1,
) -> Expression:
    """The derivative with respect to ``variable``, ``order`` times."""
    return sp.diff(expression, sp.Symbol(variable), order)


@node(category="Symbolic", title="Integrate")
def integrate(
    expression: Expression,
    variable: Variable = "x",
    lower: Annotated[str, Param(description="Empty for an indefinite integral")] = "",
    upper: Annotated[str, Param(description="Empty for an indefinite integral")] = "",
) -> Expression:
    """The integral with respect to ``variable``: indefinite (without a
    constant), or between two limits, which may be symbols such as 0 and L."""
    x = sp.Symbol(variable)
    if not lower.strip() and not upper.strip():
        return sp.integrate(expression, x)
    return sp.integrate(expression, (x, parse_expression(lower), parse_expression(upper)))


@integrate.check
def _check_integrate(lower: str = "", upper: str = "") -> str | None:
    if bool(lower.strip()) != bool(upper.strip()):
        return "Give both limits, or neither for an indefinite integral"
    for text in (lower, upper):
        if text.strip() and (problem := _problem(parse_expression, text)):
            return problem
    return None


# --- solvers ---------------------------------------------------------------------------------


class Solution(NamedTuple):
    solution: Expression
    solutions: list[Expression]
    count: int


@node(category="Symbolic", title="Solve")
def solve(
    equation: Equation | Expression,
    unknown: Annotated[str, Param(options_from="equation.symbols")] = "x",
    pick: Annotated[int, Param(min=0, description="Which solution, when there are several")] = 0,
) -> Solution:
    """Solve an equation (or ``expression = 0``) for one unknown, in symbols.
    ``solution`` is the solution numbered ``pick``, ``solutions`` all of them."""
    x = sp.Symbol(unknown)
    found = sp.solve(equation, x, dict=False)
    if not found:
        raise ValueError(f"No solution for {unknown}")
    found = [sp.sympify(s) for s in found]
    if pick >= len(found):
        raise ValueError(f"There are {len(found)} solutions: pick 0 to {len(found) - 1}")
    return Solution(found[pick], found, len(found))


@node(category="Symbolic", title="Solve ODE")
def solve_ode(
    equation: Equation,
    function: Annotated[str, Param(options_from="equation.functions")] = "y",
    variable: Annotated[str, Param(options_from="equation.symbols")] = "x",
    conditions: Annotated[
        str,
        Param(
            multiline=True,
            description="Boundary or initial conditions, comma separated: y(0) = 0, y'(L) = 0",
        ),
    ] = "",
) -> Expression:
    """Solve an ordinary differential equation for ``function(variable)``.
    With enough conditions the constants of integration are found; without,
    they stay as C1, C2, ... The result is the right-hand side, y(x)."""
    x = sp.Symbol(variable)
    f = sp.Function(function)(x)
    ics = {}
    for text in _split_top_level(conditions):
        cond = parse_equation(text, variable=variable)
        ics[cond.lhs] = cond.rhs
    result = sp.dsolve(equation, f, ics=ics or None)
    if isinstance(result, list):
        result = result[0]
    return result.rhs


@solve_ode.check
def _check_solve_ode(conditions: str = "", variable: str = "x") -> str | None:
    for text in _split_top_level(conditions):
        try:
            cond = parse_equation(text, variable=variable)
        except ParseError as exc:
            return f"Condition '{text}': {exc}"
        if cond.rhs == 0 and cond.lhs == 0:
            return f"Condition '{text}' says nothing"
    return None


# --- numbers, with units ---------------------------------------------------------------------


@node(category="Symbolic", title="Values", fold=True)
def values(
    text: Annotated[
        str,
        Param(
            multiline=True,
            description="name = value with a unit, one per line: E = 200 GPa, L = 6 m, n = 3",
        ),
    ] = "E = 200 GPa\nL = 6 m",
) -> SymbolValues:
    """Numbers, usually with units, for the symbols of an expression. A value
    without a unit is a plain number."""
    out = SymbolValues()
    for name, value in split_assignments(text):
        try:
            out[name] = float(value)
        except ValueError:
            try:
                out[name] = parse(value)
            # Pint's parser raises anything from AssertionError to TokenError on bad text
            except Exception as exc:
                raise ParseError(f"{name}: cannot read '{value}' ({exc})") from None
    return out


@values.check
def _check_values(text: str = "") -> str | None:
    return _problem(values, text)


@node(category="Symbolic", title="Set Value", fold=True)
def set_value(
    values: SymbolValues | None = None,
    name: str = "x",
    value: Quantity | float = 0.0,
) -> SymbolValues:
    """Add a linked value (a quantity or a number from elsewhere in the graph)
    to a set of values, under ``name``."""
    out = SymbolValues(values or {})
    out[_symbol_name(name)] = value
    return out


def _symbol_name(name: str) -> str:
    """``name`` as the parser spells the symbol (``lambda`` is ``lamda``)."""
    return _RENAMED.get(name.strip(), name.strip())


@set_value.check
def _check_set_value(name: str = "x") -> str | None:
    key = _symbol_name(name)
    if not key.isidentifier() or key.startswith("_") or keyword.iskeyword(key):
        return f"'{name}' is not a valid symbol name"
    return None


def _unit_problem(unit: str) -> str | None:
    if unit.strip() and dims_or_none(unit) is None:
        return f"Unknown unit '{unit}'"
    return None


def _call(expr: sp.Basic, vals: dict[str, Any]) -> Any:
    """``expr`` with ``vals`` put in, computed with NumPy: quantities keep
    their units, and Pint refuses to add metres to seconds."""
    if functions := functions_of(expr):
        raise ValueError(
            f"The expression still has the unknown function(s) {', '.join(functions)}: "
            "solve for them first"
        )
    missing = [s for s in symbols_of(expr) if s not in vals]
    if missing:
        raise KeyError(f"No value for {', '.join(missing)}")
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    fn = sp.lambdify(syms, expr, modules="numpy")
    return fn(*(vals[s.name] for s in syms))


def _in_unit(result: Any, unit: str) -> Any:
    reg = ureg()
    if not is_quantity(result):
        return reg.Quantity(result, unit.strip() or None)
    if unit.strip():
        return result.to(unit.strip())
    return result.to_reduced_units()


@node(category="Symbolic", title="Evaluate")
def evaluate(
    expression: Expression,
    values: SymbolValues,
    unit: Annotated[str, Param(description="Unit of the result; empty: worked out")] = "",
) -> Quantity:
    """Put numbers with units into an expression. Units are carried through,
    so the result has the right dimension: a moment from kN/m and m comes out
    in kN·m. An array value (a quantity holding many numbers) gives an array."""
    return _in_unit(_call(expression, values), unit)


@evaluate.check
def _check_evaluate(
    expression: Expression | None = None, values: SymbolValues | None = None, unit: str = ""
) -> str | None:
    if problem := _unit_problem(unit):
        return problem
    if expression is None or values is None:
        return None
    try:
        result = _in_unit(_call(expression, values), unit)
    except (KeyError, ValueError) as exc:
        return str(exc).strip("'\"")
    except pint.DimensionalityError as exc:
        return str(exc)
    # whatever the run would raise (1/x at x = 0 with plain numbers...), shown while editing
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if not unit.strip() and not result.dimensionless:
        return warning(f"The result is in {result.units:~P}: set a unit to be sure")
    return None


class Sweep(NamedTuple):
    x: Quantity
    value: Quantity
    peak: Quantity
    peak_at: Quantity
    minimum: Quantity
    maximum: Quantity


@node(category="Symbolic", title="Evaluate Over Range")
def evaluate_range(
    expression: Expression,
    values: SymbolValues,
    variable: Variable = "x",
    start: Annotated[str, Param(description="An expression in the values: 0, L/2")] = "0",
    stop: Annotated[str, Param(description="An expression in the values: L")] = "L",
    points: Annotated[int, Param(min=2, max=100_000)] = 201,
    unit: Annotated[str, Param(description="Unit of the result; empty: worked out")] = "",
    variable_unit: Annotated[str, Param(description="Unit of x; empty: that of start")] = "",
) -> Sweep:
    """Evaluate an expression at evenly spaced values of one variable, such as
    the deflection along a beam. ``peak`` is the value largest in size (with
    its sign) and ``peak_at`` where it occurs. Array quantities plot through
    Magnitude (Array)."""
    lo, hi = (_call(parse_expression(t), values) for t in (start, stop))
    reg = ureg()
    # a bare number (usually 0) takes the unit of the other end, or variable_unit
    unit_of = next((v.units for v in (lo, hi) if is_quantity(v)), variable_unit.strip() or None)
    lo, hi = (v if is_quantity(v) else reg.Quantity(v, unit_of) for v in (lo, hi))
    hi = hi.to(lo.units)
    x = reg.Quantity(np.linspace(float(lo.magnitude), float(hi.magnitude), points), lo.units)
    if variable_unit.strip():
        x = x.to(variable_unit.strip())
    y = _call(expression, {**values, variable: x})
    if not is_quantity(y) or np.ndim(y.magnitude) == 0:  # does not depend on the variable
        y = y * np.ones(points)
    y = _in_unit(y, unit)
    mag = np.asarray(y.magnitude, dtype=np.float64)
    i = int(np.nanargmax(np.abs(mag)))
    return Sweep(
        x=x,
        value=y,
        peak=y[i],
        peak_at=x[i],
        minimum=y[int(np.nanargmin(mag))],
        maximum=y[int(np.nanargmax(mag))],
    )


@evaluate_range.check
def _check_evaluate_range(
    values: SymbolValues | None = None,
    start: str = "0",
    stop: str = "L",
    unit: str = "",
    variable_unit: str = "",
) -> str | None:
    for u in (unit, variable_unit):
        if problem := _unit_problem(u):
            return problem
    for text in (start, stop):
        if problem := _problem(parse_expression, text):
            return problem
        if values is not None:
            try:
                _call(parse_expression(text), values)
            except (KeyError, ValueError) as exc:
                return f"{text}: {str(exc).strip(chr(39))}"
    return None


# --- iteration ---------------------------------------------------------------------------------
#
# A graph runs each node once, from inputs to outputs, and a link that would
# close a loop is refused ("the graph contains a cycle"). A loop is therefore
# the inside of a node: Iterate below is a while loop, and its history makes
# every pass visible.


class Iteration(NamedTuple):
    result: Quantity
    iterations: int
    converged: bool
    step: NDArray[np.float64]
    history: NDArray[np.float64]
    change: NDArray[np.float64]


IterateMethod = Literal["fixed point", "newton"]


def _step_expression(expression: sp.Basic, variable: str, method: str) -> sp.Basic:
    """The expression each pass evaluates to get the next x."""
    x = next((s for s in expression.free_symbols if s.name == variable), sp.Symbol(variable))
    if method == "newton":
        f = expression.lhs - expression.rhs if isinstance(expression, Equation) else expression
        return x - f / sp.diff(f, x)
    if isinstance(expression, Equation):
        if expression.lhs != x:
            raise ValueError(f"For a fixed point, write the equation as {variable} = g({variable})")
        return expression.rhs
    return expression


def _plain(value: Any) -> float:
    """A number without its unit (dimensionless ratios are reduced first)."""
    if is_quantity(value):
        value = value.to_reduced_units().magnitude
    return float(np.real_if_close(value))


@node(category="Symbolic", title="Iterate")
def iterate(
    expression: Expression | Equation,
    ctx: RunContext,
    values: SymbolValues | None = None,
    variable: Variable = "x",
    method: IterateMethod = "fixed point",
    start: Annotated[
        str, Param(description="The first guess, an expression in the values: 0.02, L/2")
    ] = "1",
    digits: Annotated[
        int,
        Param(
            widget="number",
            min=1,
            max=15,
            description="Stop when a pass changes x by less than 10^-digits of x",
        ),
    ] = 10,
    max_iterations: Annotated[int, Param(widget="number", min=1, max=100_000)] = 100,
    unit: Annotated[str, Param(description="Unit of the result; empty: worked out")] = "",
) -> Iteration:
    """Repeat a step until the answer stops changing: a while loop in one node.

    * **fixed point**: x ← g(x), where the expression is g(x), or the
      equation is written x = g(x).
    * **newton**: solves expression = 0 (or lhs = rhs) with
      x ← x − f(x)/f′(x). The derivative is worked out symbolically.

    Stops when a pass changes x by at most 10^-``digits``·|x|, so x is good
    to about that many significant digits, or after ``max_iterations`` passes
    (``converged`` is then false). ``history`` is x
    after each pass and ``change`` its relative change, for a convergence plot.
    """
    tolerance = 10.0**-digits
    vals = dict(values or {})
    step_expr = _step_expression(expression, variable, method)
    x = _call(parse_expression(start), vals)
    x = x if is_quantity(x) else float(x)  # floats, not exact integers that grow without bound
    history: list[float] = []
    changes: list[float] = []
    converged = False
    with np.errstate(all="ignore"):
        for n in range(1, max_iterations + 1):
            try:
                new = _call(step_expr, {**vals, variable: x})
                change = abs(_plain((new - x) / new)) if _plain(new) != 0 else abs(_plain(new - x))
            except (OverflowError, ZeroDivisionError):
                new, change = float("inf"), float("inf")
            if not np.isfinite(_plain(new)) or not np.isfinite(change):
                raise ValueError(
                    f"Pass {n} gave {variable} = {_plain(new):g}: the iteration diverged. "
                    "Try another start or method"
                )
            history.append(_plain(new))
            changes.append(change)
            x = new
            if change <= tolerance:
                converged = True
                break
    n = len(history)
    ctx.log(
        f"{method}: converged in {n} passes, {variable} = {_plain(x):.10g}"
        if converged
        else f"{method}: not converged after {n} passes (last change {changes[-1]:.2g})"
    )
    return Iteration(
        result=_in_unit(x, unit),
        iterations=n,
        converged=converged,
        step=np.arange(1, n + 1, dtype=np.float64),
        history=np.asarray(history, dtype=np.float64),
        change=np.asarray(changes, dtype=np.float64),
    )


@iterate.check
def _check_iterate(
    expression: Expression | Equation | None = None,
    variable: str = "x",
    method: str = "fixed point",
    start: str = "1",
    unit: str = "",
) -> str | None:
    if problem := _unit_problem(unit) or _problem(parse_expression, start):
        return problem
    if expression is None:
        return None
    if variable not in symbols_of(expression):
        return f"The expression has no symbol '{variable}' to iterate on"
    try:
        _step_expression(expression, variable, method)
    except ValueError as exc:
        return str(exc)
    return None


# --- reports ---------------------------------------------------------------------------------


@node(category="Symbolic", title="Expression To Math", fold=True)
def to_math(
    expression: Expression | Equation,
    left: Annotated[str, Param(description="Optional left-hand side, e.g. M(x) or sigma_max")] = "",
) -> str:
    """Typst math for the report's Add Equation node: the expression as it
    would be typeset, optionally as ``left = expression``."""
    math = typst_math(expression)
    if left.strip():
        math = f"{typst_math(parse_expression(left))} = {math}"
    return TypstMath(math)  # a str, previewed typeset


@to_math.check
def _check_to_math(left: str = "") -> str | None:
    return _problem(parse_expression, left) if left.strip() else None
