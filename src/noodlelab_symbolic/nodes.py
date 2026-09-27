"""Symbolic nodes: expressions, equations, calculus, solvers, and evaluation
with units."""

from __future__ import annotations

import keyword
import math
import re
from functools import lru_cache
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
import pint
import sympy as sp
from numpy.typing import NDArray

from noodlelab import Param, Quantity, RunContext, node, warning
from noodlelab.core import constants, uncertainty
from noodlelab.core.units import dims_or_none, is_quantity, parse, ureg
from noodlelab.reports.math import TypstMath

from .parse import _RENAMED, ParseError, parse_equation, parse_expression, split_assignments
from .types import (
    Equation,
    EquationSystem,
    Expression,
    SymbolicMatrix,
    SymbolValues,
    functions_of,
    symbols_of,
)
from .typst import typst_math

EXPRESSION_HELP = (
    "SymPy notation: w*L^2/12, sqrt(x), sin(2 pi f t). Juxtaposition multiplies (2 x, E I). "
    "E and I are symbols; e is Euler's number. {a} … {d} insert linked expressions. "
    "Other constants (c, g0, k_B...) are symbols until a Values line names them: g = g0."
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


def _parse_lines(text: str, linked: dict[str, Any]) -> list[Equation]:
    """One equation per line (or per ``;``), skipping blank lines and ``#`` comments."""
    found = []
    for i, raw in enumerate(re.split(r"[;\n]", text or ""), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            found.append(parse_equation(line, linked))
        except ParseError as exc:
            raise ParseError(f"Line {i}: {exc}") from None
    if not found:
        raise ParseError("Enter the equations, one per line, such as x + y = 3")
    return found


@node(category="Symbolic", title="Equations", fold=True)
def equations(
    text: Annotated[
        str, Param(multiline=True, description=EXPRESSION_HELP + " One equation per line.")
    ] = "x + y = 3\nx - y = 1",
    a: Linked = None,
    b: Linked = None,
    c: Linked = None,
    d: Linked = None,
) -> EquationSystem:
    """Several equations, one per line, to be solved together with Solve
    System or Solve Numerically, or written as a matrix with Linear System:
    one balance equation per mass of a structure, say. Lines starting with
    ``#`` are comments."""
    return EquationSystem(_parse_lines(text, {"a": a, "b": b, "c": c, "d": d}))


@equations.check
def _check_equations(
    text: str = "", a: Any = None, b: Any = None, c: Any = None, d: Any = None
) -> str | None:
    return _problem(equations, text, a, b, c, d)


def _immutable(result: Any) -> Any:
    """SymPy may hand back a mutable matrix, which is no expression: freeze it."""
    return sp.ImmutableMatrix(result) if isinstance(result, sp.MatrixBase) else result


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
    if isinstance(expression, sp.MatrixBase):
        return _immutable(expression.applyfunc(fn))
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
    return _immutable(expression.subs(pairs))


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
    return _immutable(sp.diff(expression, sp.Symbol(variable), order))


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
        return _immutable(sp.integrate(expression, x))
    limits = (x, parse_expression(lower), parse_expression(upper))
    return _immutable(sp.integrate(expression, limits))


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
    if isinstance(equation, sp.MatrixBase):
        raise TypeError("This is a matrix: use Linear System or Solve System")
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


def _system(equations: Any) -> list[Equation]:
    return list(equations) if isinstance(equations, tuple) else [equations]


def _unknowns(text: str, eqs: list[Equation]) -> list[sp.Symbol]:
    """The symbols named in ``text`` ("x1, x2"), as the equations have them."""
    have = {s.name: s for e in eqs for s in e.free_symbols if isinstance(s, sp.Symbol)}
    names = [_symbol_name(n) for n in _split_top_level(text)]
    if not names:
        raise ValueError("Name the unknowns, comma separated: x1, x2")
    missing = [n for n in names if n not in have]
    if missing:
        listed = ", ".join(sorted(have)) or "none"
        raise ValueError(f"{', '.join(missing)}: not in the equations, which have {listed}")
    return [have[n] for n in names]


def _tidy(expr: sp.Basic) -> sp.Basic:
    """A solution in its most readable form: F(k1 + k2)/(k1(k1 + 2 k2)), not expanded."""
    try:
        return sp.factor(expr)
    except Exception:  # factor gives up on some expressions: keep them as found
        return expr


class SystemSolution(NamedTuple):
    answers: EquationSystem
    solution: Expression
    solutions: SymbolicMatrix
    count: int


@node(category="Symbolic", title="Solve System")
def solve_system(
    equations: EquationSystem | Equation,
    unknowns: Annotated[str, Param(description="The unknowns, comma separated: x1, x2")] = "x1, x2",
    unknown: Annotated[
        str, Param(description="The unknown that solution gives, such as x2; empty: the first")
    ] = "",
    pick: Annotated[
        int, Param(min=0, description="Which solution, when there are several (from 0)")
    ] = 0,
) -> SystemSolution:
    """Solve several equations together for several unknowns, in symbols.
    ``answers`` shows them all (x1 = …, x2 = …), ``solution`` is one of them
    for Evaluate, and ``solutions`` is the column of all of them, in the order
    of ``unknowns``. An unknown the equations leave free stays a symbol."""
    eqs = _system(equations)
    syms = _unknowns(unknowns, eqs)
    names = [s.name for s in syms]
    found = sp.solve(eqs, syms, dict=True)
    if not found:
        raise ValueError(f"The equations have no solution for {', '.join(names)}")
    if pick >= len(found):
        raise ValueError(f"There are {len(found)} solutions: pick 0 to {len(found) - 1}")
    values = [_tidy(found[pick].get(s, s)) for s in syms]
    which = _symbol_name(unknown) or names[0]
    if which not in names:
        raise ValueError(f"{which} is not one of the unknowns ({', '.join(names)})")
    answers = EquationSystem(sp.Eq(s, v, evaluate=False) for s, v in zip(syms, values, strict=True))
    return SystemSolution(
        answers, values[names.index(which)], sp.ImmutableMatrix(values), len(found)
    )


@solve_system.check
def _check_solve_system(equations: Any = None, unknowns: str = "x1, x2", unknown: str = "") -> Any:
    names = [_symbol_name(n) for n in _split_top_level(unknowns)]
    if not names:
        return "Name the unknowns, comma separated: x1, x2"
    if unknown.strip() and _symbol_name(unknown) not in names:
        return f"{unknown} is not one of the unknowns ({', '.join(names)})"
    if equations is None:
        return None
    eqs = _system(equations)
    try:
        _unknowns(unknowns, eqs)
    except ValueError as exc:
        return str(exc)
    if len(names) > len(eqs):
        return warning(
            f"{len(names)} unknowns but {len(eqs)} equation(s): some unknowns will stay free"
        )
    return None


class LinearForm(NamedTuple):
    A: SymbolicMatrix
    b: SymbolicMatrix
    unknowns: list[str]


@node(category="Symbolic", title="Linear System")
def linear_system(
    equations: EquationSystem | Equation,
    unknowns: Annotated[str, Param(description="The unknowns, comma separated: x1, x2")] = "x1, x2",
) -> LinearForm:
    """Write linear equations as a matrix equation, A x = b, with x the
    unknowns in the order given: the equilibrium of masses on springs gives
    the stiffness matrix K and the force vector F. Evaluate Matrix puts
    numbers into both, and Solve Linear System solves it."""
    from sympy.solvers.solveset import NonlinearError

    eqs = _system(equations)
    syms = _unknowns(unknowns, eqs)
    try:
        a, b = sp.linear_eq_to_matrix([e.lhs - e.rhs for e in eqs], syms)
    except NonlinearError as exc:
        names = ", ".join(s.name for s in syms)
        why = " ".join(str(exc).split())
        raise ValueError(f"The equations are not linear in {names}: {why}") from None
    return LinearForm(sp.ImmutableMatrix(a), sp.ImmutableMatrix(b), [s.name for s in syms])


@linear_system.check
def _check_linear_system(equations: Any = None, unknowns: str = "x1, x2") -> str | None:
    if equations is None:
        return None
    try:
        _unknowns(unknowns, _system(equations))
    except ValueError as exc:
        return str(exc)
    return None


# --- numbers, with units ---------------------------------------------------------------------


@node(category="Symbolic", title="Values", fold=True)
def values(
    text: Annotated[
        str,
        Param(
            multiline=True,
            description="name = value with a unit, one per line: E = 200 GPa, L = 6 m, n = 3. "
            "A constant by name: g = g0, or just the line c",
        ),
    ] = "E = 200 GPa\nL = 6 m",
) -> SymbolValues:
    """Numbers, usually with units, for the symbols of an expression. A value
    without a unit is a plain number. A value can also name a constant, with
    its unit and uncertainty: ``g = g0`` is standard gravity, and a line that
    is just a name, such as ``c``, means ``c = c``. Constants are looked up
    before units, so ``h`` is Planck's constant here, not an hour: write
    ``t = 1 h`` for an hour. A name that means something else as a unit is
    flagged with a warning."""
    out = SymbolValues()
    for name, value in split_assignments(text, bare=True):
        out[name] = _value(name, value)
    return out


def _value(name: str, value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        pass
    if value.isidentifier() and (const := constants.get(value)) is not None:
        return const.value_of()
    try:
        if uncertainty.is_uncertain_text(value):
            # Pint's own reader gets "9.81(2)" wrong (± 0.2) and refuses "±"; ours
            # also names the value after its line, for uncertainty budgets
            x = uncertainty.parse(value, name=name)
            return x if is_quantity(x) else ureg().Quantity(x, "")
        return parse(value)
    # Pint's parser raises anything from AssertionError to TokenError on bad text
    except Exception as exc:
        if value.isidentifier():  # meant as a constant, most likely
            raise ParseError(
                f"{name}: '{value}' is neither a number with a unit nor a known constant"
                f"{constants.suggestion(value)}"
            ) from None
        raise ParseError(f"{name}: cannot read '{value}' ({exc})") from None


@values.uses_constants
def _values_constants(text: str = "") -> list[str]:
    try:
        return [v for _, v in split_assignments(text, bare=True) if v.isidentifier()]
    except ParseError:
        return []


@values.check
def _check_values(text: str = ""):
    if problem := _problem(values, text):
        return problem
    clashes = [_clash(name, value) for name, value in split_assignments(text, bare=True)]
    clashes = [c for c in clashes if c]
    return warning("; ".join(clashes)) if clashes else None


def _clash(name: str, value: str) -> str | None:
    """A bare name that is a constant here but a different unit to Pint (``h``:
    Planck's constant, not an hour), said plainly, since either may be meant.
    Names that mean the same either way (``c``, ``g0``, ``atm``) are fine, and
    so is a line naming the constant itself (``h``, or ``h = h``)."""
    const = constants.get(value) if value.isidentifier() and value != name else None
    if const is None:
        return None
    try:
        unit = ureg().Quantity(1, value)
    except Exception:  # not a unit at all: no ambiguity
        return None
    try:
        same = math.isclose(unit.to(const.unit or "").magnitude, const.value, rel_tol=1e-6)
    except Exception:  # a different dimension
        same = False
    if same:
        return None
    return (
        f"{name} = {value} takes the constant {value} ({const.title or const.name}, "
        f"{const.display()}), not the unit {unit.units}: write '{name} = 1 {value}' for the unit"
    )


@node(category="Symbolic", title="Set Value", fold=True, vectorized=True)
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
    if expr.has(sp.Sum, sp.Product):
        expr = _expand_sums(expr, vals)
    if functions := functions_of(expr):
        raise ValueError(
            f"The expression still has the unknown function(s) {', '.join(functions)}: "
            "solve for them first"
        )
    missing = [s for s in symbols_of(expr) if s not in vals]
    if missing:
        raise KeyError(f"No value for {', '.join(missing)}{constants.hint(missing)}")
    names, fn = _compiled(expr)
    args = {s: vals[s] for s in names}
    if any(uncertainty.is_uncertain(v) for v in args.values()):
        # NumPy's functions cannot take uncertain numbers: propagate their uncertainty
        # the way every node does (GUM, first order), which keeps correlations too
        return uncertainty.lift(lambda **kw: fn(*(kw[s] for s in names)), args, set(args))
    return fn(*args.values())


def _expand_sums(expr: sp.Basic, vals: dict[str, Any]) -> sp.Basic:
    """Sums and products written out term by term. NumPy has nothing for a sum
    over a symbolic index, and SymPy's closed forms (``harmonic``...) have no
    NumPy function either, so the limits get their values first."""
    loops = expr.atoms(sp.Sum, sp.Product)
    bounds = {
        s for loop in loops for _, lo, hi in loop.limits for s in lo.free_symbols | hi.free_symbols
    }
    subs = {}
    for sym in bounds:
        if sym.name not in vals:
            continue  # the "No value for ..." below says so
        v = vals[sym.name]
        v = v.m_as("") if is_quantity(v) else v
        if not float(v).is_integer():
            raise ValueError(f"{sym.name} = {v} is a limit of a sum or product: it must be whole")
        subs[sym] = int(v)
    return expr.subs(subs).doit()


# lambdify writes and compiles Python source: ~1 ms, far more than the call (Iterate
# calls it once per pass, Monte Carlo once per trial). SymPy compares expressions
# structurally, so x + 1 and x + 1.0 are different keys.
@lru_cache(maxsize=256)
def _compiled(expr: sp.Basic) -> tuple[tuple[str, ...], Any]:
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    return tuple(s.name for s in syms), sp.lambdify(syms, expr, modules=[_ELEMENTWISE, "numpy"])


def _elementwise(fn: Any) -> Any:
    """A function of one number (from :mod:`math`) that also takes arrays and
    dimensionless quantities, as NumPy's own functions do."""
    each = np.vectorize(fn, otypes=[np.float64])

    def call(x: Any) -> Any:
        if is_quantity(x):
            x = x.m_as("")  # a unit that does not cancel is an error here
        out = each(x)
        return float(out) if np.ndim(out) == 0 else out

    return call


# functions NumPy lacks: lambdify would call math's, which take one number only
_ELEMENTWISE = {"erf": _elementwise(math.erf), "erfc": _elementwise(math.erfc)}


def _in_unit(result: Any, unit: str) -> Any:
    reg = ureg()
    if not is_quantity(result):
        return reg.Quantity(result, unit.strip() or None)
    if unit.strip():
        return result.to(unit.strip())
    return result.to_reduced_units()


@node(category="Symbolic", title="Evaluate", vectorized=True)
def evaluate(
    expression: Expression,
    values: SymbolValues,
    unit: Annotated[str, Param(description="Unit of the result; empty: worked out")] = "",
) -> Quantity:
    """Put numbers with units into an expression. Units are carried through,
    so the result has the right dimension: a moment from kN/m and m comes out
    in kN·m. An array value (a quantity holding many numbers) gives an array."""
    if isinstance(expression, sp.MatrixBase):
        raise TypeError(MATRIX_TO_EVALUATE)
    try:
        return _in_unit(_call(expression, values), unit)
    except pint.OffsetUnitCalculusError:
        raise TypeError(_offset_problem(values)) from None


MATRIX_TO_EVALUATE = "This is a matrix: use Evaluate Matrix"


def _offset_problem(values: dict[str, Any]) -> str:
    """Pint's "ambiguous operation with offset unit" names units, not the
    value to fix. °C and °F are offset scales: 2 × 20 °C is not 40 °C, so a
    formula needs kelvin, or a temperature difference."""
    offset = [
        f"{name} ({v:~P})"
        for name, v in values.items()
        if is_quantity(v) and not getattr(v, "_is_multiplicative", True)
    ]
    which = ", ".join(offset) or "a value"
    verb = "are temperatures" if len(offset) > 1 else "is a temperature"
    return (
        f"{which} {verb} on an offset scale, which a formula cannot use (SymPy writes "
        "even a - b as a + (-1)·b): convert to K first (Convert Units), or give a "
        "temperature difference in delta_degC"
    )


@evaluate.check
def _check_evaluate(
    expression: Expression | None = None, values: SymbolValues | None = None, unit: str = ""
) -> str | None:
    if problem := _unit_problem(unit):
        return problem
    if isinstance(expression, sp.MatrixBase):
        return MATRIX_TO_EVALUATE
    if expression is None or values is None:
        return None
    try:
        result = _in_unit(_call(expression, values), unit)
    except (KeyError, ValueError) as exc:
        return str(exc).strip("'\"")
    except pint.OffsetUnitCalculusError:
        return _offset_problem(values)
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


# --- matrices and numerical solving ------------------------------------------------------------


def _nominal_values(vals: dict[str, Any]) -> dict[str, Any]:
    return {k: uncertainty.nominal(v) for k, v in vals.items()}


def _evaluate_matrix(matrix: Any, vals: dict[str, Any], unit: str) -> Any:
    """Every entry evaluated, in one unit: ``unit``, or else that of the first
    entry with one. Zeros fit any unit; anything else that does not is an error
    naming both entries."""
    if not isinstance(matrix, sp.MatrixBase):
        raise TypeError("This is not a matrix: use Evaluate")
    reg = ureg()
    vals = _nominal_values(vals)
    rows, cols = matrix.shape
    entries: dict[tuple[int, int], Any] = {}
    for i in range(rows):
        for j in range(cols):
            v = _call(matrix[i, j], vals)
            if np.ndim(v.magnitude if is_quantity(v) else v) != 0:
                raise ValueError(f"Entry ({i + 1}, {j + 1}) is an array: an entry is one number")
            entries[i, j] = v
    target = reg.Unit(unit.strip()) if unit.strip() else None
    first = None
    if target is None:
        with_unit = [(k, v) for k, v in entries.items() if is_quantity(v) and v.magnitude != 0]
        if with_unit:
            first, q = with_unit[0]
            target = q.to_reduced_units().units
        else:
            target = reg.dimensionless
    out = np.zeros((rows, cols))
    for (i, j), v in entries.items():
        where = f"({i + 1}, {j + 1})"
        if not is_quantity(v):
            v = reg.Quantity(v, "")
        m = complex(v.magnitude)
        if m.imag:
            raise ValueError(f"Entry {where} is complex ({m:g})")
        if m.real == 0:
            continue  # a zero fits any unit
        try:
            out[i, j] = v.to(target).magnitude
        except pint.DimensionalityError:
            than = f"entry ({first[0] + 1}, {first[1] + 1}) is in" if first else "the unit is"
            got = f"in {v.units:~P}" if f"{v.units:~P}" else "a plain number"
            raise ValueError(
                f"Entry {where} is {got}, but {than} {target:~P}: a matrix has one unit "
                "for every entry, and mixed-unit matrices are not supported"
            ) from None
    return reg.Quantity(out, target)


@node(category="Symbolic", title="Evaluate Matrix", sample=False)
def evaluate_matrix(
    matrix: Expression,
    values: SymbolValues,
    unit: Annotated[str, Param(description="Unit of every entry; empty: worked out")] = "",
) -> Quantity:
    """Put numbers with units into a symbolic matrix, such as the stiffness
    matrix from Linear System, for the matrix nodes. A matrix has one unit
    for every entry: entries that come out in different units are an error
    (zeros fit any unit). Values with uncertainties are used at their nominal
    values, since a matrix carries none: use Monte Carlo to propagate them."""
    return _evaluate_matrix(matrix, values, unit)


@evaluate_matrix.check
def _check_evaluate_matrix(matrix: Any = None, values: Any = None, unit: str = "") -> Any:
    if problem := _unit_problem(unit):
        return problem
    if matrix is not None and not isinstance(matrix, sp.MatrixBase):
        return "This is not a matrix: use Evaluate"
    if matrix is None or values is None:
        return None
    try:
        _evaluate_matrix(matrix, values, unit)
    except (KeyError, ValueError, TypeError) as exc:
        return str(exc).strip("'\"")
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if any(uncertainty.is_uncertain(v) for v in values.values()):
        return warning(
            "Values with uncertainties are used at their nominal values: a matrix carries no "
            "uncertainty. Use Monte Carlo to propagate them"
        )
    if not unit.strip():
        return warning("Set a unit to be sure what the matrix is in")
    return None


class NumericSolution(NamedTuple):
    values: SymbolValues
    solution: Quantity
    converged: bool
    residual: float
    iterations: int
    summary: dict[str, float]


def _guesses(text: str) -> dict[str, Any]:
    return {name: _value(name, value) for name, value in split_assignments(text)}


def _root(
    eqs: list[Equation],
    guesses: dict[str, Any],
    known: dict[str, Any],
    method: str,
    max_iterations: int,
) -> tuple[dict[str, Any], Any, float]:
    """Solve for the guessed unknowns with SciPy, on plain numbers: each
    unknown in the unit of its guess, each residual in base units and scaled
    by its size at the guess, so equations in N and in mm count alike."""
    from scipy.optimize import root

    reg = ureg()
    names = list(guesses)
    units_ = [g.units if is_quantity(g) else None for g in guesses.values()]
    z0 = np.array([float(g.magnitude if is_quantity(g) else g) for g in guesses.values()])
    residuals = [e.lhs - e.rhs for e in eqs]

    def raw(z: Any) -> Any:
        vals = dict(known)
        for n, u, x in zip(names, units_, z, strict=True):
            vals[n] = reg.Quantity(x, u) if u is not None else float(x)
        out = []
        for r in residuals:
            v = _call(r, vals)
            v = v.to_base_units().magnitude if is_quantity(v) else v
            out.append(float(v))
        return np.array(out)

    r0 = np.abs(raw(z0))
    scale = np.where(r0 > 0, r0, 1.0)
    options = {"maxfev": max_iterations} if method == "hybr" else {"maxiter": max_iterations}
    sol = root(lambda z: raw(z) / scale, z0, method=method, options=options)
    found = {
        n: reg.Quantity(float(x), u) if u is not None else float(x)
        for n, u, x in zip(names, units_, sol.x, strict=True)
    }
    residual = float(np.max(np.abs(raw(sol.x) / scale))) if len(residuals) else 0.0
    return found, sol, residual


@node(category="Symbolic", title="Solve Numerically", sample=False)
def solve_numeric(
    equations: EquationSystem | Equation,
    guesses: Annotated[
        str,
        Param(
            multiline=True,
            description="A first guess for each unknown, with its unit, one per line: x1 = 1 cm",
        ),
    ] = "x = 1",
    values: SymbolValues | None = None,
    method: Annotated[
        Literal["hybr", "lm"],
        Param(description="hybr: as many equations as unknowns; lm: least squares, more is fine"),
    ] = "hybr",
    max_iterations: Annotated[int, Param(min=1, max=100_000)] = 1000,
) -> NumericSolution:
    """Solve equations, linear or not, numerically for the unknowns that have
    a guess, starting from those guesses (SciPy's root). Each unknown comes
    out in the unit of its guess. The other symbols take ``values``, whose
    uncertainties are propagated to the solutions (GUM).

    ``values`` is the input values plus the solutions, ready for Evaluate.
    ``residual`` is the largest equation error, relative to its size at the
    guess. A solve that does not converge says so rather than failing: a
    different guess often helps."""
    eqs = _system(equations)
    first = _guesses(guesses)
    if not first:
        raise ValueError("Give a first guess for each unknown: x1 = 1 cm")
    names = [s.name for s in _unknowns(", ".join(first), eqs)]
    if method == "hybr" and len(eqs) != len(names):
        raise ValueError(
            f"{len(eqs)} equation(s) for {len(names)} unknown(s): hybr needs as many of each; "
            "lm solves more equations than unknowns in the least-squares sense"
        )
    known = dict(values or {})
    for v in known.values():
        if np.ndim(v.magnitude if is_quantity(v) else v) != 0:
            raise ValueError("Solve Numerically takes single values, not arrays")

    def solved(**kw: Any) -> dict[str, Any]:
        return _root(eqs, first, kw, method, max_iterations)[0]

    _, sol, residual = _root(eqs, first, _nominal_values(known), method, max_iterations)
    found = uncertainty.lift(solved, known, set(known))
    reg = ureg()
    head = found[names[0]]
    summary: dict[str, float] = {}
    for n, v in found.items():
        label = f"{n} ({v.units:~P})" if is_quantity(v) and f"{v.units:~P}" else n
        summary[label] = float(uncertainty.nominal(v.magnitude if is_quantity(v) else v))
    summary["Largest relative residual"] = residual
    return NumericSolution(
        SymbolValues({**known, **found}),
        head if is_quantity(head) else reg.Quantity(head, ""),
        bool(sol.success),
        residual,
        int(getattr(sol, "nfev", 0)),
        summary,
    )


@solve_numeric.check
def _check_solve_numeric(equations: Any = None, guesses: str = "", method: str = "hybr") -> Any:
    try:
        first = _guesses(guesses)
    except ParseError as exc:
        return str(exc)
    if not first:
        return "Give a first guess for each unknown: x1 = 1 cm"
    if equations is None:
        return None
    eqs = _system(equations)
    try:
        _unknowns(", ".join(first), eqs)
    except ValueError as exc:
        return str(exc)
    if method == "hybr" and len(eqs) != len(first):
        return f"{len(eqs)} equation(s) for {len(first)} unknown(s): hybr needs as many of each"
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
    max_iterations: Annotated[int, Param(min=1, max=100_000)] = 100,
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
    expression: Expression | Equation | EquationSystem,
    left: Annotated[str, Param(description="Optional left-hand side, e.g. M(x) or sigma_max")] = "",
) -> str:
    """Typst math for the report's Add Equation node: the expression as it
    would be typeset, optionally as ``left = expression``. A system of
    equations is typeset one equation per line."""
    math = typst_math(tuple(expression) if isinstance(expression, tuple) else expression)
    if left.strip():
        math = f"{typst_math(parse_expression(left))} = {math}"
    return TypstMath(math)  # a str, previewed typeset


@to_math.check
def _check_to_math(left: str = "") -> str | None:
    return _problem(parse_expression, left) if left.strip() else None
