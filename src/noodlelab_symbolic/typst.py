"""SymPy expressions as Typst math, for the report's Add Equation node.

Typst math reads much like plain text (``x^2``, ``a/b``, ``sqrt(x)``, Greek
letters by name), so this is a small printer rather than a translation from
LaTeX. Symbol names follow SymPy's conventions: ``sigma_max`` is σ with the
subscript "max", ``x2`` is x₂, and Greek names become letters.
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.printing.conventions import split_super_sub

_GREEK = set(
    [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    ]
)
_GREEK |= {g.capitalize() for g in _GREEK}
_FUNCTIONS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "cot": "cot",
    "sec": "sec",
    "csc": "csc",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "log": "ln",
    "Max": "max",
    "Min": "min",
    "sign": 'op("sgn")',
}


def _name(text: str) -> str:
    """One piece of a symbol name: a letter, a Greek letter, digits, or upright text."""
    if text == "lamda":
        text = "lambda"
    if len(text) == 1 or text in _GREEK or text.isdigit():
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def symbol_name(name: str) -> str:
    base, supers, subs = split_super_sub(name)
    out = _name(base)
    if subs:
        out += (
            "_(" + ",".join(_name(s) for s in subs) + ")" if len(subs) > 1 else "_" + _name(subs[0])
        )
    if supers:
        out += "^(" + ",".join(_name(s) for s in supers) + ")"
    return out


def _number(value: sp.Number) -> str:
    if isinstance(value, sp.Integer):
        return str(int(value))
    if isinstance(value, sp.Rational):
        return f"{value.p}/{value.q}"
    text = f"{float(value):.6g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return f"{mantissa} times 10^({int(exponent)})"
    return text


class _Printer:
    def __call__(self, expr: Any) -> str:
        return self.p(sp.sympify(expr))

    # --- helpers ----------------------------------------------------------------------------

    def group(self, expr: sp.Basic) -> str:
        """``expr`` as a factor of a product: parenthesized when it is a sum."""
        text = self.p(expr)
        if isinstance(expr, sp.Add | Relational) or (isinstance(expr, sp.Number) and expr < 0):
            return f"({text})"
        return text

    def base(self, expr: sp.Basic) -> str:
        """``expr`` as the base of a power."""
        text = self.p(expr)
        if (
            isinstance(expr, sp.Add | sp.Mul | sp.Pow | Relational)
            or (isinstance(expr, sp.Rational | sp.Float) and not isinstance(expr, sp.Integer))
            or (isinstance(expr, sp.Number) and expr < 0)
        ):
            return f"({text})"
        return text

    def operand(self, expr: sp.Basic) -> str:
        """``expr`` as a numerator, denominator or exponent: Typst drops the
        parentheses around it when rendering, so they only group."""
        text = self.p(expr)
        if (isinstance(expr, sp.Symbol | sp.Integer) and not expr.is_negative) or (
            text.isidentifier()
        ):
            return text
        return f"({text})"

    def product(self, factors: list[sp.Basic]) -> str:
        parts: list[str] = []
        previous_number = False
        for f in factors:
            text = self.group(f)
            if parts and previous_number and (isinstance(f, sp.Number) or text[:1].isdigit()):
                parts.append("dot")
            parts.append(text)
            previous_number = isinstance(f, sp.Number)
        return " ".join(parts)

    # --- dispatch ---------------------------------------------------------------------------

    def p(self, expr: Any) -> str:
        for klass in type(expr).__mro__:
            method = getattr(self, f"_print_{klass.__name__}", None)
            if method is not None:
                return method(expr)
        return '"' + str(expr).replace('"', "'") + '"'

    def _print_Symbol(self, expr: sp.Symbol) -> str:
        return symbol_name(expr.name)

    def _print_Number(self, expr: sp.Number) -> str:
        return _number(expr)

    def _print_Pi(self, expr: Any) -> str:
        return "pi"

    def _print_Exp1(self, expr: Any) -> str:
        return "e"

    def _print_ImaginaryUnit(self, expr: Any) -> str:
        return "j"

    def _print_Infinity(self, expr: Any) -> str:
        return "infinity"

    def _print_NegativeInfinity(self, expr: Any) -> str:
        return "-infinity"

    def _print_Add(self, expr: sp.Add) -> str:
        terms = expr.as_ordered_terms()
        out = self.p(terms[0])
        for term in terms[1:]:
            if term.could_extract_minus_sign():
                out += " - " + self.p(-term)
            else:
                out += " + " + self.p(term)
        return out

    def _print_Mul(self, expr: sp.Mul) -> str:
        if expr.could_extract_minus_sign():
            inner = -expr
            text = self.p(inner)
            return f"-({text})" if isinstance(inner, sp.Add) else f"-{text}"
        numer, denom = sp.fraction(expr, exact=True)
        if denom != 1:
            return f"{self.operand(numer)}/{self.operand(denom)}"
        factors = list(sp.Mul.make_args(expr))
        numbers = [f for f in factors if isinstance(f, sp.Number)]
        others = [f for f in factors if not isinstance(f, sp.Number)]
        return self.product(numbers + others)

    def _print_Pow(self, expr: sp.Pow) -> str:
        b, ex = expr.as_base_exp()
        if ex == sp.Rational(1, 2):
            return f"sqrt({self.p(b)})"
        if ex == sp.Rational(1, 3):
            return f"root(3, {self.p(b)})"
        if ex.is_negative:
            return f"1/{self.operand(b**-ex)}"
        if b == sp.E:
            return f"e^{self.operand(ex)}"
        return f"{self.base(b)}^{self.operand(ex)}"

    def _print_exp(self, expr: Any) -> str:
        return f"e^{self.operand(expr.args[0])}"

    def _print_Abs(self, expr: Any) -> str:
        return f"abs({self.p(expr.args[0])})"

    def _print_Function(self, expr: Any) -> str:
        name = type(expr).__name__
        if isinstance(expr, AppliedUndef):
            head = symbol_name(name)
        else:
            head = _FUNCTIONS.get(name, f'op("{name}")')
        return f"{head}({', '.join(self.p(a) for a in expr.args)})"

    def _print_Derivative(self, expr: sp.Derivative) -> str:
        f = expr.expr
        order = sum(n for _, n in expr.variable_count)
        d = "upright(d)"
        top = f"{d}^{order}" if order > 1 else d
        bottom = " ".join(
            f"{d} {self.base(v)}" + (f"^{n}" if n > 1 else "") for v, n in expr.variable_count
        )
        if isinstance(f, AppliedUndef):
            return f"({top} {symbol_name(type(f).__name__)})/({bottom})"
        if isinstance(f, sp.Symbol):
            return f"({top} {self.p(f)})/({bottom})"
        return f"({top})/({bottom}) ({self.p(f)})"

    def _print_Subs(self, expr: sp.Subs) -> str:
        inner, variables, points = expr.args
        if (
            isinstance(inner, sp.Derivative)
            and isinstance(inner.expr, AppliedUndef)
            and len(variables) == 1
            and len(inner.variable_count) == 1
        ):
            ((v, n),) = inner.variable_count
            if v == variables[0]:
                primes = "'" * n
                return f"{symbol_name(type(inner.expr).__name__)}{primes}({self.p(points[0])})"
        subs = ", ".join(
            f"{self.p(v)} = {self.p(pt)}" for v, pt in zip(variables, points, strict=True)
        )
        return f"lr(({self.p(inner)}) |)_({subs})"

    def _print_Integral(self, expr: sp.Integral) -> str:
        out = ""
        for limit in reversed(expr.limits):
            bounds = limit[1:]
            head = "integral"
            if len(bounds) == 2:
                head += f"_({self.p(bounds[0])})^({self.p(bounds[1])})"
            out = f"{head} " + out
        dvars = " ".join(f"upright(d) {self.p(lim[0])}" for lim in expr.limits)
        return f"{out}{self.group(expr.function)} {dvars}"

    def _print_Relational(self, expr: Relational) -> str:
        op = {"==": "=", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}[expr.rel_op]
        return f"{self.p(expr.lhs)} {op} {self.p(expr.rhs)}"

    def _print_Piecewise(self, expr: sp.Piecewise) -> str:
        rows = []
        for value, cond in expr.args:
            when = '"otherwise"' if cond is sp.true else f'"if" {self.p(cond)}'
            rows.append(f"{self.p(value)} & {when}")
        return "cases(" + ", ".join(rows) + ")"


def typst_math(expr: Any) -> str:
    """Typst math for a SymPy expression or equation, without the ``$`` signs."""
    return _Printer()(expr)
