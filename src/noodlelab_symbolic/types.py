"""The symbolic pack's types: expressions, equations and symbol values.

Each gets a socket colour, a preview (the expression typeset), meta
for the editor's dropdowns (the symbols in an expression, the names in a set
of values) and a checkpoint codec that needs no pickle.
"""

from __future__ import annotations

import ast
import functools
import io
import json
from typing import IO, Any

import numpy as np
import sympy as sp
from sympy.core.relational import Equality

from noodlelab import (
    Preview,
    register_codec,
    register_meta,
    register_preview,
    register_sampler,
    register_type,
)
from noodlelab.plugin.math import math_preview
from noodlelab.plugin.units import is_quantity, magnitude, ureg

from .typst import typst_math

Expression = sp.Expr
Equation = Equality


SymbolicMatrix = sp.ImmutableMatrix  # an Expr too, so matrices reach Simplify, Substitute...


class EquationSystem(tuple):
    """Several equations to be solved together, such as one balance equation
    per mass: a tuple of SymPy equations."""

    @property
    def symbols(self) -> list[str]:
        return sorted({s for e in self for s in symbols_of(e)})

    def __repr__(self) -> str:
        return "EquationSystem(" + "; ".join(sp.sstr(e) for e in self) + ")"


class SymbolValues(dict):
    """Values for the symbols of an expression, by name: numbers, or Pint
    quantities (scalars or arrays). Evaluate uses them to put numbers, with
    their units, into an expression."""

    def __repr__(self) -> str:
        return "SymbolValues(" + ", ".join(f"{k} = {_show(v)}" for k, v in self.items()) + ")"


def _show(value: Any) -> str:
    if is_quantity(value):
        if np.ndim(value.magnitude):
            return f"array {np.shape(value.magnitude)} {value.units:~P}"
        return f"{value:.6g~P}"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


register_type(sp.Expr, "EXPRESSION", "#c678dd", "A symbolic expression (SymPy)")
register_type(Equality, "EQUATION", "#a35fc4", "A symbolic equation, lhs = rhs (SymPy)")
register_type(SymbolValues, "SYMBOL_VALUES", "#d19a66", "Values for symbols, with units")
register_type(EquationSystem, "EQUATIONS", "#8f4bb8", "Equations solved together (SymPy)")
register_type(SymbolicMatrix, "SYMBOLIC_MATRIX", "#b07cd8", "A matrix of expressions (SymPy)")


def symbols_of(expr: sp.Basic) -> list[str]:
    return sorted(s.name for s in expr.free_symbols if isinstance(s, sp.Symbol))


def functions_of(expr: sp.Basic) -> list[str]:
    from sympy.core.function import AppliedUndef

    return sorted({type(f).__name__ for f in expr.atoms(AppliedUndef)})


# --- previews and meta ----------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


@register_preview("sympy.core.basic.Basic")
def _preview_basic(value: sp.Basic, ctx: Any) -> Preview:
    """Typeset with Typst, as the report would show it, with the pretty-printed
    and plain forms as text; only the text when it cannot be typeset."""
    pretty = sp.pretty(value, use_unicode=True, wrap_line=True, num_columns=100)
    summary = _clip(sp.sstr(value), 60)
    text = _clip(f"{pretty}\n\n{sp.sstr(value)}", 4000)
    try:
        math = typst_math(value)
    except Exception:  # a preview is best effort: fall back to the text forms
        return Preview(kind="text", summary=summary, text=text)
    preview = math_preview(math, summary=summary, text=text)
    return preview if preview.kind == "math" else Preview(kind="text", summary=summary, text=text)


@register_preview(EquationSystem)
def _preview_system(value: EquationSystem, ctx: Any) -> Preview:
    """Typeset one equation per line, aligned at the = signs."""
    plain = "\n".join(sp.sstr(e) for e in value)
    summary = _clip("; ".join(sp.sstr(e) for e in value), 60) or "no equations"
    text = _clip(plain, 4000)
    try:
        math = typst_math(tuple(value))
    except Exception:  # a preview is best effort: fall back to the text
        return Preview(kind="text", summary=summary, text=text)
    preview = math_preview(math, summary=summary, text=text)
    return preview if preview.kind == "math" else Preview(kind="text", summary=summary, text=text)


@register_meta(EquationSystem)
def _meta_system(value: EquationSystem) -> dict[str, Any]:
    return {
        "symbols": value.symbols,
        "functions": sorted({f for e in value for f in functions_of(e)}),
    }


@register_meta("sympy.core.basic.Basic")
def _meta_basic(value: sp.Basic) -> dict[str, Any]:
    return {"symbols": symbols_of(value), "functions": functions_of(value)}


@register_preview(SymbolValues)
def _preview_values(value: SymbolValues, ctx: Any) -> Preview:
    lines = [f"{k} = {_show(v)}" for k, v in value.items()]
    return Preview(
        kind="text", summary=_clip(", ".join(lines) or "no values", 60), text="\n".join(lines)
    )


@register_meta(SymbolValues)
def _meta_values(value: SymbolValues) -> dict[str, Any]:
    return {"keys": list(value), "symbols": list(value)}


# --- checkpoints: srepr, read back without eval ---------------------------------------------


@functools.cache
def _sympy_names() -> dict[str, Any]:
    """The names srepr() writes: SymPy classes and singletons (pi, oo...)."""
    names = {}
    for name in dir(sp):
        obj = getattr(sp, name)
        if (isinstance(obj, type) and issubclass(obj, sp.Basic)) or isinstance(obj, sp.Basic):
            names[name] = obj
    names.update(
        Tuple=sp.Tuple,
        Function=sp.Function,
        Symbol=sp.Symbol,
        Dummy=sp.Dummy,
        true=sp.true,
        false=sp.false,
    )
    return names


def from_srepr(text: str) -> sp.Basic:
    """The inverse of ``sympy.srepr``, reading only calls of SymPy classes with
    literal arguments: loading a checkpoint cannot run code."""
    names = _sympy_names()

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, str | int | float | bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.Tuple | ast.List):
            return tuple(ev(e) for e in node.elts)
        if isinstance(node, ast.Call):
            func = ev(node.func)
            if not (
                (isinstance(func, type) and issubclass(func, sp.Basic))
                or func in (sp.Function, sp.Symbol)
                or isinstance(func, sp.FunctionClass)
            ):
                raise ValueError(f"not a SymPy class: {ast.dump(node.func)[:80]}")
            kwargs = {k.arg: ev(k.value) for k in node.keywords if k.arg}
            return func(*(ev(a) for a in node.args), **kwargs)
        raise ValueError(f"unexpected {type(node).__name__} in a stored expression")

    return ev(ast.parse(text, mode="eval").body)


def _save_sympy(value: sp.Basic, fh: IO[bytes]) -> None:
    text = sp.srepr(value)
    if from_srepr(text) != value:
        raise TypeError("the expression does not read back exactly")
    fh.write(text.encode())


def _load_sympy(fh: IO[bytes], info: dict[str, Any]) -> sp.Basic:
    return from_srepr(fh.read().decode())


register_codec(
    "sympy.core.basic.Basic", "sympy-srepr", save=_save_sympy, load=_load_sympy, suffix=".txt"
)


def _save_system(value: EquationSystem, fh: IO[bytes]) -> None:
    texts = [sp.srepr(e) for e in value]
    if tuple(from_srepr(t) for t in texts) != tuple(value):
        raise TypeError("the equations do not read back exactly")
    fh.write(json.dumps(texts).encode())


def _load_system(fh: IO[bytes], info: dict[str, Any]) -> EquationSystem:
    return EquationSystem(from_srepr(t) for t in json.loads(fh.read().decode()))


register_codec(
    EquationSystem, "sympy-equations", save=_save_system, load=_load_system, suffix=".json"
)


@register_sampler(EquationSystem)
def _sample_system(value: EquationSystem, size: int) -> EquationSystem:
    return value  # every equation counts: a few of them is a different problem


def _save_values(value: SymbolValues, fh: IO[bytes]) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    info: dict[str, Any] = {}
    for i, (name, v) in enumerate(value.items()):
        m = magnitude(v)
        if isinstance(m, bool) or not isinstance(m, int | float | np.ndarray | np.generic):
            raise TypeError(f"{name}: a {type(m).__name__} has no safe format")
        arrays[f"v{i}"] = np.asarray(m)
        info[name] = {
            "units": str(v.units) if is_quantity(v) else None,
            "kind": type(m).__name__ if isinstance(m, int | float) else "numpy",
        }
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    fh.write(buf.getvalue())
    return {"values": json.dumps(info)}


def _load_values(fh: IO[bytes], info: dict[str, Any]) -> SymbolValues:
    data = np.load(io.BytesIO(fh.read()), allow_pickle=False)
    out = SymbolValues()
    for i, (name, spec) in enumerate(json.loads(info["values"]).items()):
        m: Any = data[f"v{i}"]
        if spec["kind"] in ("float", "int"):
            m = {"float": float, "int": int}[spec["kind"]](m[()])
        elif m.ndim == 0:
            m = m[()]
        out[name] = m if spec["units"] is None else ureg().Quantity(m, spec["units"])
    return out


register_codec(SymbolValues, "symbol-values", save=_save_values, load=_load_values, suffix=".npz")
