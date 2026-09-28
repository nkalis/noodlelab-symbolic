"""Symbolic maths with SymPy. Part of the ``maths`` tier:
``pip install "noodlelab[maths]"``.

Expressions and equations are typed as text and become SymPy objects that
flow between nodes: differentiate, integrate, substitute, simplify, solve
algebraic equations and ODEs with boundary conditions. **Equations** holds a
system, which **Solve System** solves together in symbols, **Solve
Numerically** with SciPy, and **Linear System** writes as a matrix, A x = b.
**Evaluate** (and **Evaluate Matrix**) then puts in numbers with Pint units,
so a formula checks its own dimensions, and **Expression To Math** typesets
any result in a report.

* :mod:`.parse`: reading typed text safely (no ``eval`` of arbitrary Python)
* :mod:`.types`: socket types, previews, meta and checkpoint codecs
* :mod:`.typst`: SymPy to Typst math
* :mod:`.nodes`: the nodes
"""

from __future__ import annotations

from noodlelab.plugin import require

require("maths", "numpy", "pint", "sympy")

from .nodes import *  # noqa: F403
from .types import SymbolValues  # noqa: F401
