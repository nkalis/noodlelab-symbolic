"""Symbolic maths with SymPy. Requires the ``symbolic`` extra (included in
``science``): ``pip install "noodlelab[symbolic]"``.

Expressions and equations are typed as text and become SymPy objects that
flow between nodes: differentiate, integrate, substitute, simplify, solve
algebraic equations and ODEs with boundary conditions. **Evaluate** then puts
in numbers with Pint units, so a formula checks its own dimensions, and
**Expression To Math** typesets any result in a report.

* :mod:`.parse`: reading typed text safely (no ``eval`` of arbitrary Python)
* :mod:`.types`: socket types, previews, meta and checkpoint codecs
* :mod:`.typst`: SymPy to Typst math
* :mod:`.nodes`: the nodes
"""

from __future__ import annotations

try:
    import numpy  # noqa: F401
    import pint  # noqa: F401
    import sympy  # noqa: F401
except ImportError as exc:  # shown in the editor's pack errors
    raise ImportError(
        f"{exc.name} is not installed. Install the symbolic extra: "
        'uv pip install "noodlelab[symbolic]"'
    ) from exc

from .nodes import *  # noqa: F403
from .types import SymbolValues  # noqa: F401
