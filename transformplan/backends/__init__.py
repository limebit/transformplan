"""Backend abstraction for TransformPlan.

This package provides the backend interface and implementations for executing
DataFrame operations. Each backend implements the same set of operations,
allowing TransformPlan to work with different DataFrame libraries.

Classes:
    Backend: Abstract base class defining the operation interface.
    PolarsBackend: Default backend using Polars for execution.
    DuckDBBackend: Optional backend using DuckDB for execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformplan.backends.base import Backend
from transformplan.backends.polars import PolarsBackend

if TYPE_CHECKING:
    from transformplan.backends.duckdb import DuckDBBackend

__all__ = [
    "Backend",
    "DuckDBBackend",
    "PolarsBackend",
]


def __getattr__(name: str) -> type:
    """Lazy import for optional backends."""
    if name == "DuckDBBackend":
        from transformplan.backends.duckdb import DuckDBBackend

        return DuckDBBackend
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
