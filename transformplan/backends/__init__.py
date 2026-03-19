"""Backend abstraction for TransformPlan.

This package provides the backend interface and implementations for executing
DataFrame operations. Each backend implements the same set of operations,
allowing TransformPlan to work with different DataFrame libraries.

Classes:
    Backend: Abstract base class defining the operation interface.
    PolarsBackend: Default backend using Polars for execution.
"""

from transformplan.backends.base import Backend
from transformplan.backends.polars import PolarsBackend

__all__ = [
    "Backend",
    "PolarsBackend",
]
