"""Shared helpers for operation mixins.

Functions:
    as_columns: Normalize a column argument to a list of column names.
"""

from __future__ import annotations

from typing import Sequence


def as_columns(column: str | Sequence[str]) -> list[str]:
    """Normalize a column argument to a list of names.

    Operations that work in place on a column accept either a single name or a
    sequence of names. A sequence registers one step per column, which keeps the
    protocol step-granular and leaves backends and validators untouched.

    Args:
        column: A single column name or a sequence of names.

    Returns:
        List of column names.
    """
    return [column] if isinstance(column, str) else list(column)
