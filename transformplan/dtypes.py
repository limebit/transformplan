"""Canonical dtype names for serializable transformation plans.

A plan must survive a round trip through JSON. Polars dtype objects do not:
``json.dumps`` rejects them, and ``to_python()`` renders them as bare names that
no import statement provides. This module defines a canonical string name per
dtype so that plans store a name, not an object.

The names follow Polars spelling ("Float64", "String", "Boolean"), and each
backend resolves them to its own type system via ``Backend.resolve_dtype()``.

Functions:
    normalize_dtype: Convert a dtype specification to its canonical name.
    canonical_dtype_names: List all supported canonical names.

Example:
    >>> normalize_dtype(pl.Float64)
    'Float64'
    >>> normalize_dtype(float)
    'Float64'
    >>> normalize_dtype("Utf8")
    'String'
"""

from __future__ import annotations

import datetime
from typing import Any

import polars as pl

# Canonical name -> Polars dtype. The canonical spelling is the Polars one.
_CANONICAL_TO_POLARS: dict[str, Any] = {
    "Int8": pl.Int8,
    "Int16": pl.Int16,
    "Int32": pl.Int32,
    "Int64": pl.Int64,
    "UInt8": pl.UInt8,
    "UInt16": pl.UInt16,
    "UInt32": pl.UInt32,
    "UInt64": pl.UInt64,
    "Float32": pl.Float32,
    "Float64": pl.Float64,
    "Boolean": pl.Boolean,
    "String": pl.String,
    "Binary": pl.Binary,
    "Date": pl.Date,
    "Datetime": pl.Datetime,
    "Time": pl.Time,
    "Duration": pl.Duration,
}

# Canonical name -> DuckDB SQL type.
_CANONICAL_TO_DUCKDB: dict[str, str] = {
    "Int8": "TINYINT",
    "Int16": "SMALLINT",
    "Int32": "INTEGER",
    "Int64": "BIGINT",
    "UInt8": "UTINYINT",
    "UInt16": "USMALLINT",
    "UInt32": "UINTEGER",
    "UInt64": "UBIGINT",
    "Float32": "FLOAT",
    "Float64": "DOUBLE",
    "Boolean": "BOOLEAN",
    "String": "VARCHAR",
    "Binary": "BLOB",
    "Date": "DATE",
    "Datetime": "TIMESTAMP",
    "Time": "TIME",
    "Duration": "INTERVAL",
}

# Builtin Python types accepted as dtype shorthands.
_PYTHON_TO_CANONICAL: list[tuple[type, str]] = [
    (bool, "Boolean"),  # before int: bool is a subclass of int
    (int, "Int64"),
    (float, "Float64"),
    (str, "String"),
    (bytes, "Binary"),
    (datetime.datetime, "Datetime"),
    (datetime.date, "Date"),
    (datetime.time, "Time"),
    (datetime.timedelta, "Duration"),
]

# Accepted spellings that are not the canonical one.
_ALIASES: dict[str, str] = {
    "Utf8": "String",
    "Bool": "Boolean",
    "Str": "String",
}


def canonical_dtype_names() -> list[str]:
    """List the canonical dtype names.

    Returns:
        Sorted list of supported names.
    """
    return sorted(_CANONICAL_TO_POLARS)


def normalize_dtype(dtype: Any) -> Any:  # noqa: ANN401
    """Convert a dtype specification to its canonical name.

    Accepts a canonical name, a known alias, a Polars dtype class, or a builtin
    Python type. Anything not recognized is returned unchanged, so that exotic
    or parameterized dtypes (``pl.Datetime("us")``, ``pl.Categorical``,
    ``pl.List``) keep working exactly as before -- such plans simply remain
    non-serializable, as they were prior to this function existing.

    Args:
        dtype: Dtype specification.

    Returns:
        Canonical name string, or the input unchanged if not recognized.

    Raises:
        ValueError: If a string is given that is neither canonical nor a known
            alias. Strings are unambiguously meant as names, so a typo is
            reported when the plan is built rather than when it runs.
    """
    if isinstance(dtype, str):
        resolved = _ALIASES.get(dtype, dtype)
        if resolved not in _CANONICAL_TO_POLARS:
            msg = (
                f"Unknown dtype name: {dtype!r}. "
                f"Supported names: {', '.join(canonical_dtype_names())}"
            )
            raise ValueError(msg)
        return resolved

    # Builtin Python types. Identity comparison avoids triggering the custom
    # __eq__ that Polars dtypes define.
    for py_type, name in _PYTHON_TO_CANONICAL:
        if dtype is py_type:
            return name

    # Polars dtype classes. Parameterized instances (pl.Datetime("us")) have no
    # __name__ and fall through unchanged, keeping their parameters intact.
    name = getattr(dtype, "__name__", None)
    if isinstance(name, str):
        canonical = _ALIASES.get(name, name)
        if _CANONICAL_TO_POLARS.get(canonical) is dtype:
            return canonical

    return dtype


def to_polars_dtype(dtype: Any) -> Any:  # noqa: ANN401
    """Resolve a dtype specification to a Polars dtype.

    Args:
        dtype: Canonical name, alias, Python type, or Polars dtype.

    Returns:
        Polars dtype, or the input unchanged if it is not a known name.
    """
    normalized = normalize_dtype(dtype)
    if isinstance(normalized, str):
        return _CANONICAL_TO_POLARS[normalized]
    return normalized


def to_duckdb_type(dtype: Any) -> str:  # noqa: ANN401
    """Resolve a dtype specification to a DuckDB SQL type name.

    Args:
        dtype: Canonical name, alias, Python type, or Polars dtype.

    Returns:
        DuckDB SQL type string. Unrecognized dtypes fall back to VARCHAR.
    """
    normalized = normalize_dtype(dtype)
    if isinstance(normalized, str):
        return _CANONICAL_TO_DUCKDB[normalized]
    return "VARCHAR"
