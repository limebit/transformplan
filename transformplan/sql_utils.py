"""Shared SQL helper functions for DuckDB backend and filter SQL generation.

Functions:
    sql_quote_identifier: Double-quote a SQL identifier with escaping.
    sql_format_value: Format a Python value as a SQL literal.
    sql_escape_like: Escape LIKE metacharacters for SQL.
"""

from __future__ import annotations

import datetime
import math
from typing import Any


def sql_quote_identifier(name: str) -> str:
    """Double-quote a SQL identifier, escaping embedded double quotes.

    Returns:
        Quoted identifier string.
    """
    return '"' + name.replace('"', '""') + '"'


def sql_format_value(value: Any) -> str:  # noqa: ANN401
    """Format a Python value as a DuckDB-compatible SQL literal.

    Handles None, bool, int, float (including nan/inf), str, date, and datetime.

    Returns:
        SQL literal string.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        if math.isnan(value):
            return "'NaN'::DOUBLE"
        if math.isinf(value):
            return "'+Infinity'::DOUBLE" if value > 0 else "'-Infinity'::DOUBLE"
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime.datetime):
        return f"'{value.isoformat()}'::TIMESTAMP"
    if isinstance(value, datetime.date):
        return f"'{value.isoformat()}'::DATE"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return "'" + str(value).replace("'", "''") + "'"


def sql_escape_like(pattern: str) -> str:
    r"""Escape LIKE metacharacters and single quotes for SQL.

    Returns:
        Escaped pattern string.
    """
    return (
        pattern.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("'", "''")
    )
