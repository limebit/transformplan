"""DuckDB backend for TransformPlan.

This module implements all 89 operations using DuckDB's ``DuckDBPyRelation``
as the data type.  Every operation takes a relation, generates SQL using the
relation's ``sql_query()`` as a subquery, and returns a new relation — keeping
the pipeline composable and lazy.

Classes:
    DuckDBBackend: Backend implementation using DuckDB.
"""

from __future__ import annotations

import datetime
import hashlib
import math
import secrets
import string
from typing import Any, ClassVar, Literal, Sequence

import duckdb

from transformplan.backends.base import (
    AggFunction,
    Backend,
    ClosedInterval,
    FeatureRange,
    FillNullStrategy,
    Numeric,
    RankMethod,
)
from transformplan.filters import Filter
from transformplan.sql_utils import sql_format_value as _v
from transformplan.sql_utils import sql_quote_identifier as _q

# =============================================================================
# SQL helpers
# =============================================================================

_DTYPE_TO_DUCKDB: dict[type, str] = {
    int: "BIGINT",
    float: "DOUBLE",
    str: "VARCHAR",
    bool: "BOOLEAN",
    datetime.datetime: "TIMESTAMP",
    datetime.date: "DATE",
}

_VALID_AGG_FUNCTIONS = {"first", "sum", "mean", "median", "min", "max", "count"}


def _dtype_to_duckdb(dtype: type) -> str:
    """Map Python type to DuckDB SQL type name.

    Returns:
        DuckDB type string.
    """
    return _DTYPE_TO_DUCKDB.get(dtype, "VARCHAR")


def _sub(rel: duckdb.DuckDBPyRelation) -> str:
    """Wrap a relation's query as a subquery.

    Returns:
        SQL subquery string.
    """
    return f"({rel.sql_query()}) AS _t"


class DuckDBBackend(Backend):
    """Backend implementation using DuckDB for all operations."""

    name = "duckdb"

    def __init__(self, con: duckdb.DuckDBPyConnection | None = None) -> None:
        """Initialize DuckDBBackend.

        Args:
            con: DuckDB connection. If None, creates an in-memory connection.
        """
        self._con = con or duckdb.connect()

    def __repr__(self) -> str:
        """Return string representation of the backend."""
        return f"DuckDBBackend(con={self._con!r})"

    # =========================================================================
    # Meta methods (4)
    # =========================================================================

    def compute_hash(self, data: duckdb.DuckDBPyRelation) -> str:
        # Deterministic hash: sort columns, sort rows, hash all content
        cols = sorted(data.columns)
        col_list = ", ".join(_q(c) for c in cols)
        concat_expr = " || '|' || ".join(
            f"COALESCE({_q(c)}::VARCHAR, '')" for c in cols
        )
        sql = (
            f"SELECT md5(string_agg(row_hash, '' ORDER BY row_hash)) AS h "
            f"FROM (SELECT md5({concat_expr}) AS row_hash "
            f"FROM (SELECT {col_list} FROM {_sub(data)}) AS _s) AS _r"
        )
        result = self._con.sql(sql).fetchone()
        if result is None or result[0] is None:
            return hashlib.md5(b"empty").hexdigest()[:16]
        return str(result[0])[:16]

    def get_shape(self, data: duckdb.DuckDBPyRelation) -> tuple[int, int]:
        count_result = self._con.sql(f"SELECT COUNT(*) FROM {_sub(data)}").fetchone()
        rows = count_result[0] if count_result else 0
        return (rows, len(data.columns))

    def get_columns(self, data: duckdb.DuckDBPyRelation) -> list[str]:
        return list(data.columns)

    def get_schema(self, data: duckdb.DuckDBPyRelation) -> dict[str, Any]:
        return dict(zip(data.columns, [str(t) for t in data.types], strict=False))

    # =========================================================================
    # Type system methods (13)
    # =========================================================================

    _NUMERIC_PREFIXES = frozenset(
        {
            "BIGINT",
            "INTEGER",
            "SMALLINT",
            "TINYINT",
            "HUGEINT",
            "UBIGINT",
            "UINTEGER",
            "USMALLINT",
            "UTINYINT",
            "DOUBLE",
            "FLOAT",
            "REAL",
            "DECIMAL",
        }
    )

    _STRING_PREFIXES = frozenset({"VARCHAR", "TEXT", "STRING", "CHAR", "BLOB"})

    _DATETIME_PREFIXES = frozenset(
        {
            "DATE",
            "TIMESTAMP",
            "TIMESTAMP WITH TIME ZONE",
            "TIMESTAMP_S",
            "TIMESTAMP_MS",
            "TIMESTAMP_NS",
            "TIME",
            "INTERVAL",
        }
    )

    def is_numeric_type(self, dtype: Any) -> bool:
        s = str(dtype).upper()
        return any(s == t or s.startswith(t + "(") for t in self._NUMERIC_PREFIXES)

    def is_string_type(self, dtype: Any) -> bool:
        s = str(dtype).upper()
        return any(s == t or s.startswith(t + "(") for t in self._STRING_PREFIXES)

    def is_datetime_type(self, dtype: Any) -> bool:
        s = str(dtype).upper()
        return any(s == t or s.startswith(t + "(") for t in self._DATETIME_PREFIXES)

    def is_boolean_type(self, dtype: Any) -> bool:
        return str(dtype).upper() == "BOOLEAN"

    def is_list_type(self, dtype: Any) -> bool:
        s = str(dtype).upper()
        return s.endswith("[]") or s.startswith("LIST")

    def float_type(self) -> str:
        return "DOUBLE"

    def string_type(self) -> str:
        return "VARCHAR"

    def integer_type(self) -> str:
        return "BIGINT"

    def unsigned_int_type(self) -> str:
        return "UINTEGER"

    def boolean_type(self) -> str:
        return "BOOLEAN"

    def date_type(self) -> str:
        return "DATE"

    def duration_type(self) -> str:
        return "INTERVAL"

    def type_name(self, dtype: Any) -> str:
        return str(dtype)

    # =========================================================================
    # Column operations (13)
    # =========================================================================

    def col_drop(
        self, data: duckdb.DuckDBPyRelation, column: str
    ) -> duckdb.DuckDBPyRelation:
        cols = [c for c in data.columns if c != column]
        col_list = ", ".join(_q(c) for c in cols)
        return self._con.sql(f"SELECT {col_list} FROM {_sub(data)}")

    def col_rename(
        self, data: duckdb.DuckDBPyRelation, column: str, new_name: str
    ) -> duckdb.DuckDBPyRelation:
        cols = [
            f"{_q(c)} AS {_q(new_name)}" if c == column else _q(c) for c in data.columns
        ]
        col_list = ", ".join(cols)
        return self._con.sql(f"SELECT {col_list} FROM {_sub(data)}")

    def col_cast(
        self, data: duckdb.DuckDBPyRelation, column: str, dtype: type
    ) -> duckdb.DuckDBPyRelation:
        ddb_type = _dtype_to_duckdb(dtype)
        cols = [
            f"CAST({_q(c)} AS {ddb_type}) AS {_q(c)}" if c == column else _q(c)
            for c in data.columns
        ]
        col_list = ", ".join(cols)
        return self._con.sql(f"SELECT {col_list} FROM {_sub(data)}")

    def col_reorder(
        self, data: duckdb.DuckDBPyRelation, columns: list[str]
    ) -> duckdb.DuckDBPyRelation:
        col_list = ", ".join(_q(c) for c in columns)
        return self._con.sql(f"SELECT {col_list} FROM {_sub(data)}")

    def col_select(
        self, data: duckdb.DuckDBPyRelation, columns: list[str]
    ) -> duckdb.DuckDBPyRelation:
        col_list = ", ".join(_q(c) for c in columns)
        return self._con.sql(f"SELECT {col_list} FROM {_sub(data)}")

    def col_duplicate(
        self, data: duckdb.DuckDBPyRelation, column: str, new_name: str
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, {_q(column)} AS {_q(new_name)} FROM {_sub(data)}"
        )

    def col_fill_null(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        value: Any,
        strategy: FillNullStrategy | None,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        if strategy is not None:
            if strategy == "forward":
                # Forward fill using last non-null value
                fill_expr = (
                    f"COALESCE({qc}, "
                    f"LAST_VALUE({qc} IGNORE NULLS) OVER ("
                    f"ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING))"
                )
            elif strategy == "backward":
                fill_expr = (
                    f"COALESCE({qc}, "
                    f"FIRST_VALUE({qc} IGNORE NULLS) OVER ("
                    f"ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING))"
                )
            elif strategy == "min":
                fill_expr = f"COALESCE({qc}, MIN({qc}) OVER())"
            elif strategy == "max":
                fill_expr = f"COALESCE({qc}, MAX({qc}) OVER())"
            elif strategy == "mean":
                fill_expr = f"COALESCE({qc}, AVG({qc}) OVER())"
            elif strategy == "zero":
                fill_expr = f"COALESCE({qc}, 0)"
            else:
                fill_expr = qc
            cols = [
                f"{fill_expr} AS {qc}" if c == column else _q(c) for c in data.columns
            ]
        else:
            fill_expr = f"COALESCE({qc}, {_v(value)})"
            cols = [
                f"{fill_expr} AS {qc}" if c == column else _q(c) for c in data.columns
            ]
        col_list = ", ".join(cols)
        return self._con.sql(f"SELECT {col_list} FROM {_sub(data)}")

    def col_drop_null(
        self, data: duckdb.DuckDBPyRelation, columns: list[str] | None
    ) -> duckdb.DuckDBPyRelation:
        target_cols = columns if columns is not None else list(data.columns)
        conditions = " AND ".join(f"{_q(c)} IS NOT NULL" for c in target_cols)
        return self._con.sql(f"SELECT * FROM {_sub(data)} WHERE {conditions}")

    def col_drop_zero(
        self, data: duckdb.DuckDBPyRelation, column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(f"SELECT * FROM {_sub(data)} WHERE {_q(column)} != 0")

    def col_add(
        self,
        data: duckdb.DuckDBPyRelation,
        new_column: str,
        expr: str | None,
        value: Any,
    ) -> duckdb.DuckDBPyRelation:
        if expr is not None:
            add_expr = f"{_q(expr)} AS {_q(new_column)}"
        else:
            add_expr = f"{_v(value)} AS {_q(new_column)}"
        return self._con.sql(f"SELECT *, {add_expr} FROM {_sub(data)}")

    def col_add_uuid(
        self, data: duckdb.DuckDBPyRelation, column: str, length: int
    ) -> duckdb.DuckDBPyRelation:
        # DuckDB uuid() generates full UUIDs; we generate Python-side
        # for length control.
        # NOTE: ROW_NUMBER() OVER () has undefined ordering in SQL, but
        # since UUIDs are random, the non-deterministic assignment does
        # not affect correctness — each row gets a unique random ID.
        count = self.get_shape(data)[0]
        chars = string.ascii_letters + string.digits
        ids = [
            "".join(secrets.choice(chars) for _ in range(length)) for _ in range(count)
        ]
        # Create a temp table with the IDs and join
        id_values = ", ".join(f"({_v(uid)})" for uid in ids)
        if count == 0:
            return self._con.sql(
                f"SELECT *, NULL::VARCHAR AS {_q(column)} FROM {_sub(data)} WHERE FALSE"
            )
        uuid_sql = (
            f"SELECT _t.*, _u.{_q(column)} FROM "
            f"(SELECT *, ROW_NUMBER() OVER () AS _rn FROM {_sub(data)}) AS _t "
            f"JOIN (SELECT ROW_NUMBER() OVER () AS _rn, "
            f"col0 AS {_q(column)} "
            f"FROM (VALUES {id_values}) AS _vals) AS _u "
            f"ON _t._rn = _u._rn"
        )
        # Drop the _rn column
        rel = self._con.sql(uuid_sql)
        keep_cols = [c for c in rel.columns if c != "_rn"]
        col_list = ", ".join(_q(c) for c in keep_cols)
        return self._con.sql(f"SELECT {col_list} FROM ({rel.sql_query()}) AS _r")

    def col_hash(
        self,
        data: duckdb.DuckDBPyRelation,
        columns: list[str],
        new_column: str,
        salt: str,
    ) -> duckdb.DuckDBPyRelation:
        parts = " || '|' || ".join(f"COALESCE({_q(c)}::VARCHAR, '')" for c in columns)
        hash_expr = f"substr(md5({parts} || {_v(salt)}), 1, 16) AS {_q(new_column)}"
        return self._con.sql(f"SELECT *, {hash_expr} FROM {_sub(data)}")

    def col_coalesce(
        self, data: duckdb.DuckDBPyRelation, columns: list[str], new_column: str
    ) -> duckdb.DuckDBPyRelation:
        # Cast all to VARCHAR to handle mixed types (like Polars does)
        coalesce_expr = ", ".join(f"{_q(c)}::VARCHAR" for c in columns)
        return self._con.sql(
            f"SELECT *, COALESCE({coalesce_expr}) AS {_q(new_column)} FROM {_sub(data)}"
        )

    def col_expr(
        self,
        data: duckdb.DuckDBPyRelation,
        new_column: str,
        expr: str,
        dtype: str | None,
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, ({expr}) AS {_q(new_column)} FROM {_sub(data)}"
        )

    # =========================================================================
    # Math operations (27)
    # =========================================================================

    def math_add(
        self, data: duckdb.DuckDBPyRelation, column: str, value: Numeric
    ) -> duckdb.DuckDBPyRelation:
        return self._replace_col(data, column, f"{_q(column)} + {_v(value)}")

    def math_subtract(
        self, data: duckdb.DuckDBPyRelation, column: str, value: Numeric
    ) -> duckdb.DuckDBPyRelation:
        return self._replace_col(data, column, f"{_q(column)} - {_v(value)}")

    def math_multiply(
        self, data: duckdb.DuckDBPyRelation, column: str, value: Numeric
    ) -> duckdb.DuckDBPyRelation:
        return self._replace_col(data, column, f"{_q(column)} * {_v(value)}")

    def math_divide(
        self, data: duckdb.DuckDBPyRelation, column: str, value: Numeric
    ) -> duckdb.DuckDBPyRelation:
        return self._replace_col(data, column, f"{_q(column)} / {_v(value)}")

    def math_clamp(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        expr = qc
        if lower is not None:
            expr = f"GREATEST({_v(lower)}, {expr})"
        if upper is not None:
            expr = f"LEAST({_v(upper)}, {expr})"
        return self._replace_col(data, column, expr)

    def math_abs(
        self, data: duckdb.DuckDBPyRelation, column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._replace_col(data, column, f"ABS({_q(column)})")

    def math_round(
        self, data: duckdb.DuckDBPyRelation, column: str, decimals: int
    ) -> duckdb.DuckDBPyRelation:
        return self._replace_col(data, column, f"ROUND({_q(column)}, {decimals})")

    def math_set_min(
        self, data: duckdb.DuckDBPyRelation, column: str, min_value: Numeric
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        expr = f"CASE WHEN {qc} < {_v(min_value)} THEN {_v(min_value)} ELSE {qc} END"
        return self._replace_col(data, column, expr)

    def math_set_max(
        self, data: duckdb.DuckDBPyRelation, column: str, max_value: Numeric
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        expr = f"CASE WHEN {qc} > {_v(max_value)} THEN {_v(max_value)} ELSE {qc} END"
        return self._replace_col(data, column, expr)

    def math_add_columns(
        self,
        data: duckdb.DuckDBPyRelation,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, {_q(column_a)} + {_q(column_b)} AS {_q(new_column)} "
            f"FROM {_sub(data)}"
        )

    def math_subtract_columns(
        self,
        data: duckdb.DuckDBPyRelation,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, {_q(column_a)} - {_q(column_b)} AS {_q(new_column)} "
            f"FROM {_sub(data)}"
        )

    def math_multiply_columns(
        self,
        data: duckdb.DuckDBPyRelation,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, {_q(column_a)} * {_q(column_b)} AS {_q(new_column)} "
            f"FROM {_sub(data)}"
        )

    def math_divide_columns(
        self,
        data: duckdb.DuckDBPyRelation,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, {_q(column_a)} / {_q(column_b)} AS {_q(new_column)} "
            f"FROM {_sub(data)}"
        )

    def math_percent_of(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        total_column: str,
        new_column: str,
        multiply_by: float,
    ) -> duckdb.DuckDBPyRelation:
        expr = (
            f"{_q(column)} / {_q(total_column)} * {_v(multiply_by)} AS {_q(new_column)}"
        )
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def math_cumsum(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        new_column: str,
        group_by: list[str] | None,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        partition = ""
        if group_by:
            partition = "PARTITION BY " + ", ".join(_q(g) for g in group_by) + " "
        # Use ROW_NUMBER to preserve input order
        expr = (
            f"SUM({qc}) OVER ({partition}"
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
            f"AS {_q(new_column)}"
        )
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def math_rank(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        new_column: str,
        method: RankMethod,
        descending: bool,  # noqa: FBT001
        group_by: list[str] | None,
    ) -> duckdb.DuckDBPyRelation:
        order = "DESC" if descending else "ASC"
        partition = ""
        if group_by:
            partition = "PARTITION BY " + ", ".join(_q(g) for g in group_by) + " "

        # Map rank methods to DuckDB window functions
        rank_func_map = {
            "average": "RANK",
            "min": "RANK",
            "max": "RANK",
            "dense": "DENSE_RANK",
            "ordinal": "ROW_NUMBER",
            "random": "ROW_NUMBER",
        }
        func = rank_func_map.get(method, "RANK")
        expr = (
            f"CAST({func}() OVER ({partition}ORDER BY {_q(column)} {order}) "
            f"AS BIGINT) AS {_q(new_column)}"
        )
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    _VALID_AGG_MAP: ClassVar[dict[str, str]] = {
        "min": "MIN",
        "max": "MAX",
        "mean": "AVG",
        "median": "MEDIAN",
        "sum": "SUM",
        "first": "FIRST",
        "count": "COUNT",
    }

    def math_diff_from_agg(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        agg: str,
        new_column: str,
        group_by: list[str] | None,
    ) -> duckdb.DuckDBPyRelation:
        if agg not in self._VALID_AGG_MAP:
            msg = f"Invalid aggregate function: {agg}"
            raise ValueError(msg)
        func = self._VALID_AGG_MAP[agg]
        partition = ""
        if group_by:
            partition = "PARTITION BY " + ", ".join(_q(g) for g in group_by)
        expr = (
            f"({_q(column)} - {func}({_q(column)}) OVER ({partition})) "
            f"AS {_q(new_column)}"
        )
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def math_diff_lag(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        order_by: list[str],
        new_column: str,
        group_by: list[str] | None,
        lag: int,
    ) -> duckdb.DuckDBPyRelation:
        partition = ""
        if group_by:
            partition = "PARTITION BY " + ", ".join(_q(g) for g in group_by)
        order = "ORDER BY " + ", ".join(_q(o) for o in order_by)
        window = f"{partition} {order}".strip()
        expr = (
            f"({_q(column)} - LAG({_q(column)}, {lag}) OVER ({window})) "
            f"AS {_q(new_column)}"
        )
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def math_standardize(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        mean: Numeric | None,
        std: Numeric | None,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        if mean is not None and std is not None:
            computed_mean = float(mean)
            computed_std = float(std)
        else:
            stats = self._con.sql(
                f"SELECT AVG({qc}) AS m, STDDEV_SAMP({qc}) AS s FROM {_sub(data)}"
            ).fetchone()
            assert stats is not None
            computed_mean = float(mean if mean is not None else (stats[0] or 0.0))
            computed_std = float(std if std is not None else (stats[1] or 0.0))

        if computed_std == 0:
            return self._con.sql(f"SELECT *, 0.0 AS {_q(new_column)} FROM {_sub(data)}")
        expr = f"({qc} - {_v(computed_mean)}) / {_v(computed_std)} AS {_q(new_column)}"
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def math_minmax(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        min_val: Numeric | None,
        max_val: Numeric | None,
        feature_range: FeatureRange,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        if min_val is not None and max_val is not None:
            cmin = float(min_val)
            cmax = float(max_val)
        else:
            stats = self._con.sql(
                f"SELECT MIN({qc}), MAX({qc}) FROM {_sub(data)}"
            ).fetchone()
            assert stats is not None
            cmin = float(min_val if min_val is not None else (stats[0] or 0.0))
            cmax = float(max_val if max_val is not None else (stats[1] or 0.0))

        a, b = feature_range
        if cmax == cmin:
            mid = (a + b) / 2
            return self._con.sql(
                f"SELECT *, {_v(mid)} AS {_q(new_column)} FROM {_sub(data)}"
            )
        expr = (
            f"{_v(a)} + ({qc} - {_v(cmin)}) * {_v(b - a)} / {_v(cmax - cmin)} "
            f"AS {_q(new_column)}"
        )
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def math_robust_scale(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        median: Numeric | None,
        iqr: Numeric | None,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        if median is not None and iqr is not None:
            cmed = float(median)
            ciqr = float(iqr)
        else:
            stats = self._con.sql(
                f"SELECT MEDIAN({qc}), "
                f"QUANTILE_CONT({qc}, 0.75) - QUANTILE_CONT({qc}, 0.25) "
                f"FROM {_sub(data)}"
            ).fetchone()
            assert stats is not None
            cmed = float(median if median is not None else (stats[0] or 0.0))
            ciqr = float(iqr if iqr is not None else (stats[1] or 0.0))

        if ciqr == 0:
            return self._con.sql(f"SELECT *, 0.0 AS {_q(new_column)} FROM {_sub(data)}")
        expr = f"({qc} - {_v(cmed)}) / {_v(ciqr)} AS {_q(new_column)}"
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def math_log(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        base: Numeric | None,
        offset: Numeric,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        inner = f"({qc} + {_v(offset)})"
        if base is None:
            expr = f"LN({inner})"
        elif base == 10:
            expr = f"LOG10({inner})"
        else:
            expr = f"LN({inner}) / {_v(math.log(base))}"
        return self._con.sql(f"SELECT *, {expr} AS {_q(new_column)} FROM {_sub(data)}")

    def math_sqrt(
        self, data: duckdb.DuckDBPyRelation, column: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, SQRT({_q(column)}) AS {_q(new_column)} FROM {_sub(data)}"
        )

    def math_power(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        exponent: Numeric,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, POWER({_q(column)}, {_v(exponent)}) AS {_q(new_column)} "
            f"FROM {_sub(data)}"
        )

    def math_winsorize(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
        lower_value: Numeric | None,
        upper_value: Numeric | None,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        # Compute bounds
        lower_bound: float | None = (
            float(lower_value) if lower_value is not None else None
        )
        if lower_bound is None and lower is not None:
            row = self._con.sql(
                f"SELECT QUANTILE_CONT({qc}, {_v(lower)}) FROM {_sub(data)}"
            ).fetchone()
            lower_bound = float(row[0]) if row else None

        upper_bound: float | None = (
            float(upper_value) if upper_value is not None else None
        )
        if upper_bound is None and upper is not None:
            row = self._con.sql(
                f"SELECT QUANTILE_CONT({qc}, {_v(upper)}) FROM {_sub(data)}"
            ).fetchone()
            upper_bound = float(row[0]) if row else None

        expr = qc
        if lower_bound is not None:
            expr = f"GREATEST({_v(lower_bound)}, {expr})"
        if upper_bound is not None:
            expr = f"LEAST({_v(upper_bound)}, {expr})"
        return self._con.sql(f"SELECT *, {expr} AS {_q(new_column)} FROM {_sub(data)}")

    # =========================================================================
    # Row operations (14)
    # =========================================================================

    def rows_filter(
        self, data: duckdb.DuckDBPyRelation, filter: dict[str, Any]
    ) -> duckdb.DuckDBPyRelation:
        sql_where = Filter.from_dict(filter).to_sql()
        return self._con.sql(f"SELECT * FROM {_sub(data)} WHERE {sql_where}")

    def rows_drop(
        self, data: duckdb.DuckDBPyRelation, filter: dict[str, Any]
    ) -> duckdb.DuckDBPyRelation:
        sql_where = Filter.from_dict(filter).to_sql()
        return self._con.sql(f"SELECT * FROM {_sub(data)} WHERE NOT ({sql_where})")

    def rows_drop_nulls(
        self, data: duckdb.DuckDBPyRelation, columns: list[str] | None
    ) -> duckdb.DuckDBPyRelation:
        target_cols = columns if columns is not None else list(data.columns)
        conditions = " AND ".join(f"{_q(c)} IS NOT NULL" for c in target_cols)
        return self._con.sql(f"SELECT * FROM {_sub(data)} WHERE {conditions}")

    def rows_flag(
        self,
        data: duckdb.DuckDBPyRelation,
        filter: dict[str, Any],
        new_column: str,
        true_value: Any,
        false_value: Any,
    ) -> duckdb.DuckDBPyRelation:
        sql_where = Filter.from_dict(filter).to_sql()
        expr = (
            f"CASE WHEN {sql_where} THEN {_v(true_value)} "
            f"ELSE {_v(false_value)} END AS {_q(new_column)}"
        )
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def rows_unique(
        self,
        data: duckdb.DuckDBPyRelation,
        columns: list[str] | None,
        keep: Literal["first", "last", "any", "none"],
    ) -> duckdb.DuckDBPyRelation:
        target_cols = columns if columns is not None else list(data.columns)
        partition = ", ".join(_q(c) for c in target_cols)
        all_cols = ", ".join(_q(c) for c in data.columns)

        if keep == "none":
            # Keep only rows that appear exactly once
            return self._con.sql(
                f"SELECT {all_cols} FROM ("
                f"SELECT *, COUNT(*) OVER (PARTITION BY {partition}) AS _cnt "
                f"FROM {_sub(data)}) AS _r WHERE _cnt = 1"
            )
        if keep == "last":
            return self._con.sql(
                f"SELECT {all_cols} FROM ("
                f"SELECT *, ROW_NUMBER() OVER "
                f"(PARTITION BY {partition} ORDER BY "
                f"ROWID DESC) AS _rn FROM {_sub(data)}) AS _r "
                f"WHERE _rn = 1"
            )
        # "first" or "any" — keep first occurrence
        return self._con.sql(
            f"SELECT {all_cols} FROM ("
            f"SELECT *, ROW_NUMBER() OVER "
            f"(PARTITION BY {partition}) AS _rn "
            f"FROM {_sub(data)}) AS _r WHERE _rn = 1"
        )

    def rows_deduplicate(
        self,
        data: duckdb.DuckDBPyRelation,
        columns: list[str],
        sort_by: str,
        keep: Literal["first", "last"],
        descending: bool,  # noqa: FBT001
    ) -> duckdb.DuckDBPyRelation:
        partition = ", ".join(_q(c) for c in columns)
        order_dir = "DESC" if descending else "ASC"
        # For "last", reverse the sort
        if keep == "last":
            order_dir = "ASC" if descending else "DESC"
        all_cols = ", ".join(_q(c) for c in data.columns)
        return self._con.sql(
            f"SELECT {all_cols} FROM ("
            f"SELECT *, ROW_NUMBER() OVER "
            f"(PARTITION BY {partition} ORDER BY {_q(sort_by)} {order_dir}) "
            f"AS _rn FROM {_sub(data)}) AS _r WHERE _rn = 1"
        )

    def rows_sort(
        self,
        data: duckdb.DuckDBPyRelation,
        by: list[str],
        descending: bool | Sequence[bool],  # noqa: FBT001
    ) -> duckdb.DuckDBPyRelation:
        if isinstance(descending, bool):
            desc_list = [descending] * len(by)
        else:
            desc_list = list(descending)
        order_parts = [
            f"{_q(col)} {'DESC' if desc else 'ASC'}"
            for col, desc in zip(by, desc_list, strict=False)
        ]
        order_clause = ", ".join(order_parts)
        return self._con.sql(f"SELECT * FROM {_sub(data)} ORDER BY {order_clause}")

    def rows_head(
        self, data: duckdb.DuckDBPyRelation, n: int
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(f"SELECT * FROM {_sub(data)} LIMIT {n}")

    def rows_tail(
        self, data: duckdb.DuckDBPyRelation, n: int
    ) -> duckdb.DuckDBPyRelation:
        all_cols = ", ".join(_q(c) for c in data.columns)
        return self._con.sql(
            f"SELECT {all_cols} FROM ("
            f"SELECT *, ROW_NUMBER() OVER () AS _rn, "
            f"COUNT(*) OVER () AS _cnt "
            f"FROM {_sub(data)}) AS _r WHERE _rn > _cnt - {n}"
        )

    def rows_sample(
        self,
        data: duckdb.DuckDBPyRelation,
        n: int | None,
        fraction: float | None,
        seed: int | None,
    ) -> duckdb.DuckDBPyRelation:
        seed_clause = f" REPEATABLE({seed})" if seed is not None else ""
        if n is not None:
            return self._con.sql(
                f"SELECT * FROM {_sub(data)} USING SAMPLE {n} ROWS{seed_clause}"
            )
        if fraction is not None:
            pct = fraction * 100
            return self._con.sql(
                f"SELECT * FROM {_sub(data)} USING SAMPLE {pct} PERCENT{seed_clause}"
            )
        return data

    def rows_explode(
        self, data: duckdb.DuckDBPyRelation, column: str
    ) -> duckdb.DuckDBPyRelation:
        other_cols = [c for c in data.columns if c != column]
        other_cols_list = ", ".join(f"_t.{_q(c)}" for c in other_cols)
        qc = _q(column)
        if other_cols:
            return self._con.sql(
                f"SELECT {other_cols_list}, UNNEST(_t.{qc}) AS {qc} FROM {_sub(data)}"
            )
        return self._con.sql(f"SELECT UNNEST(_t.{qc}) AS {qc} FROM {_sub(data)}")

    def rows_melt(
        self,
        data: duckdb.DuckDBPyRelation,
        id_columns: list[str],
        value_columns: list[str],
        variable_name: str,
        value_name: str,
    ) -> duckdb.DuckDBPyRelation:
        val_cols = ", ".join(_q(c) for c in value_columns)
        return self._con.sql(
            f"UNPIVOT {_sub(data)} "
            f"ON {val_cols} "
            f"INTO NAME {_q(variable_name)} VALUE {_q(value_name)}"
        )

    def rows_pivot(
        self,
        data: duckdb.DuckDBPyRelation,
        index: list[str],
        columns: str,
        values: str,
        aggregate_function: AggFunction,
    ) -> duckdb.DuckDBPyRelation:
        if aggregate_function not in _VALID_AGG_FUNCTIONS:
            msg = f"Invalid aggregate function: {aggregate_function}"
            raise ValueError(msg)
        idx = ", ".join(_q(c) for c in index)
        return self._con.sql(
            f"PIVOT {_sub(data)} "
            f"ON {_q(columns)} "
            f"USING {aggregate_function}({_q(values)}) "
            f"GROUP BY {idx}"
        )

    # =========================================================================
    # String operations (10)
    # =========================================================================

    def str_replace(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        pattern: str,
        replacement: str,
        literal: bool,  # noqa: FBT001
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        if literal:
            expr = f"REPLACE({qc}, {_v(pattern)}, {_v(replacement)})"
        else:
            expr = f"regexp_replace({qc}, {_v(pattern)}, {_v(replacement)}, 'g')"
        return self._replace_col(data, column, expr)

    def str_slice(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        offset: int,
        length: int | None,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        # DuckDB substr is 1-indexed
        start = offset + 1
        if length is not None:
            expr = f"SUBSTR({qc}, {start}, {length})"
        else:
            expr = f"SUBSTR({qc}, {start})"
        return self._replace_col(data, column, expr)

    def str_truncate(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        max_length: int,
        suffix: str,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        cut_length = max_length - len(suffix)
        expr = (
            f"CASE WHEN LENGTH({qc}) > {max_length} "
            f"THEN SUBSTR({qc}, 1, {cut_length}) || {_v(suffix)} "
            f"ELSE {qc} END"
        )
        return self._replace_col(data, column, expr)

    def str_split(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        separator: str,
        new_columns: list[str] | None,
        keep_original: bool,  # noqa: FBT001
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        if new_columns is None:
            # Split into list then explode
            result = self._replace_col(
                data, column, f"string_split({qc}, {_v(separator)})"
            )
            return self.rows_explode(result, column)

        # Generate split_part for each column
        parts = [
            f"split_part({qc}, {_v(separator)}, {i + 1}) AS {_q(nc)}"
            for i, nc in enumerate(new_columns)
        ]
        parts_sql = ", ".join(parts)
        if keep_original:
            return self._con.sql(f"SELECT *, {parts_sql} FROM {_sub(data)}")
        # Drop original, keep rest plus new
        other_cols = [c for c in data.columns if c != column]
        other_sql = ", ".join(_q(c) for c in other_cols)
        prefix = f"{other_sql}, " if other_sql else ""
        return self._con.sql(f"SELECT {prefix}{parts_sql} FROM {_sub(data)}")

    def str_lower(
        self, data: duckdb.DuckDBPyRelation, column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._replace_col(data, column, f"LOWER({_q(column)})")

    def str_upper(
        self, data: duckdb.DuckDBPyRelation, column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._replace_col(data, column, f"UPPER({_q(column)})")

    def str_strip(
        self, data: duckdb.DuckDBPyRelation, column: str, chars: str | None
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        expr = f"TRIM({qc})" if chars is None else f"TRIM(BOTH {_v(chars)} FROM {qc})"
        return self._replace_col(data, column, expr)

    def str_pad(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        length: int,
        fill_char: str,
        side: str,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        if side == "left":
            expr = f"LPAD({qc}, {length}, {_v(fill_char)})"
        else:
            expr = f"RPAD({qc}, {length}, {_v(fill_char)})"
        return self._replace_col(data, column, expr)

    def str_concat(
        self,
        data: duckdb.DuckDBPyRelation,
        columns: list[str],
        new_column: str,
        separator: str,
    ) -> duckdb.DuckDBPyRelation:
        parts = f" || {_v(separator)} || ".join(
            f"COALESCE({_q(c)}::VARCHAR, '')" for c in columns
        )
        return self._con.sql(f"SELECT *, {parts} AS {_q(new_column)} FROM {_sub(data)}")

    def str_extract(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        pattern: str,
        group_index: int,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        expr = f"regexp_extract({_q(column)}, {_v(pattern)}, {group_index})"
        return self._con.sql(f"SELECT *, {expr} AS {_q(new_column)} FROM {_sub(data)}")

    # =========================================================================
    # Datetime operations (13)
    # =========================================================================

    def dt_year(
        self, data: duckdb.DuckDBPyRelation, column: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, YEAR({_q(column)}) AS {_q(new_column)} FROM {_sub(data)}"
        )

    def dt_month(
        self, data: duckdb.DuckDBPyRelation, column: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, MONTH({_q(column)}) AS {_q(new_column)} FROM {_sub(data)}"
        )

    def dt_day(
        self, data: duckdb.DuckDBPyRelation, column: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, DAY({_q(column)}) AS {_q(new_column)} FROM {_sub(data)}"
        )

    def dt_week(
        self, data: duckdb.DuckDBPyRelation, column: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, WEEKOFYEAR({_q(column)}) AS {_q(new_column)} FROM {_sub(data)}"
        )

    def dt_quarter(
        self, data: duckdb.DuckDBPyRelation, column: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, QUARTER({_q(column)}) AS {_q(new_column)} FROM {_sub(data)}"
        )

    def dt_year_month(
        self, data: duckdb.DuckDBPyRelation, column: str, new_column: str, fmt: str
    ) -> duckdb.DuckDBPyRelation:
        ddb_fmt = self._polars_fmt_to_duckdb(fmt)
        return self._con.sql(
            f"SELECT *, strftime({_q(column)}, {_v(ddb_fmt)}) AS {_q(new_column)} "
            f"FROM {_sub(data)}"
        )

    def dt_quarter_year(
        self, data: duckdb.DuckDBPyRelation, column: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        expr = f"'Q' || QUARTER({qc})::VARCHAR || '-' || YEAR({qc})::VARCHAR"
        return self._con.sql(f"SELECT *, {expr} AS {_q(new_column)} FROM {_sub(data)}")

    def dt_calendar_week(
        self, data: duckdb.DuckDBPyRelation, column: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        expr = (
            f"ISOYEAR({qc})::VARCHAR || '-W' || LPAD(WEEKOFYEAR({qc})::VARCHAR, 2, '0')"
        )
        return self._con.sql(f"SELECT *, {expr} AS {_q(new_column)} FROM {_sub(data)}")

    def dt_parse(
        self, data: duckdb.DuckDBPyRelation, column: str, fmt: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        ddb_fmt = self._polars_fmt_to_duckdb(fmt)
        return self._con.sql(
            f"SELECT *, strptime({_q(column)}::VARCHAR, {_v(ddb_fmt)})::DATE "
            f"AS {_q(new_column)} FROM {_sub(data)}"
        )

    def dt_format(
        self, data: duckdb.DuckDBPyRelation, column: str, fmt: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        ddb_fmt = self._polars_fmt_to_duckdb(fmt)
        return self._con.sql(
            f"SELECT *, strftime({_q(column)}, {_v(ddb_fmt)}) AS {_q(new_column)} "
            f"FROM {_sub(data)}"
        )

    def dt_diff_days(
        self,
        data: duckdb.DuckDBPyRelation,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        return self._con.sql(
            f"SELECT *, DATEDIFF('day', {_q(column_b)}, {_q(column_a)}) "
            f"AS {_q(new_column)} FROM {_sub(data)}"
        )

    def dt_age_years(
        self,
        data: duckdb.DuckDBPyRelation,
        birth_column: str,
        reference_column: str | None,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        ref = _q(reference_column) if reference_column else "CURRENT_DATE"
        expr = f"DATEDIFF('day', {_q(birth_column)}, {ref}) // 365"
        return self._con.sql(f"SELECT *, {expr} AS {_q(new_column)} FROM {_sub(data)}")

    def dt_truncate(
        self, data: duckdb.DuckDBPyRelation, column: str, every: str, new_column: str
    ) -> duckdb.DuckDBPyRelation:
        unit = self._polars_every_to_duckdb(every)
        return self._con.sql(
            f"SELECT *, date_trunc({_v(unit)}, {_q(column)}) "
            f"AS {_q(new_column)} FROM {_sub(data)}"
        )

    def dt_is_between(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        start: str,
        end: str,
        new_column: str,
        closed: ClosedInterval,
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        start_val = f"DATE {_v(start)}"
        end_val = f"DATE {_v(end)}"

        if closed == "both":
            cond = f"{qc} >= {start_val} AND {qc} <= {end_val}"
        elif closed == "left":
            cond = f"{qc} >= {start_val} AND {qc} < {end_val}"
        elif closed == "right":
            cond = f"{qc} > {start_val} AND {qc} <= {end_val}"
        else:  # "none"
            cond = f"{qc} > {start_val} AND {qc} < {end_val}"

        return self._con.sql(
            f"SELECT *, ({cond}) AS {_q(new_column)} FROM {_sub(data)}"
        )

    # =========================================================================
    # Map operations (10)
    # =========================================================================

    def map_values(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        mapping: dict[Any, Any],
        default: Any,
        keep_unmapped: bool,  # noqa: FBT001
    ) -> duckdb.DuckDBPyRelation:
        if not mapping:
            return data
        qc = _q(column)
        cases = " ".join(
            f"WHEN {qc} = {_v(k)} THEN {_v(v)}" for k, v in mapping.items()
        )
        if keep_unmapped:
            expr = f"CASE {cases} ELSE {qc} END"
        else:
            expr = f"CASE {cases} ELSE {_v(default)} END"
        return self._replace_col(data, column, expr)

    def map_case(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        cases: list[tuple[Any, Any]],
        default: Any,
        new_column: str,
    ) -> duckdb.DuckDBPyRelation:
        if not cases:
            return self._con.sql(
                f"SELECT *, {_v(default)} AS {_q(new_column)} FROM {_sub(data)}"
            )
        qc = _q(column)
        case_clauses = " ".join(
            f"WHEN {qc} = {_v(cond)} THEN {_v(result)}" for cond, result in cases
        )
        expr = f"CASE {case_clauses} ELSE {_v(default)} END AS {_q(new_column)}"
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def map_from_column(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        lookup_column: str,
        value_column: str,
        new_column: str,
        default: Any,
    ) -> duckdb.DuckDBPyRelation:
        # Build lookup from the data itself
        lookup_rows = self._con.sql(
            f"SELECT DISTINCT {_q(lookup_column)}, {_q(value_column)} FROM {_sub(data)}"
        ).fetchall()
        if not lookup_rows:
            return self._con.sql(
                f"SELECT *, {_v(default)} AS {_q(new_column)} FROM {_sub(data)}"
            )
        qc = _q(column)
        case_clauses = " ".join(
            f"WHEN {qc} = {_v(row[0])} THEN {_v(row[1])}" for row in lookup_rows
        )
        expr = f"CASE {case_clauses} ELSE {_v(default)} END AS {_q(new_column)}"
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def map_discretize(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        bins: list[float],
        labels: list[str] | None,
        new_column: str,
        right: bool,  # noqa: FBT001
    ) -> duckdb.DuckDBPyRelation:
        edges = [-float("inf"), *bins, float("inf")]
        if labels is None:
            labels = []
            for i in range(len(edges) - 1):
                if right:
                    labels.append(f"({edges[i]}, {edges[i + 1]}]")
                else:
                    labels.append(f"[{edges[i]}, {edges[i + 1]})")

        qc = _q(column)
        case_parts = []
        for i in range(len(edges) - 1):
            lo = edges[i]
            hi = edges[i + 1]
            if right:
                if lo == -float("inf"):
                    cond = f"{qc} <= {_v(hi)}"
                elif hi == float("inf"):
                    cond = f"{qc} > {_v(lo)}"
                else:
                    cond = f"{qc} > {_v(lo)} AND {qc} <= {_v(hi)}"
            elif hi == float("inf"):
                cond = f"{qc} >= {_v(lo)}"
            elif lo == -float("inf"):
                cond = f"{qc} < {_v(hi)}"
            else:
                cond = f"{qc} >= {_v(lo)} AND {qc} < {_v(hi)}"
            case_parts.append(f"WHEN {cond} THEN {_v(labels[i])}")

        case_sql = " ".join(case_parts)
        expr = f"CASE {case_sql} ELSE NULL END AS {_q(new_column)}"
        return self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

    def map_onehot(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        categories: list[Any] | None,
        prefix: str,
        drop: Any | None,
        drop_original: bool,  # noqa: FBT001
        unknown_value: str,
    ) -> duckdb.DuckDBPyRelation:
        if categories is None:
            rows = self._con.sql(
                f"SELECT DISTINCT {_q(column)} FROM {_sub(data)} "
                f"WHERE {_q(column)} IS NOT NULL ORDER BY {_q(column)}"
            ).fetchall()
            categories = [r[0] for r in rows]

        drop_category: Any | None = None
        if drop is not None and categories:
            if drop in categories:
                drop_category = drop
            elif drop == "first":
                drop_category = categories[0]
            elif drop == "last":
                drop_category = categories[-1]
            else:
                drop_category = drop

        qc = _q(column)
        new_cols = []
        for cat in categories:
            if drop_category is not None and cat == drop_category:
                continue
            col_name = f"{prefix}_{cat}"
            if unknown_value == "all_zero":
                expr = f"CASE WHEN {qc} = {_v(cat)} THEN 1 ELSE 0 END AS {_q(col_name)}"
            else:
                cat_list = ", ".join(_v(c) for c in categories)
                expr = (
                    f"CASE WHEN {qc} = {_v(cat)} THEN 1 "
                    f"WHEN {qc} IN ({cat_list}) THEN 0 "
                    f"ELSE NULL END AS {_q(col_name)}"
                )
            new_cols.append(expr)

        new_cols_sql = ", ".join(new_cols) if new_cols else "1 AS _dummy"
        result = self._con.sql(f"SELECT *, {new_cols_sql} FROM {_sub(data)}")

        if drop_original:
            keep = [c for c in result.columns if c != column]
            col_list = ", ".join(_q(c) for c in keep)
            result = self._con.sql(
                f"SELECT {col_list} FROM ({result.sql_query()}) AS _r"
            )

        return result

    def map_ordinal(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        categories: list[Any] | None,
        new_column: str,
        drop_original: bool,  # noqa: FBT001
        unknown_value: int,
    ) -> duckdb.DuckDBPyRelation:
        if categories is None:
            rows = self._con.sql(
                f"SELECT DISTINCT {_q(column)} FROM {_sub(data)} "
                f"WHERE {_q(column)} IS NOT NULL ORDER BY {_q(column)}"
            ).fetchall()
            categories = [r[0] for r in rows]

        if not categories:
            return self._con.sql(
                f"SELECT *, {_v(unknown_value)} AS {_q(new_column)} FROM {_sub(data)}"
            )

        qc = _q(column)
        case_clauses = " ".join(
            f"WHEN {qc} = {_v(cat)} THEN {idx}" for idx, cat in enumerate(categories)
        )
        expr = f"CASE {case_clauses} ELSE {_v(unknown_value)} END AS {_q(new_column)}"
        result = self._con.sql(f"SELECT *, {expr} FROM {_sub(data)}")

        if drop_original and new_column != column:
            keep = [c for c in result.columns if c != column]
            col_list = ", ".join(_q(c) for c in keep)
            result = self._con.sql(
                f"SELECT {col_list} FROM ({result.sql_query()}) AS _r"
            )

        return result

    def map_label(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        categories: list[Any] | None,
        new_column: str,
        drop_original: bool,  # noqa: FBT001
        unknown_value: int,
    ) -> duckdb.DuckDBPyRelation:
        return self.map_ordinal(
            data, column, categories, new_column, drop_original, unknown_value
        )

    def map_bool_to_int(
        self, data: duckdb.DuckDBPyRelation, column: str
    ) -> duckdb.DuckDBPyRelation:
        return self._replace_col(data, column, f"{_q(column)}::BIGINT")

    def map_null_to_value(
        self, data: duckdb.DuckDBPyRelation, column: str, value: Any
    ) -> duckdb.DuckDBPyRelation:
        return self._replace_col(data, column, f"COALESCE({_q(column)}, {_v(value)})")

    def map_value_to_null(
        self, data: duckdb.DuckDBPyRelation, column: str, value: Any
    ) -> duckdb.DuckDBPyRelation:
        qc = _q(column)
        expr = f"CASE WHEN {qc} = {_v(value)} THEN NULL ELSE {qc} END"
        return self._replace_col(data, column, expr)

    # =========================================================================
    # Join operations (1)
    # =========================================================================

    def join(
        self,
        data: duckdb.DuckDBPyRelation,
        right_data: duckdb.DuckDBPyRelation,
        on: list[str],
        how: str,
        suffix: str,
        left_on: list[str] | None = None,
        right_on: list[str] | None = None,
        select_columns: list[str] | None = None,
    ) -> duckdb.DuckDBPyRelation:
        left_cols = left_on or on
        right_cols = right_on or on
        right_join_col_set = set(right_cols)

        # Determine which right-side columns to include
        right_schema_cols = list(right_data.columns)
        if select_columns is not None:
            right_keep = list(dict.fromkeys(right_cols + list(select_columns)))
            right_schema_cols = [c for c in right_schema_cols if c in set(right_keep)]

        left_col_set = set(data.columns)

        # Build SELECT list
        select_parts = [f"_left.{_q(c)}" for c in data.columns]
        for c in right_schema_cols:
            if c in right_join_col_set:
                continue
            alias = f"{c}{suffix}" if c in left_col_set else c
            select_parts.append(f"_right.{_q(c)} AS {_q(alias)}")

        select_sql = ", ".join(select_parts)

        # Build ON clause
        on_parts = [
            f"_left.{_q(lc)} = _right.{_q(rc)}"
            for lc, rc in zip(left_cols, right_cols, strict=False)
        ]
        on_sql = " AND ".join(on_parts)

        join_type = "INNER" if how == "inner" else "LEFT"
        left_sub = f"({data.sql_query()}) AS _left"
        right_sub = f"({right_data.sql_query()}) AS _right"

        sql = (
            f"SELECT {select_sql} FROM {left_sub} "
            f"{join_type} JOIN {right_sub} ON {on_sql}"
        )
        return self._con.sql(sql)

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _replace_col(
        self,
        data: duckdb.DuckDBPyRelation,
        column: str,
        expr: str,
    ) -> duckdb.DuckDBPyRelation:
        """Replace a column with an expression, preserving column order.

        Returns:
            New relation with the column replaced.
        """
        cols = [f"{expr} AS {_q(c)}" if c == column else _q(c) for c in data.columns]
        col_list = ", ".join(cols)
        return self._con.sql(f"SELECT {col_list} FROM {_sub(data)}")

    @staticmethod
    def _polars_fmt_to_duckdb(fmt: str) -> str:
        """Convert Polars strftime format to DuckDB strftime format.

        They are largely compatible (both use C strftime conventions).

        Returns:
            DuckDB format string.
        """
        return fmt

    @staticmethod
    def _polars_every_to_duckdb(every: str) -> str:
        """Map Polars truncation 'every' strings to DuckDB date_trunc units.

        Returns:
            DuckDB date_trunc unit string.
        """
        mapping = {
            "1d": "day",
            "1w": "week",
            "1mo": "month",
            "1q": "quarter",
            "1y": "year",
            "1h": "hour",
            "1m": "minute",
            "1s": "second",
        }
        return mapping.get(every, every)
