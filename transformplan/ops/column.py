"""Column operations mixin.

This module provides the ColumnOps mixin class with operations for adding,
dropping, renaming, casting, and transforming DataFrame columns.

Classes:
    ColumnOps: Mixin providing column-level operations.

Operations:
    col_drop: Drop a column.
    col_rename: Rename a column.
    col_cast: Cast column to different dtype.
    col_reorder: Reorder columns (drops unlisted).
    col_select: Keep only specified columns.
    col_duplicate: Copy a column.
    col_fill_null: Fill null values.
    col_drop_null: Drop rows with nulls.
    col_drop_zero: Drop rows with zero values.
    col_add: Add new column with value or expression.
    col_add_uuid: Add column with unique identifiers.
    col_hash: Hash columns into new column.
    col_coalesce: First non-null across columns.
    col_expr: Add column from SQL expression.

Example:
    >>> plan = TransformPlan().col_rename("old", "new").col_drop("temp")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

from transformplan.dtypes import normalize_dtype
from transformplan.ops._common import as_columns

if TYPE_CHECKING:
    from typing import Any

    from typing_extensions import Self

FillNullStrategy = Literal["forward", "backward", "min", "max", "mean", "zero"]


class ColumnOps:
    """Mixin providing column-level operations."""

    if TYPE_CHECKING:

        def _register(
            self,
            op_name: str,
            params: dict[str, Any],
        ) -> Self: ...

        def _register_each(
            self,
            op_name: str,
            params_list: list[dict[str, Any]],
        ) -> Self: ...

    def col_drop(self, column: str | Sequence[str]) -> Self:
        """Drop one or more columns from the DataFrame.

        Args:
            column: Column name, or a sequence of names to drop.

        Returns:
            Self for method chaining.
        """
        return self._register_each(
            "col_drop", [{"column": col} for col in as_columns(column)]
        )

    def col_rename(self, column: str, new_name: str) -> Self:
        """Rename a column.

        Returns:
            Self for method chaining.
        """
        return self._register("col_rename", {"column": column, "new_name": new_name})

    def col_cast(self, column: str | Sequence[str], dtype: str | type) -> Self:
        """Cast one or more columns to a different dtype.

        The dtype is stored under its canonical name (e.g. ``"Float64"``) so the
        plan stays serializable. Polars dtypes and builtin Python types are
        accepted and normalized; each backend resolves the name to its own type.

        Args:
            column: Column name, or a sequence of names to cast.
            dtype: Target dtype as a canonical name (``"Float64"``), a Polars
                dtype (``pl.Float64``), or a Python type (``float``).

        Returns:
            Self for method chaining.
        """
        normalized = normalize_dtype(dtype)
        return self._register_each(
            "col_cast",
            [{"column": col, "dtype": normalized} for col in as_columns(column)],
        )

    def col_reorder(self, columns: Sequence[str]) -> Self:
        """Reorder columns. Unlisted columns are dropped.

        Returns:
            Self for method chaining.
        """
        return self._register("col_reorder", {"columns": list(columns)})

    def col_duplicate(self, column: str, new_name: str) -> Self:
        """Duplicate a column under a new name.

        Returns:
            Self for method chaining.
        """
        return self._register("col_duplicate", {"column": column, "new_name": new_name})

    def col_fill_null(
        self,
        column: str | Sequence[str],
        value: Any = None,  # noqa: ANN401
        strategy: FillNullStrategy | None = None,
    ) -> Self:
        """Fill null values in a column.

        Args:
            column: Column name, or a sequence of names to fill.
            value: Value to fill nulls with (if strategy is None).
            strategy: Fill strategy - 'forward', 'backward', 'mean', 'min', 'max',
                'zero', 'one'.

        Returns:
            Self for method chaining.
        """
        return self._register_each(
            "col_fill_null",
            [
                {"column": col, "value": value, "strategy": strategy}
                for col in as_columns(column)
            ],
        )

    def col_drop_null(self, columns: str | Sequence[str] | None = None) -> Self:
        """Drop rows with null values in specified columns.

        Args:
            columns: Column(s) to check for nulls. If None, checks all columns.

        Returns:
            Self for method chaining.
        """
        if isinstance(columns, str):
            columns = [columns]
        return self._register("col_drop_null", {"columns": columns})

    def col_drop_zero(self, column: str | Sequence[str]) -> Self:
        """Drop rows where the specified column(s) are zero.

        Args:
            column: Column name, or a sequence of names to check.

        Returns:
            Self for method chaining.
        """
        return self._register_each(
            "col_drop_zero", [{"column": col} for col in as_columns(column)]
        )

    def col_add(
        self,
        new_column: str,
        expr: str | float | None = None,
        value: Any = None,  # noqa: ANN401
    ) -> Self:
        """Add a new column with a constant value or expression.

        Args:
            new_column: Name of the new column.
            expr: Column name to copy from, or None for constant value.
            value: Constant value to fill the column with.

        Returns:
            Self for method chaining.
        """
        return self._register(
            "col_add", {"new_column": new_column, "expr": expr, "value": value}
        )

    def col_add_uuid(self, column: str, length: int = 16) -> Self:
        """Add a column with unique random identifiers.

        Args:
            column: Name of the new column.
            length: Length of the identifier string.

        Returns:
            Self for method chaining.
        """
        return self._register("col_add_uuid", {"column": column, "length": length})

    def col_hash(
        self,
        columns: str | Sequence[str],
        new_column: str,
        salt: str = "",
    ) -> Self:
        """Hash one or more columns into a new column.

        Args:
            columns: Column(s) to hash.
            new_column: Name for the hash column.
            salt: Optional salt to add to the hash.

        Returns:
            Self for method chaining.
        """
        if isinstance(columns, str):
            columns = [columns]
        return self._register(
            "col_hash",
            {"columns": list(columns), "new_column": new_column, "salt": salt},
        )

    def col_coalesce(
        self,
        columns: Sequence[str],
        new_column: str,
    ) -> Self:
        """Take the first non-null value across multiple columns.

        Args:
            columns: Columns to coalesce (in priority order).
            new_column: Name for the result column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            "col_coalesce", {"columns": list(columns), "new_column": new_column}
        )

    def col_expr(
        self,
        new_column: str,
        expr: str,
        dtype: str | None = None,
    ) -> Self:
        """Add a new column computed from a SQL expression.

        This is an escape hatch for arbitrary SQL expressions that work
        across both backends. The expression is passed through without
        sanitization — use with trusted input only.

        For Polars, the expression is parsed via ``pl.sql_expr()``.
        For DuckDB, the expression is embedded as raw SQL.

        Args:
            new_column: Name of the new column.
            expr: SQL expression string.
            dtype: Optional type hint for validation. One of
                "float", "string", "integer", "boolean", "date".

        Returns:
            Self for method chaining.
        """
        return self._register(
            "col_expr",
            {"new_column": new_column, "expr": expr, "dtype": dtype},
        )

    def col_select(self, columns: Sequence[str]) -> Self:
        """Keep only the specified columns (order preserved).

        Args:
            columns: Columns to keep.

        Returns:
            Self for method chaining.
        """
        return self._register("col_select", {"columns": list(columns)})
