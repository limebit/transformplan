"""Row operations mixin.

This module provides the RowOps mixin class with operations for filtering,
sorting, and transforming DataFrame rows.

Classes:
    RowOps: Mixin providing row-level operations.

Filtering Operations:
    rows_filter: Keep rows matching filter.
    rows_drop: Drop rows matching filter.
    rows_drop_nulls: Drop rows with null values.
    rows_flag: Add flag column based on condition.

Deduplication:
    rows_unique: Keep unique rows.
    rows_deduplicate: Deduplicate with sort order.

Ordering:
    rows_sort: Sort by columns.
    rows_head: Keep first n rows.
    rows_tail: Keep last n rows.
    rows_sample: Random sample.

Reshaping:
    rows_explode: Explode list column.
    rows_melt: Wide to long format.
    rows_pivot: Long to wide format.

Example:
    >>> from transformplan import Col
    >>> plan = TransformPlan().rows_filter(Col("age") >= 18).rows_sort("name")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

import polars as pl

from ..filters import Filter

if TYPE_CHECKING:
    from typing import Any, Callable

    from typing_extensions import Self


class RowOps:
    """Mixin providing row-level operations."""

    if TYPE_CHECKING:

        def _register(
            self,
            method: Callable[..., pl.DataFrame],
            params: dict[str, Any],
        ) -> Self: ...

    def rows_drop_nulls(self, columns: str | Sequence[str] | None = None) -> Self:
        """Drop rows with null values in specified columns (or any column if None).

        Returns:
            Self for method chaining.
        """
        if isinstance(columns, str):
            columns = [columns]
        return self._register(self._rows_drop_nulls, {"columns": columns})

    def _rows_drop_nulls(
        self, data: pl.DataFrame, columns: list[str] | None
    ) -> pl.DataFrame:
        return data.drop_nulls(subset=columns)

    def rows_unique(
        self,
        columns: str | Sequence[str] | None = None,
        keep: Literal["first", "last", "any", "none"] = "first",
    ) -> Self:
        """Keep unique rows based on specified columns.

        Returns:
            Self for method chaining.
        """
        if isinstance(columns, str):
            columns = [columns]
        return self._register(self._rows_unique, {"columns": columns, "keep": keep})

    def _rows_unique(
        self,
        data: pl.DataFrame,
        columns: list[str] | None,
        keep: Literal["first", "last", "any", "none"],
    ) -> pl.DataFrame:
        return data.unique(subset=columns, keep=keep)

    def rows_filter(self, filter: Filter | dict[str, Any]) -> Self:
        """Filter rows using a serializable Filter expression.

        Returns:
            Self for method chaining.

        Example:
            from transformplan.filters import Col

            .rows_filter(Col("age") > 18)
            .rows_filter((Col("status") == "active") & (Col("score") >= 50))
        """
        filter_dict = filter if isinstance(filter, dict) else filter.to_dict()
        return self._register(self._rows_filter, {"filter": filter_dict})

    def _rows_filter(self, data: pl.DataFrame, filter: dict[str, Any]) -> pl.DataFrame:
        expr = Filter.from_dict(filter).to_expr()
        return data.filter(expr)

    def rows_drop(self, filter: Filter | dict[str, Any]) -> Self:
        """Drop rows matching a filter (inverse of rows_filter).

        Returns:
            Self for method chaining.

        Example:
            .rows_drop(Col("status") == "deleted")
        """
        filter_dict = filter if isinstance(filter, dict) else filter.to_dict()
        return self._register(self._rows_drop, {"filter": filter_dict})

    def _rows_drop(self, data: pl.DataFrame, filter: dict[str, Any]) -> pl.DataFrame:
        expr = Filter.from_dict(filter).to_expr()
        return data.filter(~expr)

    def rows_deduplicate(
        self,
        columns: str | Sequence[str],
        sort_by: str,
        keep: Literal["first", "last"] = "first",
        *,
        descending: bool = False,
    ) -> Self:
        """Deduplicate rows by keeping first/last based on sort order.

        Args:
            columns: Columns that define duplicates.
            sort_by: Column to sort by before deduplication.
            keep: Keep 'first' or 'last' after sorting.
            descending: Sort in descending order.

        Returns:
            Self for method chaining.
        """
        if isinstance(columns, str):
            columns = [columns]
        return self._register(
            self._rows_deduplicate,
            {
                "columns": list(columns),
                "sort_by": sort_by,
                "keep": keep,
                "descending": descending,
            },
        )

    def _rows_deduplicate(
        self,
        data: pl.DataFrame,
        columns: list[str],
        sort_by: str,
        keep: Literal["first", "last"],
        descending: bool,  # noqa: FBT001
    ) -> pl.DataFrame:
        sorted_data = data.sort(sort_by, descending=descending)
        return sorted_data.unique(subset=columns, keep=keep, maintain_order=True)

    def rows_explode(self, column: str) -> Self:
        """Explode a list column into multiple rows.

        Returns:
            Self for method chaining.
        """
        return self._register(self._rows_explode, {"column": column})

    def _rows_explode(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.explode(column)

    def rows_melt(
        self,
        id_columns: Sequence[str],
        value_columns: Sequence[str],
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> Self:
        """Unpivot a DataFrame from wide to long format.

        Args:
            id_columns: Columns to keep as identifiers.
            value_columns: Columns to unpivot.
            variable_name: Name for the variable column.
            value_name: Name for the value column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._rows_melt,
            {
                "id_columns": list(id_columns),
                "value_columns": list(value_columns),
                "variable_name": variable_name,
                "value_name": value_name,
            },
        )

    def _rows_melt(
        self,
        data: pl.DataFrame,
        id_columns: list[str],
        value_columns: list[str],
        variable_name: str,
        value_name: str,
    ) -> pl.DataFrame:
        return data.unpivot(
            index=id_columns,
            on=value_columns,
            variable_name=variable_name,
            value_name=value_name,
        )

    def rows_sample(
        self,
        n: int | None = None,
        fraction: float | None = None,
        seed: int | None = None,
    ) -> Self:
        """Sample rows from the DataFrame.

        Args:
            n: Number of rows to sample.
            fraction: Fraction of rows to sample (0.0 to 1.0).
            seed: Random seed for reproducibility.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._rows_sample, {"n": n, "fraction": fraction, "seed": seed}
        )

    def _rows_sample(
        self,
        data: pl.DataFrame,
        n: int | None,
        fraction: float | None,
        seed: int | None,
    ) -> pl.DataFrame:
        return data.sample(n=n, fraction=fraction, seed=seed)

    def rows_head(self, n: int = 5) -> Self:
        """Keep only the first n rows.

        Returns:
            Self for method chaining.
        """
        return self._register(self._rows_head, {"n": n})

    def _rows_head(self, data: pl.DataFrame, n: int) -> pl.DataFrame:
        return data.head(n)

    def rows_tail(self, n: int = 5) -> Self:
        """Keep only the last n rows.

        Returns:
            Self for method chaining.
        """
        return self._register(self._rows_tail, {"n": n})

    def _rows_tail(self, data: pl.DataFrame, n: int) -> pl.DataFrame:
        return data.tail(n)

    def rows_sort(
        self,
        by: str | Sequence[str],
        *,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        """Sort rows by one or more columns.

        Args:
            by: Column(s) to sort by.
            descending: Sort direction (single bool or list matching columns).

        Returns:
            Self for method chaining.
        """
        if isinstance(by, str):
            by = [by]
        return self._register(
            self._rows_sort, {"by": list(by), "descending": descending}
        )

    def _rows_sort(
        self,
        data: pl.DataFrame,
        by: list[str],
        descending: bool | Sequence[bool],  # noqa: FBT001
    ) -> pl.DataFrame:
        return data.sort(by, descending=descending)

    def rows_flag(
        self,
        filter: Filter | dict[str, Any],
        new_column: str,
        *,
        true_value: Any = True,  # noqa: ANN401
        false_value: Any = False,  # noqa: ANN401
    ) -> Self:
        """Add a flag column based on a filter condition (without dropping rows).

        Args:
            filter: Filter condition.
            new_column: Name for the flag column.
            true_value: Value when condition is True.
            false_value: Value when condition is False.

        Returns:
            Self for method chaining.
        """
        filter_dict = filter if isinstance(filter, dict) else filter.to_dict()
        return self._register(
            self._rows_flag,
            {
                "filter": filter_dict,
                "new_column": new_column,
                "true_value": true_value,
                "false_value": false_value,
            },
        )

    def _rows_flag(
        self,
        data: pl.DataFrame,
        filter: dict[str, Any],
        new_column: str,
        true_value: Any,  # noqa: ANN401
        false_value: Any,  # noqa: ANN401
    ) -> pl.DataFrame:
        expr = Filter.from_dict(filter).to_expr()
        return data.with_columns(
            pl.when(expr)
            .then(pl.lit(true_value))
            .otherwise(pl.lit(false_value))
            .alias(new_column)
        )

    def rows_pivot(
        self,
        index: str | Sequence[str],
        columns: str,
        values: str,
        aggregate_function: str = "first",
    ) -> Self:
        """Pivot from long to wide format.

        Args:
            index: Column(s) to use as row identifiers.
            columns: Column whose unique values become new columns.
            values: Column containing values to fill.
            aggregate_function: How to aggregate ('first', 'sum', 'mean', 'count',
                etc.).

        Returns:
            Self for method chaining.
        """
        if isinstance(index, str):
            index = [index]
        return self._register(
            self._rows_pivot,
            {
                "index": list(index),
                "columns": columns,
                "values": values,
                "aggregate_function": aggregate_function,
            },
        )

    def _rows_pivot(
        self,
        data: pl.DataFrame,
        index: list[str],
        columns: str,
        values: str,
        aggregate_function: str,
    ) -> pl.DataFrame:
        return data.pivot(
            index=index,
            on=columns,
            values=values,
            aggregate_function=aggregate_function,
        )
