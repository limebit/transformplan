"""Row operations mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import polars as pl

from ..filters import Filter

if TYPE_CHECKING:
    from typing import Self


class RowOps:
    """Mixin providing row-level operations."""

    def rows_drop_nulls(self, columns: str | Sequence[str] | None = None) -> Self:
        """Drop rows with null values in specified columns (or any column if None)."""
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
        """Keep unique rows based on specified columns."""
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

    def rows_filter(self, filter: Filter | dict) -> Self:
        """Filter rows using a serializable Filter expression.

        Example:
            from transformplan.filters import Col

            .rows_filter(Col("age") > 18)
            .rows_filter((Col("status") == "active") & (Col("score") >= 50))
        """
        if isinstance(filter, dict):
            filter_dict = filter
        else:
            filter_dict = filter.to_dict()
        return self._register(self._rows_filter, {"filter": filter_dict})

    def _rows_filter(self, data: pl.DataFrame, filter: dict) -> pl.DataFrame:
        expr = Filter.from_dict(filter).to_expr()
        return data.filter(expr)
