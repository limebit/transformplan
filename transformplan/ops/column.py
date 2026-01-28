"""Column operations mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import polars as pl

if TYPE_CHECKING:
    from typing import Self


class ColumnOps:
    """Mixin providing column-level operations."""

    def col_drop(self, column: str) -> Self:
        """Drop a column from the DataFrame."""
        return self._register(self._col_drop, {"column": column})

    def _col_drop(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.drop(column)

    def col_rename(self, column: str, new_name: str) -> Self:
        """Rename a column."""
        return self._register(self._col_rename, {"column": column, "new_name": new_name})

    def _col_rename(self, data: pl.DataFrame, column: str, new_name: str) -> pl.DataFrame:
        return data.rename({column: new_name})

    def col_cast(self, column: str, dtype: type) -> Self:
        """Cast a column to a different dtype."""
        return self._register(self._col_cast, {"column": column, "dtype": dtype})

    def _col_cast(self, data: pl.DataFrame, column: str, dtype: type) -> pl.DataFrame:
        return data.with_columns(pl.col(column).cast(dtype))

    def col_reorder(self, columns: Sequence[str]) -> Self:
        """Reorder columns. Unlisted columns are dropped."""
        return self._register(self._col_reorder, {"columns": list(columns)})

    def _col_reorder(self, data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        return data.select(columns)

    def col_duplicate(self, column: str, new_name: str) -> Self:
        """Duplicate a column under a new name."""
        return self._register(self._col_duplicate, {"column": column, "new_name": new_name})

    def _col_duplicate(self, data: pl.DataFrame, column: str, new_name: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).alias(new_name))
