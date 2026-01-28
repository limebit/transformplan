"""Math operations mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import polars as pl

if TYPE_CHECKING:
    from typing import Self

Numeric = Union[int, float]


class MathOps:
    """Mixin providing mathematical operations on columns."""

    def math_add(self, column: str, value: Numeric) -> Self:
        """Add a scalar value to a column."""
        return self._register(self._math_add, {"column": column, "value": value})

    def _math_add(self, data: pl.DataFrame, column: str, value: Numeric) -> pl.DataFrame:
        return data.with_columns(pl.col(column) + value)

    def math_subtract(self, column: str, value: Numeric) -> Self:
        """Subtract a scalar value from a column."""
        return self._register(self._math_subtract, {"column": column, "value": value})

    def _math_subtract(self, data: pl.DataFrame, column: str, value: Numeric) -> pl.DataFrame:
        return data.with_columns(pl.col(column) - value)

    def math_multiply(self, column: str, value: Numeric) -> Self:
        """Multiply a column by a scalar value."""
        return self._register(self._math_multiply, {"column": column, "value": value})

    def _math_multiply(self, data: pl.DataFrame, column: str, value: Numeric) -> pl.DataFrame:
        return data.with_columns(pl.col(column) * value)

    def math_divide(self, column: str, value: Numeric) -> Self:
        """Divide a column by a scalar value."""
        return self._register(self._math_divide, {"column": column, "value": value})

    def _math_divide(self, data: pl.DataFrame, column: str, value: Numeric) -> pl.DataFrame:
        return data.with_columns(pl.col(column) / value)

    def math_clamp(
        self,
        column: str,
        lower: Numeric | None = None,
        upper: Numeric | None = None,
    ) -> Self:
        """Clamp column values to a range."""
        return self._register(self._math_clamp, {"column": column, "lower": lower, "upper": upper})

    def _math_clamp(
        self,
        data: pl.DataFrame,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).clip(lower, upper))
