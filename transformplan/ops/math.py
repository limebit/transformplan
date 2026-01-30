"""Math operations mixin.

This module provides the MathOps mixin class with arithmetic and numeric
operations on DataFrame columns.

Classes:
    MathOps: Mixin providing mathematical operations.

Scalar Operations:
    math_add: Add scalar to column.
    math_subtract: Subtract scalar from column.
    math_multiply: Multiply column by scalar.
    math_divide: Divide column by scalar.
    math_abs: Absolute value.
    math_round: Round to decimal places.
    math_clamp: Clamp values to range.
    math_set_min: Set minimum value.
    math_set_max: Set maximum value.

Column Operations:
    math_add_columns: Add two columns.
    math_subtract_columns: Subtract columns.
    math_multiply_columns: Multiply columns.
    math_divide_columns: Divide columns.
    math_percent_of: Calculate percentage.

Aggregate Operations:
    math_cumsum: Cumulative sum.
    math_rank: Rank values.

Scaling Operations:
    math_standardize: Z-score standardization (mean=0, std=1).
    math_minmax: Min-max normalization to a range.
    math_robust_scale: Robust scaling using median and IQR.

Transform Operations:
    math_log: Logarithmic transform.
    math_sqrt: Square root transform.
    math_power: Power transform.

Outlier Handling:
    math_winsorize: Clip values to percentiles or explicit bounds.

Example:
    >>> plan = TransformPlan().math_multiply("price", 1.1).math_round("price", 2)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Union, cast

import polars as pl

if TYPE_CHECKING:
    from typing import Any, Callable

    from typing_extensions import Self

Numeric = Union[int, float]
RankMethod = Literal["average", "min", "max", "dense", "ordinal", "random"]
FeatureRange = tuple[Numeric, Numeric]


class MathOps:
    """Mixin providing mathematical operations on columns."""

    if TYPE_CHECKING:

        def _register(
            self,
            method: Callable[..., pl.DataFrame],
            params: dict[str, Any],
        ) -> Self: ...

    def math_add(self, column: str, value: Numeric) -> Self:
        """Add a scalar value to a column.

        Returns:
            Self for method chaining.
        """
        return self._register(self._math_add, {"column": column, "value": value})

    def _math_add(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column) + value)

    def math_subtract(self, column: str, value: Numeric) -> Self:
        """Subtract a scalar value from a column.

        Returns:
            Self for method chaining.
        """
        return self._register(self._math_subtract, {"column": column, "value": value})

    def _math_subtract(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column) - value)

    def math_multiply(self, column: str, value: Numeric) -> Self:
        """Multiply a column by a scalar value.

        Returns:
            Self for method chaining.
        """
        return self._register(self._math_multiply, {"column": column, "value": value})

    def _math_multiply(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column) * value)

    def math_divide(self, column: str, value: Numeric) -> Self:
        """Divide a column by a scalar value.

        Returns:
            Self for method chaining.
        """
        return self._register(self._math_divide, {"column": column, "value": value})

    def _math_divide(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column) / value)

    def math_clamp(
        self,
        column: str,
        lower: Numeric | None = None,
        upper: Numeric | None = None,
    ) -> Self:
        """Clamp column values to a range.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_clamp, {"column": column, "lower": lower, "upper": upper}
        )

    def _math_clamp(
        self,
        data: pl.DataFrame,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).clip(lower, upper))

    def math_add_columns(self, column_a: str, column_b: str, new_column: str) -> Self:
        """Add two columns together into a new column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_add_columns,
            {"column_a": column_a, "column_b": column_b, "new_column": new_column},
        )

    def _math_add_columns(
        self, data: pl.DataFrame, column_a: str, column_b: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column_a) + pl.col(column_b)).alias(new_column)
        )

    def math_subtract_columns(
        self, column_a: str, column_b: str, new_column: str
    ) -> Self:
        """Subtract column_b from column_a into a new column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_subtract_columns,
            {"column_a": column_a, "column_b": column_b, "new_column": new_column},
        )

    def _math_subtract_columns(
        self, data: pl.DataFrame, column_a: str, column_b: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column_a) - pl.col(column_b)).alias(new_column)
        )

    def math_multiply_columns(
        self, column_a: str, column_b: str, new_column: str
    ) -> Self:
        """Multiply two columns together into a new column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_multiply_columns,
            {"column_a": column_a, "column_b": column_b, "new_column": new_column},
        )

    def _math_multiply_columns(
        self, data: pl.DataFrame, column_a: str, column_b: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column_a) * pl.col(column_b)).alias(new_column)
        )

    def math_divide_columns(
        self, column_a: str, column_b: str, new_column: str
    ) -> Self:
        """Divide column_a by column_b into a new column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_divide_columns,
            {"column_a": column_a, "column_b": column_b, "new_column": new_column},
        )

    def _math_divide_columns(
        self, data: pl.DataFrame, column_a: str, column_b: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column_a) / pl.col(column_b)).alias(new_column)
        )

    def math_set_min(self, column: str, min_value: Numeric) -> Self:
        """Set a minimum value for a column (values below are raised to min).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_set_min, {"column": column, "min_value": min_value}
        )

    def _math_set_min(
        self, data: pl.DataFrame, column: str, min_value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.when(pl.col(column) < min_value)
            .then(min_value)
            .otherwise(pl.col(column))
            .alias(column)
        )

    def math_set_max(self, column: str, max_value: Numeric) -> Self:
        """Set a maximum value for a column (values above are lowered to max).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_set_max, {"column": column, "max_value": max_value}
        )

    def _math_set_max(
        self, data: pl.DataFrame, column: str, max_value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.when(pl.col(column) > max_value)
            .then(max_value)
            .otherwise(pl.col(column))
            .alias(column)
        )

    def math_abs(self, column: str) -> Self:
        """Take absolute value of a column.

        Returns:
            Self for method chaining.
        """
        return self._register(self._math_abs, {"column": column})

    def _math_abs(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).abs())

    def math_round(self, column: str, decimals: int = 0) -> Self:
        """Round a column to specified decimal places.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_round, {"column": column, "decimals": decimals}
        )

    def _math_round(
        self, data: pl.DataFrame, column: str, decimals: int
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).round(decimals))

    def math_percent_of(
        self,
        column: str,
        total_column: str,
        new_column: str,
        multiply_by: float = 100.0,
    ) -> Self:
        """Calculate percentage of one column relative to another.

        Args:
            column: Numerator column.
            total_column: Denominator column.
            new_column: Name for result column.
            multiply_by: Multiplier (default 100 for percentage).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_percent_of,
            {
                "column": column,
                "total_column": total_column,
                "new_column": new_column,
                "multiply_by": multiply_by,
            },
        )

    def _math_percent_of(
        self,
        data: pl.DataFrame,
        column: str,
        total_column: str,
        new_column: str,
        multiply_by: float,
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column) / pl.col(total_column) * multiply_by).alias(new_column)
        )

    def math_cumsum(
        self,
        column: str,
        new_column: str | None = None,
        group_by: str | list[str] | None = None,
    ) -> Self:
        """Calculate cumulative sum.

        Args:
            column: Column to sum.
            new_column: Name for result column (None = modify in place).
            group_by: Optional column(s) to group by.

        Returns:
            Self for method chaining.
        """
        if isinstance(group_by, str):
            group_by = [group_by]
        return self._register(
            self._math_cumsum,
            {
                "column": column,
                "new_column": new_column or column,
                "group_by": group_by,
            },
        )

    def _math_cumsum(
        self,
        data: pl.DataFrame,
        column: str,
        new_column: str,
        group_by: list[str] | None,
    ) -> pl.DataFrame:
        if group_by:
            return data.with_columns(
                pl.col(column).cum_sum().over(group_by).alias(new_column)
            )
        return data.with_columns(pl.col(column).cum_sum().alias(new_column))

    def math_rank(
        self,
        column: str,
        new_column: str,
        method: RankMethod = "ordinal",
        *,
        descending: bool = False,
        group_by: str | list[str] | None = None,
    ) -> Self:
        """Calculate rank of values.

        Args:
            column: Column to rank.
            new_column: Name for result column.
            method: Ranking method ('ordinal', 'dense', 'min', 'max', 'average').
            descending: Rank in descending order.
            group_by: Optional column(s) to group by.

        Returns:
            Self for method chaining.
        """
        if isinstance(group_by, str):
            group_by = [group_by]
        return self._register(
            self._math_rank,
            {
                "column": column,
                "new_column": new_column,
                "method": method,
                "descending": descending,
                "group_by": group_by,
            },
        )

    def _math_rank(
        self,
        data: pl.DataFrame,
        column: str,
        new_column: str,
        method: RankMethod,
        descending: bool,  # noqa: FBT001
        group_by: list[str] | None,
    ) -> pl.DataFrame:
        expr = pl.col(column).rank(method=method, descending=descending)
        if group_by:
            expr = expr.over(group_by)
        return data.with_columns(expr.alias(new_column))

    # =========================================================================
    # Scaling Operations
    # =========================================================================

    def math_standardize(
        self,
        column: str,
        *,
        mean: Numeric | None = None,
        std: Numeric | None = None,
        new_column: str | None = None,
    ) -> Self:
        """Standardize a column to have mean=0 and std=1 (z-score).

        Args:
            column: Column to transform.
            mean: Mean value. If None, derived from data.
            std: Standard deviation. If None, derived from data.
            new_column: Output column name (default: replace original).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_standardize,
            {
                "column": column,
                "mean": mean,
                "std": std,
                "new_column": new_column or column,
            },
        )

    def _math_standardize(
        self,
        data: pl.DataFrame,
        column: str,
        mean: Numeric | None,
        std: Numeric | None,
        new_column: str,
    ) -> pl.DataFrame:
        computed_mean = (
            float(mean)
            if mean is not None
            else cast("float", data[column].mean()) or 0.0
        )
        computed_std = (
            float(std) if std is not None else cast("float", data[column].std()) or 0.0
        )
        if computed_std == 0:
            # Avoid division by zero - return zeros
            return data.with_columns(pl.lit(0.0).alias(new_column))
        return data.with_columns(
            ((pl.col(column) - computed_mean) / computed_std).alias(new_column)
        )

    def math_minmax(
        self,
        column: str,
        *,
        min_val: Numeric | None = None,
        max_val: Numeric | None = None,
        feature_range: FeatureRange = (0, 1),
        new_column: str | None = None,
    ) -> Self:
        """Scale a column to a range using min-max normalization.

        Args:
            column: Column to transform.
            min_val: Minimum value. If None, derived from data.
            max_val: Maximum value. If None, derived from data.
            feature_range: Output range tuple (default: (0, 1)).
            new_column: Output column name (default: replace original).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_minmax,
            {
                "column": column,
                "min_val": min_val,
                "max_val": max_val,
                "feature_range": feature_range,
                "new_column": new_column or column,
            },
        )

    def _math_minmax(
        self,
        data: pl.DataFrame,
        column: str,
        min_val: Numeric | None,
        max_val: Numeric | None,
        feature_range: FeatureRange,
        new_column: str,
    ) -> pl.DataFrame:
        computed_min = (
            float(min_val)
            if min_val is not None
            else cast("float", data[column].min()) or 0.0
        )
        computed_max = (
            float(max_val)
            if max_val is not None
            else cast("float", data[column].max()) or 0.0
        )
        a, b = feature_range
        if computed_max == computed_min:
            # All values are the same - return midpoint of range
            return data.with_columns(pl.lit((a + b) / 2).alias(new_column))
        return data.with_columns(
            (
                a
                + (pl.col(column) - computed_min)
                * (b - a)
                / (computed_max - computed_min)
            ).alias(new_column)
        )

    def math_robust_scale(
        self,
        column: str,
        *,
        median: Numeric | None = None,
        iqr: Numeric | None = None,
        new_column: str | None = None,
    ) -> Self:
        """Scale a column using median and interquartile range.

        Robust to outliers compared to standardization.

        Args:
            column: Column to transform.
            median: Median value. If None, derived from data.
            iqr: Interquartile range (Q3 - Q1). If None, derived from data.
            new_column: Output column name (default: replace original).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_robust_scale,
            {
                "column": column,
                "median": median,
                "iqr": iqr,
                "new_column": new_column or column,
            },
        )

    def _math_robust_scale(
        self,
        data: pl.DataFrame,
        column: str,
        median: Numeric | None,
        iqr: Numeric | None,
        new_column: str,
    ) -> pl.DataFrame:
        computed_median = (
            float(median)
            if median is not None
            else cast("float", data[column].median()) or 0.0
        )
        if iqr is None:
            q1 = cast("float", data[column].quantile(0.25)) or 0.0
            q3 = cast("float", data[column].quantile(0.75)) or 0.0
            computed_iqr = q3 - q1
        else:
            computed_iqr = float(iqr)
        if computed_iqr == 0:
            # Avoid division by zero - return zeros
            return data.with_columns(pl.lit(0.0).alias(new_column))
        return data.with_columns(
            ((pl.col(column) - computed_median) / computed_iqr).alias(new_column)
        )

    # =========================================================================
    # Transform Operations
    # =========================================================================

    def math_log(
        self,
        column: str,
        *,
        base: Numeric | None = None,
        offset: Numeric = 0,
        new_column: str | None = None,
    ) -> Self:
        """Apply logarithmic transform to a column.

        Args:
            column: Column to transform.
            base: Log base (default: natural log e).
            offset: Value added before log to handle zeros (default: 0).
            new_column: Output column name (default: replace original).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_log,
            {
                "column": column,
                "base": base,
                "offset": offset,
                "new_column": new_column or column,
            },
        )

    def _math_log(
        self,
        data: pl.DataFrame,
        column: str,
        base: Numeric | None,
        offset: Numeric,
        new_column: str,
    ) -> pl.DataFrame:
        expr = pl.col(column) + offset
        if base is None:
            expr = expr.log()
        elif base == 10:
            expr = expr.log10()
        else:
            # log_b(x) = ln(x) / ln(b)
            expr = expr.log() / math.log(base)
        return data.with_columns(expr.alias(new_column))

    def math_sqrt(
        self,
        column: str,
        *,
        new_column: str | None = None,
    ) -> Self:
        """Apply square root transform to a column.

        Args:
            column: Column to transform.
            new_column: Output column name (default: replace original).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_sqrt,
            {
                "column": column,
                "new_column": new_column or column,
            },
        )

    def _math_sqrt(
        self,
        data: pl.DataFrame,
        column: str,
        new_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).sqrt().alias(new_column))

    def math_power(
        self,
        column: str,
        exponent: Numeric,
        *,
        new_column: str | None = None,
    ) -> Self:
        """Apply power transform to a column.

        Args:
            column: Column to transform.
            exponent: Power to raise values to.
            new_column: Output column name (default: replace original).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_power,
            {
                "column": column,
                "exponent": exponent,
                "new_column": new_column or column,
            },
        )

    def _math_power(
        self,
        data: pl.DataFrame,
        column: str,
        exponent: Numeric,
        new_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).pow(exponent).alias(new_column))

    # =========================================================================
    # Outlier Handling
    # =========================================================================

    def math_winsorize(
        self,
        column: str,
        *,
        lower: Numeric | None = None,
        upper: Numeric | None = None,
        lower_value: Numeric | None = None,
        upper_value: Numeric | None = None,
        new_column: str | None = None,
    ) -> Self:
        """Clip values to percentiles or explicit bounds.

        Use either percentile-based (lower/upper as 0-1 fractions) or
        value-based (lower_value/upper_value as explicit bounds) clipping.

        Args:
            column: Column to transform.
            lower: Lower percentile (0-1). E.g., 0.05 for 5th percentile.
            upper: Upper percentile (0-1). E.g., 0.95 for 95th percentile.
            lower_value: Explicit lower bound (overrides lower percentile).
            upper_value: Explicit upper bound (overrides upper percentile).
            new_column: Output column name (default: replace original).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._math_winsorize,
            {
                "column": column,
                "lower": lower,
                "upper": upper,
                "lower_value": lower_value,
                "upper_value": upper_value,
                "new_column": new_column or column,
            },
        )

    def _math_winsorize(
        self,
        data: pl.DataFrame,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
        lower_value: Numeric | None,
        upper_value: Numeric | None,
        new_column: str,
    ) -> pl.DataFrame:
        # Determine lower bound
        lower_bound: float | None = (
            float(lower_value) if lower_value is not None else None
        )
        if lower_bound is None and lower is not None:
            lower_bound = cast("float", data[column].quantile(lower))

        # Determine upper bound
        upper_bound: float | None = (
            float(upper_value) if upper_value is not None else None
        )
        if upper_bound is None and upper is not None:
            upper_bound = cast("float", data[column].quantile(upper))

        return data.with_columns(
            pl.col(column).clip(lower_bound, upper_bound).alias(new_column)
        )
