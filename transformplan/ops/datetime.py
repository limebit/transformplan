"""Datetime operations mixin.

This module provides the DatetimeOps mixin class with date and time
extraction and manipulation operations.

Classes:
    DatetimeOps: Mixin providing datetime operations.

Extraction Operations:
    dt_year: Extract year.
    dt_month: Extract month.
    dt_day: Extract day.
    dt_week: Extract ISO week number.
    dt_quarter: Extract quarter (1-4).

Formatting Operations:
    dt_year_month: Create year-month string.
    dt_quarter_year: Create quarter-year string.
    dt_calendar_week: Create year-week string.
    dt_format: Format datetime as string.

Parsing:
    dt_parse: Parse string to datetime.

Arithmetic:
    dt_diff_days: Difference in days.
    dt_age_years: Calculate age in years.

Other:
    dt_truncate: Truncate to precision.
    dt_is_between: Check if in date range.

Example:
    >>> plan = TransformPlan().dt_parse("date_str").dt_year("date", "year")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from typing import Any, Callable

    from typing_extensions import Self


class DatetimeOps:
    """Mixin providing datetime operations on columns."""

    if TYPE_CHECKING:

        def _register(
            self,
            method: Callable[..., pl.DataFrame],
            params: dict[str, Any],
        ) -> Self: ...

    def dt_year(self, column: str, new_column: str | None = None) -> Self:
        """Extract year from a datetime column.

        Args:
            column: Source datetime column.
            new_column: Name for result column (None = modify in place).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_year, {"column": column, "new_column": new_column or column}
        )

    def _dt_year(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.year().alias(new_column))

    def dt_month(self, column: str, new_column: str | None = None) -> Self:
        """Extract month from a datetime column.

        Args:
            column: Source datetime column.
            new_column: Name for result column (None = modify in place).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_month, {"column": column, "new_column": new_column or column}
        )

    def _dt_month(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.month().alias(new_column))

    def dt_day(self, column: str, new_column: str | None = None) -> Self:
        """Extract day from a datetime column.

        Args:
            column: Source datetime column.
            new_column: Name for result column (None = modify in place).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_day, {"column": column, "new_column": new_column or column}
        )

    def _dt_day(self, data: pl.DataFrame, column: str, new_column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.day().alias(new_column))

    def dt_week(self, column: str, new_column: str | None = None) -> Self:
        """Extract ISO week number from a datetime column.

        Args:
            column: Source datetime column.
            new_column: Name for result column (None = modify in place).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_week, {"column": column, "new_column": new_column or column}
        )

    def _dt_week(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.week().alias(new_column))

    def dt_quarter(self, column: str, new_column: str | None = None) -> Self:
        """Extract quarter (1-4) from a datetime column.

        Args:
            column: Source datetime column.
            new_column: Name for result column (None = modify in place).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_quarter, {"column": column, "new_column": new_column or column}
        )

    def _dt_quarter(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.quarter().alias(new_column))

    def dt_year_month(self, column: str, new_column: str, fmt: str = "%Y-%m") -> Self:
        """Create a year-month string from a datetime column.

        Args:
            column: Source datetime column.
            new_column: Name for result column.
            fmt: Output format string.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_year_month,
            {"column": column, "new_column": new_column, "fmt": fmt},
        )

    def _dt_year_month(
        self, data: pl.DataFrame, column: str, new_column: str, fmt: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.strftime(fmt).alias(new_column))

    def dt_quarter_year(self, column: str, new_column: str) -> Self:
        """Create a quarter-year string (e.g., 'Q1-2024') from a datetime column.

        Args:
            column: Source datetime column.
            new_column: Name for result column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_quarter_year, {"column": column, "new_column": new_column}
        )

    def _dt_quarter_year(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(
            (
                pl.lit("Q")
                + pl.col(column).dt.quarter().cast(pl.Utf8)
                + pl.lit("-")
                + pl.col(column).dt.year().cast(pl.Utf8)
            ).alias(new_column)
        )

    def dt_calendar_week(self, column: str, new_column: str) -> Self:
        """Create a year-week string (e.g., '2024-W05') from a datetime column.

        Args:
            column: Source datetime column.
            new_column: Name for result column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_calendar_week, {"column": column, "new_column": new_column}
        )

    def _dt_calendar_week(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(
            (
                pl.col(column).dt.iso_year().cast(pl.Utf8)
                + pl.lit("-W")
                + pl.col(column).dt.week().cast(pl.Utf8).str.pad_start(2, "0")
            ).alias(new_column)
        )

    def dt_parse(
        self,
        column: str,
        fmt: str = "%Y-%m-%d",
        new_column: str | None = None,
    ) -> Self:
        """Parse a string column into a datetime.

        Args:
            column: Source string column.
            fmt: Date format string.
            new_column: Name for result column (None = modify in place).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_parse,
            {"column": column, "fmt": fmt, "new_column": new_column or column},
        )

    def _dt_parse(
        self, data: pl.DataFrame, column: str, fmt: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.col(column).str.strptime(pl.Date, fmt, strict=False).alias(new_column)
        )

    def dt_format(self, column: str, fmt: str, new_column: str | None = None) -> Self:
        """Format a datetime column as a string.

        Args:
            column: Source datetime column.
            fmt: Output format string.
            new_column: Name for result column (None = modify in place).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_format,
            {"column": column, "fmt": fmt, "new_column": new_column or column},
        )

    def _dt_format(
        self, data: pl.DataFrame, column: str, fmt: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.strftime(fmt).alias(new_column))

    def dt_diff_days(self, column_a: str, column_b: str, new_column: str) -> Self:
        """Calculate difference in days between two date columns (a - b).

        Args:
            column_a: First date column.
            column_b: Second date column.
            new_column: Name for result column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_diff_days,
            {"column_a": column_a, "column_b": column_b, "new_column": new_column},
        )

    def _dt_diff_days(
        self, data: pl.DataFrame, column_a: str, column_b: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column_a) - pl.col(column_b)).dt.total_days().alias(new_column)
        )

    def dt_age_years(
        self,
        birth_column: str,
        reference_column: str | None = None,
        new_column: str = "age",
    ) -> Self:
        """Calculate age in years from a birth date.

        Args:
            birth_column: Column containing birth dates.
            reference_column: Column containing reference dates (None = today).
            new_column: Name for result column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_age_years,
            {
                "birth_column": birth_column,
                "reference_column": reference_column,
                "new_column": new_column,
            },
        )

    def _dt_age_years(
        self,
        data: pl.DataFrame,
        birth_column: str,
        reference_column: str | None,
        new_column: str,
    ) -> pl.DataFrame:
        if reference_column is None:
            ref = pl.lit(pl.Series([None]).cast(pl.Date).fill_null(strategy="max"))
            # Use current date
            import datetime

            ref = pl.lit(datetime.date.today())
        else:
            ref = pl.col(reference_column)

        return data.with_columns(
            ((ref - pl.col(birth_column)).dt.total_days() // 365).alias(new_column)
        )

    def dt_truncate(
        self,
        column: str,
        every: str,
        new_column: str | None = None,
    ) -> Self:
        """Truncate datetime to a specified precision.

        Args:
            column: Source datetime column.
            every: Truncation interval ('1d', '1mo', '1y', '1h', etc.).
            new_column: Name for result column (None = modify in place).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_truncate,
            {"column": column, "every": every, "new_column": new_column or column},
        )

    def _dt_truncate(
        self, data: pl.DataFrame, column: str, every: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.truncate(every).alias(new_column))

    def dt_is_between(
        self,
        column: str,
        start: str,
        end: str,
        new_column: str,
        closed: str = "both",
    ) -> Self:
        """Check if date falls within a range.

        Args:
            column: Source datetime column.
            start: Start date (string, will be parsed).
            end: End date (string, will be parsed).
            new_column: Name for boolean result column.
            closed: Which endpoints to include ('both', 'left', 'right', 'none').

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._dt_is_between,
            {
                "column": column,
                "start": start,
                "end": end,
                "new_column": new_column,
                "closed": closed,
            },
        )

    def _dt_is_between(
        self,
        data: pl.DataFrame,
        column: str,
        start: str,
        end: str,
        new_column: str,
        closed: str,
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.col(column)
            .is_between(
                pl.lit(start).str.to_date(), pl.lit(end).str.to_date(), closed=closed
            )
            .alias(new_column)
        )
