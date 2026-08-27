"""Polars backend for TransformPlan.

This module implements all 89 operations using the Polars DataFrame library.
It is the default backend and the reference implementation.

Classes:
    PolarsBackend: Backend implementation using Polars.
"""

from __future__ import annotations

import datetime
import hashlib
import math
import secrets
import string
from typing import Any, Literal, Sequence, cast

import polars as pl

from transformplan.backends.base import (
    AggFunction,
    Backend,
    ClosedInterval,
    FeatureRange,
    FillNullStrategy,
    Numeric,
    RankMethod,
)
from transformplan.dtypes import to_polars_dtype
from transformplan.filters import Filter
from transformplan.protocol import frame_hash

_NUMERIC_TYPES = {
    pl.Int8(),
    pl.Int16(),
    pl.Int32(),
    pl.Int64(),
    pl.UInt8(),
    pl.UInt16(),
    pl.UInt32(),
    pl.UInt64(),
    pl.Float32(),
    pl.Float64(),
}

_STRING_TYPES = {pl.Utf8(), pl.String()}

_DATETIME_TYPES = {pl.Date(), pl.Datetime(), pl.Time(), pl.Duration()}

_BOOLEAN_TYPES = {pl.Boolean()}


class PolarsBackend(Backend):
    """Backend implementation using Polars for all operations."""

    name = "polars"

    # =========================================================================
    # Meta methods (4)
    # =========================================================================

    def compute_hash(self, data: pl.DataFrame) -> str:
        return frame_hash(data)

    def get_shape(self, data: pl.DataFrame) -> tuple[int, int]:
        return data.shape

    def get_columns(self, data: pl.DataFrame) -> list[str]:
        return data.columns

    def get_schema(self, data: pl.DataFrame) -> dict[str, Any]:
        return dict(data.schema)

    # =========================================================================
    # Type system methods (13)
    # =========================================================================

    def is_numeric_type(self, dtype: Any) -> bool:
        return dtype in _NUMERIC_TYPES or dtype.base_type()() in _NUMERIC_TYPES

    def is_string_type(self, dtype: Any) -> bool:
        return dtype in _STRING_TYPES or dtype.base_type()() in _STRING_TYPES

    def is_datetime_type(self, dtype: Any) -> bool:
        return dtype in _DATETIME_TYPES or dtype.base_type()() in _DATETIME_TYPES

    def is_boolean_type(self, dtype: Any) -> bool:
        return dtype in _BOOLEAN_TYPES or dtype.base_type()() in _BOOLEAN_TYPES

    def is_list_type(self, dtype: Any) -> bool:
        return isinstance(dtype, pl.List)

    def float_type(self) -> pl.DataType:
        return pl.Float64()

    def string_type(self) -> pl.DataType:
        return pl.Utf8()

    def integer_type(self) -> pl.DataType:
        return pl.Int64()

    def unsigned_int_type(self) -> pl.DataType:
        return pl.UInt32()

    def boolean_type(self) -> pl.DataType:
        return pl.Boolean()

    def date_type(self) -> pl.DataType:
        return pl.Date()

    def duration_type(self) -> pl.DataType:
        return pl.Duration()

    def type_name(self, dtype: Any) -> str:
        return str(dtype).split("(")[0]

    # =========================================================================
    # Column operations (13)
    # =========================================================================

    def col_drop(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.drop(column)

    def col_rename(
        self, data: pl.DataFrame, column: str, new_name: str
    ) -> pl.DataFrame:
        return data.rename({column: new_name})

    def resolve_dtype(self, dtype: Any) -> Any:
        return to_polars_dtype(dtype)

    def col_cast(
        self, data: pl.DataFrame, column: str, dtype: str | type
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).cast(self.resolve_dtype(dtype)))

    def col_reorder(self, data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        return data.select(columns)

    def col_select(self, data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        return data.select(columns)

    def col_duplicate(
        self, data: pl.DataFrame, column: str, new_name: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).alias(new_name))

    def col_fill_null(
        self,
        data: pl.DataFrame,
        column: str,
        value: Any,
        strategy: FillNullStrategy | None,
    ) -> pl.DataFrame:
        if strategy is not None:
            return data.with_columns(pl.col(column).fill_null(strategy=strategy))
        return data.with_columns(pl.col(column).fill_null(value))

    def col_drop_null(
        self, data: pl.DataFrame, columns: list[str] | None
    ) -> pl.DataFrame:
        return data.drop_nulls(subset=columns)

    def col_drop_zero(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.filter(pl.col(column) != 0)

    def col_add(
        self,
        data: pl.DataFrame,
        new_column: str,
        expr: str | None,
        value: Any,
    ) -> pl.DataFrame:
        if expr is not None:
            return data.with_columns(pl.col(expr).alias(new_column))
        return data.with_columns(pl.lit(value).alias(new_column))

    def col_add_uuid(
        self, data: pl.DataFrame, column: str, length: int
    ) -> pl.DataFrame:
        chars = string.ascii_letters + string.digits
        ids = [
            "".join(secrets.choice(chars) for _ in range(length))
            for _ in range(len(data))
        ]
        return data.with_columns(pl.Series(name=column, values=ids))

    def col_hash(
        self,
        data: pl.DataFrame,
        columns: list[str],
        new_column: str,
        salt: str,
    ) -> pl.DataFrame:
        def hash_row(values: tuple[Any, ...]) -> str:
            content = "|".join(str(v) for v in values) + salt
            return hashlib.sha256(content.encode()).hexdigest()[:16]

        combined = data.select(columns).to_numpy()
        hashes = [hash_row(tuple(row)) for row in combined]
        return data.with_columns(pl.Series(name=new_column, values=hashes))

    def col_coalesce(
        self, data: pl.DataFrame, columns: list[str], new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.coalesce([pl.col(c) for c in columns]).alias(new_column)
        )

    def col_expr(
        self,
        data: pl.DataFrame,
        new_column: str,
        expr: str,
        dtype: str | None,
    ) -> pl.DataFrame:
        return data.with_columns(pl.sql_expr(expr).alias(new_column))

    # =========================================================================
    # Math operations (27)
    # =========================================================================

    def math_add(self, data: pl.DataFrame, column: str, value: Numeric) -> pl.DataFrame:
        return data.with_columns(pl.col(column) + value)

    def math_subtract(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column) - value)

    def math_multiply(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column) * value)

    def math_divide(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column) / value)

    def math_clamp(
        self,
        data: pl.DataFrame,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).clip(lower, upper))

    def math_abs(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).abs())

    def math_round(
        self, data: pl.DataFrame, column: str, decimals: int
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).round(decimals))

    def math_set_min(
        self, data: pl.DataFrame, column: str, min_value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.when(pl.col(column) < min_value)
            .then(min_value)
            .otherwise(pl.col(column))
            .alias(column)
        )

    def math_set_max(
        self, data: pl.DataFrame, column: str, max_value: Numeric
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.when(pl.col(column) > max_value)
            .then(max_value)
            .otherwise(pl.col(column))
            .alias(column)
        )

    def math_add_columns(
        self,
        data: pl.DataFrame,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column_a) + pl.col(column_b)).alias(new_column)
        )

    def math_subtract_columns(
        self,
        data: pl.DataFrame,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column_a) - pl.col(column_b)).alias(new_column)
        )

    def math_multiply_columns(
        self,
        data: pl.DataFrame,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column_a) * pl.col(column_b)).alias(new_column)
        )

    def math_divide_columns(
        self,
        data: pl.DataFrame,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column_a) / pl.col(column_b)).alias(new_column)
        )

    def math_percent_of(
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

    def math_diff_from_agg(
        self,
        data: pl.DataFrame,
        column: str,
        agg: str,
        new_column: str,
        group_by: list[str] | None,
    ) -> pl.DataFrame:
        agg_expr = getattr(pl.col(column), agg)()
        if group_by:
            agg_expr = agg_expr.over(group_by)
        return data.with_columns((pl.col(column) - agg_expr).alias(new_column))

    def math_diff_lag(
        self,
        data: pl.DataFrame,
        column: str,
        order_by: list[str],
        new_column: str,
        group_by: list[str] | None,
        lag: int,
    ) -> pl.DataFrame:
        if group_by:
            expr = pl.col(column) - pl.col(column).shift(lag).over(
                partition_by=group_by, order_by=order_by
            )
            return data.with_columns(expr.alias(new_column))
        data = data.sort(order_by)
        return data.with_columns(
            (pl.col(column) - pl.col(column).shift(lag)).alias(new_column)
        )

    def math_standardize(
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
            return data.with_columns(pl.lit(0.0).alias(new_column))
        return data.with_columns(
            ((pl.col(column) - computed_mean) / computed_std).alias(new_column)
        )

    def math_minmax(
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
            return data.with_columns(pl.lit(0.0).alias(new_column))
        return data.with_columns(
            ((pl.col(column) - computed_median) / computed_iqr).alias(new_column)
        )

    def math_log(
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
            expr = expr.log() / math.log(base)
        return data.with_columns(expr.alias(new_column))

    def math_sqrt(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).sqrt().alias(new_column))

    def math_power(
        self,
        data: pl.DataFrame,
        column: str,
        exponent: Numeric,
        new_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).pow(exponent).alias(new_column))

    def math_winsorize(
        self,
        data: pl.DataFrame,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
        lower_value: Numeric | None,
        upper_value: Numeric | None,
        new_column: str,
    ) -> pl.DataFrame:
        lower_bound: float | None = (
            float(lower_value) if lower_value is not None else None
        )
        if lower_bound is None and lower is not None:
            lower_bound = cast("float", data[column].quantile(lower))

        upper_bound: float | None = (
            float(upper_value) if upper_value is not None else None
        )
        if upper_bound is None and upper is not None:
            upper_bound = cast("float", data[column].quantile(upper))

        return data.with_columns(
            pl.col(column).clip(lower_bound, upper_bound).alias(new_column)
        )

    # =========================================================================
    # Row operations (14)
    # =========================================================================

    def rows_filter(self, data: pl.DataFrame, filter: dict[str, Any]) -> pl.DataFrame:
        expr = Filter.from_dict(filter).to_expr()
        return data.filter(expr)

    def rows_drop(self, data: pl.DataFrame, filter: dict[str, Any]) -> pl.DataFrame:
        expr = Filter.from_dict(filter).to_expr()
        return data.filter(~expr)

    def rows_drop_nulls(
        self, data: pl.DataFrame, columns: list[str] | None
    ) -> pl.DataFrame:
        return data.drop_nulls(subset=columns)

    def rows_flag(
        self,
        data: pl.DataFrame,
        filter: dict[str, Any],
        new_column: str,
        true_value: Any,
        false_value: Any,
    ) -> pl.DataFrame:
        expr = Filter.from_dict(filter).to_expr()
        return data.with_columns(
            pl.when(expr)
            .then(pl.lit(true_value))
            .otherwise(pl.lit(false_value))
            .alias(new_column)
        )

    def rows_unique(
        self,
        data: pl.DataFrame,
        columns: list[str] | None,
        keep: Literal["first", "last", "any", "none"],
    ) -> pl.DataFrame:
        return data.unique(subset=columns, keep=keep)

    def rows_deduplicate(
        self,
        data: pl.DataFrame,
        columns: list[str],
        sort_by: str,
        keep: Literal["first", "last"],
        descending: bool,  # noqa: FBT001
    ) -> pl.DataFrame:
        sorted_data = data.sort(sort_by, descending=descending)
        return sorted_data.unique(subset=columns, keep=keep, maintain_order=True)

    def rows_sort(
        self,
        data: pl.DataFrame,
        by: list[str],
        descending: bool | Sequence[bool],  # noqa: FBT001
    ) -> pl.DataFrame:
        return data.sort(by, descending=descending)

    def rows_head(self, data: pl.DataFrame, n: int) -> pl.DataFrame:
        return data.head(n)

    def rows_tail(self, data: pl.DataFrame, n: int) -> pl.DataFrame:
        return data.tail(n)

    def rows_sample(
        self,
        data: pl.DataFrame,
        n: int | None,
        fraction: float | None,
        seed: int | None,
    ) -> pl.DataFrame:
        return data.sample(n=n, fraction=fraction, seed=seed)

    def rows_explode(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.explode(column)

    def rows_melt(
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

    def rows_pivot(
        self,
        data: pl.DataFrame,
        index: list[str],
        columns: str,
        values: str,
        aggregate_function: AggFunction,
    ) -> pl.DataFrame:
        return data.pivot(
            index=index,
            on=columns,
            values=values,
            aggregate_function=aggregate_function,  # pyright: ignore[reportArgumentType]
        )

    # =========================================================================
    # String operations (10)
    # =========================================================================

    def str_replace(
        self,
        data: pl.DataFrame,
        column: str,
        pattern: str,
        replacement: str,
        literal: bool,  # noqa: FBT001
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.col(column).str.replace_all(pattern, replacement, literal=literal)
        )

    def str_slice(
        self,
        data: pl.DataFrame,
        column: str,
        offset: int,
        length: int | None,
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).str.slice(offset, length))

    def str_truncate(
        self,
        data: pl.DataFrame,
        column: str,
        max_length: int,
        suffix: str,
    ) -> pl.DataFrame:
        cut_length = max_length - len(suffix)
        return data.with_columns(
            pl.when(pl.col(column).str.len_chars() > max_length)
            .then(pl.col(column).str.slice(0, cut_length) + suffix)
            .otherwise(pl.col(column))
            .alias(column)
        )

    def str_split(
        self,
        data: pl.DataFrame,
        column: str,
        separator: str,
        new_columns: list[str] | None,
        keep_original: bool,  # noqa: FBT001
    ) -> pl.DataFrame:
        if new_columns is None:
            result = data.with_columns(pl.col(column).str.split(separator))
            result = result.explode(column)
        else:
            split_col = pl.col(column).str.split(separator)
            for i, new_col in enumerate(new_columns):
                data = data.with_columns(split_col.list.get(i).alias(new_col))
            result = data

        if not keep_original and new_columns is not None:
            result = result.drop(column)

        return result

    def str_lower(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).str.to_lowercase())

    def str_upper(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).str.to_uppercase())

    def str_strip(
        self, data: pl.DataFrame, column: str, chars: str | None
    ) -> pl.DataFrame:
        if chars is None:
            return data.with_columns(pl.col(column).str.strip_chars())
        return data.with_columns(pl.col(column).str.strip_chars(chars))

    def str_pad(
        self,
        data: pl.DataFrame,
        column: str,
        length: int,
        fill_char: str,
        side: str,
    ) -> pl.DataFrame:
        if side == "left":
            return data.with_columns(pl.col(column).str.pad_start(length, fill_char))
        return data.with_columns(pl.col(column).str.pad_end(length, fill_char))

    def str_concat(
        self,
        data: pl.DataFrame,
        columns: list[str],
        new_column: str,
        separator: str,
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.concat_str([pl.col(c) for c in columns], separator=separator).alias(
                new_column
            )
        )

    def str_extract(
        self,
        data: pl.DataFrame,
        column: str,
        pattern: str,
        group_index: int,
        new_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.col(column)
            .str.extract(pattern, group_index=group_index)
            .alias(new_column)
        )

    # =========================================================================
    # Datetime operations (13)
    # =========================================================================

    def dt_year(self, data: pl.DataFrame, column: str, new_column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.year().alias(new_column))

    def dt_month(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.month().alias(new_column))

    def dt_day(self, data: pl.DataFrame, column: str, new_column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.day().alias(new_column))

    def dt_week(self, data: pl.DataFrame, column: str, new_column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.week().alias(new_column))

    def dt_quarter(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.quarter().alias(new_column))

    def dt_year_month(
        self, data: pl.DataFrame, column: str, new_column: str, fmt: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.strftime(fmt).alias(new_column))

    def dt_quarter_year(
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

    def dt_calendar_week(
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
        self, data: pl.DataFrame, column: str, fmt: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.col(column).str.strptime(pl.Date, fmt, strict=False).alias(new_column)
        )

    def dt_format(
        self, data: pl.DataFrame, column: str, fmt: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.strftime(fmt).alias(new_column))

    def dt_diff_days(
        self,
        data: pl.DataFrame,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(
            (pl.col(column_a) - pl.col(column_b)).dt.total_days().alias(new_column)
        )

    def dt_age_years(
        self,
        data: pl.DataFrame,
        birth_column: str,
        reference_column: str | None,
        new_column: str,
    ) -> pl.DataFrame:
        if reference_column is None:
            ref = pl.lit(datetime.date.today())
        else:
            ref = pl.col(reference_column)

        return data.with_columns(
            ((ref - pl.col(birth_column)).dt.total_days() // 365).alias(new_column)
        )

    def dt_truncate(
        self, data: pl.DataFrame, column: str, every: str, new_column: str
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).dt.truncate(every).alias(new_column))

    def dt_is_between(
        self,
        data: pl.DataFrame,
        column: str,
        start: str,
        end: str,
        new_column: str,
        closed: ClosedInterval,
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.col(column)
            .is_between(
                pl.lit(start).str.to_date(), pl.lit(end).str.to_date(), closed=closed
            )
            .alias(new_column)
        )

    # =========================================================================
    # Map operations (10)
    # =========================================================================

    def map_values(
        self,
        data: pl.DataFrame,
        column: str,
        mapping: dict[Any, Any],
        default: Any,
        keep_unmapped: bool,  # noqa: FBT001
    ) -> pl.DataFrame:
        # A when/then chain nests one expression per entry and overflows the
        # stack for large mappings; replace_strict is flat and size-independent.
        # A `None` key never matched under the chain (``col == None`` is null,
        # not true), so it is dropped to keep nulls falling through to the
        # fallback -- matching the DuckDB backend, where ``= NULL`` is never true.
        if not mapping:
            return data

        expr = pl.col(column)
        fallback = expr if keep_unmapped else pl.lit(default)

        pairs = {k: v for k, v in mapping.items() if k is not None}
        if not pairs:
            # Every key was None, so nothing can ever match: the fallback wins.
            return data.with_columns(fallback.alias(column))
        return data.with_columns(
            expr.replace_strict(pairs, default=fallback).alias(column)
        )

    def map_case(
        self,
        data: pl.DataFrame,
        column: str,
        cases: list[tuple[Any, Any]],
        default: Any,
        new_column: str,
    ) -> pl.DataFrame:
        # A when/then chain nests one expression per case and overflows the
        # stack for long case lists; replace_strict is flat and size-independent.
        # `None` conditions never matched under the chain (``col == None`` is
        # null, not true), and duplicate conditions resolved first-match-wins,
        # so both are normalised away before handing the cases to polars.
        mapping: dict[Any, Any] = {}
        for cond_val, result_val in cases:
            if cond_val is not None and cond_val not in mapping:
                mapping[cond_val] = result_val

        if not mapping:
            return data.with_columns(pl.lit(default).alias(new_column))

        return data.with_columns(
            pl.col(column)
            .replace_strict(mapping, default=pl.lit(default))
            .alias(new_column)
        )

    def map_from_column(
        self,
        data: pl.DataFrame,
        column: str,
        lookup_column: str,
        value_column: str,
        new_column: str,
        default: Any,
    ) -> pl.DataFrame:
        lookup = dict(
            zip(
                data[lookup_column].to_list(),
                data[value_column].to_list(),
                strict=False,
            )
        )
        return data.with_columns(
            pl.col(column).replace(lookup, default=default).alias(new_column)
        )

    def map_discretize(
        self,
        data: pl.DataFrame,
        column: str,
        bins: list[float],
        labels: list[str] | None,
        new_column: str,
        right: bool,  # noqa: FBT001
    ) -> pl.DataFrame:
        if labels is None:
            labels = []
            edges = [-float("inf"), *bins, float("inf")]
            for i in range(len(edges) - 1):
                if right:
                    labels.append(f"({edges[i]}, {edges[i + 1]}]")
                else:
                    labels.append(f"[{edges[i]}, {edges[i + 1]})")

        col = pl.col(column)
        edges = [-float("inf"), *bins, float("inf")]

        if right:
            cond = (col > edges[0]) & (col <= edges[1])
        else:
            cond = (col >= edges[0]) & (col < edges[1])
        chain = pl.when(cond).then(pl.lit(labels[0]))

        for i in range(1, len(edges) - 1):
            if right:
                cond = (col > edges[i]) & (col <= edges[i + 1])
            else:
                cond = (col >= edges[i]) & (col < edges[i + 1])
            chain = chain.when(cond).then(pl.lit(labels[i]))

        chain = chain.otherwise(pl.lit(None))
        return data.with_columns(chain.alias(new_column))

    def map_onehot(
        self,
        data: pl.DataFrame,
        column: str,
        categories: list[Any] | None,
        prefix: str,
        drop: Any | None,
        drop_original: bool,  # noqa: FBT001
        unknown_value: str,
    ) -> pl.DataFrame:
        if categories is None:
            categories = data[column].drop_nulls().unique().sort().to_list()

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

        new_columns = []
        for cat in categories:
            if drop_category is not None and cat == drop_category:
                continue

            col_name = f"{prefix}_{cat}"
            if unknown_value == "all_zero":
                expr = (
                    pl.when(pl.col(column) == cat)
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias(col_name)
                )
            else:
                expr = (
                    pl.when(pl.col(column) == cat)
                    .then(pl.lit(1))
                    .when(pl.col(column).is_in(categories))
                    .then(pl.lit(0))
                    .otherwise(pl.lit(None))
                    .alias(col_name)
                )
            new_columns.append(expr)

        result = data.with_columns(new_columns)

        if drop_original:
            result = result.drop(column)

        return result

    def map_ordinal(
        self,
        data: pl.DataFrame,
        column: str,
        categories: list[Any] | None,
        new_column: str,
        drop_original: bool,  # noqa: FBT001
        unknown_value: int,
    ) -> pl.DataFrame:
        if categories is None:
            categories = data[column].drop_nulls().unique().sort().to_list()

        if not categories:
            return data.with_columns(pl.lit(unknown_value).alias(new_column))

        first_cat = categories[0]
        chain = pl.when(pl.col(column) == first_cat).then(pl.lit(0))

        for idx, cat in enumerate(categories[1:], start=1):
            chain = chain.when(pl.col(column) == cat).then(pl.lit(idx))

        chain = chain.otherwise(pl.lit(unknown_value))
        result = data.with_columns(chain.alias(new_column))

        if drop_original and new_column != column:
            result = result.drop(column)

        return result

    def map_label(
        self,
        data: pl.DataFrame,
        column: str,
        categories: list[Any] | None,
        new_column: str,
        drop_original: bool,  # noqa: FBT001
        unknown_value: int,
    ) -> pl.DataFrame:
        return self.map_ordinal(
            data, column, categories, new_column, drop_original, unknown_value
        )

    def map_bool_to_int(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).cast(pl.Int64))

    def map_null_to_value(
        self, data: pl.DataFrame, column: str, value: Any
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).fill_null(value))

    def map_value_to_null(
        self, data: pl.DataFrame, column: str, value: Any
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.when(pl.col(column) == value)
            .then(pl.lit(None))
            .otherwise(pl.col(column))
            .alias(column)
        )

    # =========================================================================
    # Join operations (1)
    # =========================================================================

    def join(
        self,
        data: pl.DataFrame,
        right_data: pl.DataFrame,
        on: list[str],
        how: str,
        suffix: str,
        left_on: list[str] | None = None,
        right_on: list[str] | None = None,
        select_columns: list[str] | None = None,
    ) -> pl.DataFrame:
        right = right_data
        effective_left_on = left_on or on
        effective_right_on = right_on or on

        if select_columns is not None:
            keep = list(dict.fromkeys(effective_right_on + list(select_columns)))
            right = right.select(keep)

        join_kwargs: dict[str, Any] = {"how": how, "suffix": suffix}
        if effective_left_on != effective_right_on:
            join_kwargs["left_on"] = effective_left_on
            join_kwargs["right_on"] = effective_right_on
        else:
            join_kwargs["on"] = on

        return data.join(right, **join_kwargs)
