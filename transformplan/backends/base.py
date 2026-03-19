"""Abstract base class for TransformPlan backends.

This module defines the Backend ABC with all 86 operation methods that each
backend must implement. Using ABC (not typing.Protocol) gives runtime enforcement
that all methods are implemented, catching mistakes at instantiation.

Classes:
    Backend: Abstract base class with 86 abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Sequence, Union

if TYPE_CHECKING:
    import polars as pl
    from polars._typing import PivotAgg

Numeric = Union[int, float]
RankMethod = Literal["average", "min", "max", "dense", "ordinal", "random"]
FeatureRange = tuple[Numeric, Numeric]
FillNullStrategy = Literal["forward", "backward", "min", "max", "mean", "zero"]
ClosedInterval = Literal["left", "right", "both", "none"]


class Backend(ABC):
    """Abstract base class defining the operation interface for backends.

    Each backend must implement all 86 operations. Methods receive a DataFrame
    and operation-specific parameters, and return a transformed DataFrame.
    """

    # =========================================================================
    # Column operations (13)
    # =========================================================================

    @abstractmethod
    def col_drop(self, data: pl.DataFrame, column: str) -> pl.DataFrame: ...

    @abstractmethod
    def col_rename(
        self, data: pl.DataFrame, column: str, new_name: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_cast(
        self, data: pl.DataFrame, column: str, dtype: type
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_reorder(
        self, data: pl.DataFrame, columns: list[str]
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_select(
        self, data: pl.DataFrame, columns: list[str]
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_duplicate(
        self, data: pl.DataFrame, column: str, new_name: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_fill_null(
        self,
        data: pl.DataFrame,
        column: str,
        value: Any,  # noqa: ANN401
        strategy: FillNullStrategy | None,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_drop_null(
        self, data: pl.DataFrame, columns: list[str] | None
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_drop_zero(
        self, data: pl.DataFrame, column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_add(
        self,
        data: pl.DataFrame,
        new_column: str,
        expr: str | None,
        value: Any,  # noqa: ANN401
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_add_uuid(
        self, data: pl.DataFrame, column: str, length: int
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_hash(
        self,
        data: pl.DataFrame,
        columns: list[str],
        new_column: str,
        salt: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def col_coalesce(
        self, data: pl.DataFrame, columns: list[str], new_column: str
    ) -> pl.DataFrame: ...

    # =========================================================================
    # Math operations (26)
    # =========================================================================

    @abstractmethod
    def math_add(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_subtract(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_multiply(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_divide(
        self, data: pl.DataFrame, column: str, value: Numeric
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_clamp(
        self,
        data: pl.DataFrame,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_abs(self, data: pl.DataFrame, column: str) -> pl.DataFrame: ...

    @abstractmethod
    def math_round(
        self, data: pl.DataFrame, column: str, decimals: int
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_set_min(
        self, data: pl.DataFrame, column: str, min_value: Numeric
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_set_max(
        self, data: pl.DataFrame, column: str, max_value: Numeric
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_add_columns(
        self,
        data: pl.DataFrame,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_subtract_columns(
        self,
        data: pl.DataFrame,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_multiply_columns(
        self,
        data: pl.DataFrame,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_divide_columns(
        self,
        data: pl.DataFrame,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_percent_of(
        self,
        data: pl.DataFrame,
        column: str,
        total_column: str,
        new_column: str,
        multiply_by: float,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_cumsum(
        self,
        data: pl.DataFrame,
        column: str,
        new_column: str,
        group_by: list[str] | None,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_rank(
        self,
        data: pl.DataFrame,
        column: str,
        new_column: str,
        method: RankMethod,
        descending: bool,  # noqa: FBT001
        group_by: list[str] | None,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_standardize(
        self,
        data: pl.DataFrame,
        column: str,
        mean: Numeric | None,
        std: Numeric | None,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_minmax(
        self,
        data: pl.DataFrame,
        column: str,
        min_val: Numeric | None,
        max_val: Numeric | None,
        feature_range: FeatureRange,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_robust_scale(
        self,
        data: pl.DataFrame,
        column: str,
        median: Numeric | None,
        iqr: Numeric | None,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_log(
        self,
        data: pl.DataFrame,
        column: str,
        base: Numeric | None,
        offset: Numeric,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_sqrt(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_power(
        self,
        data: pl.DataFrame,
        column: str,
        exponent: Numeric,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def math_winsorize(
        self,
        data: pl.DataFrame,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
        lower_value: Numeric | None,
        upper_value: Numeric | None,
        new_column: str,
    ) -> pl.DataFrame: ...

    # =========================================================================
    # Row operations (14)
    # =========================================================================

    @abstractmethod
    def rows_filter(
        self, data: pl.DataFrame, filter: dict[str, Any]
    ) -> pl.DataFrame: ...

    @abstractmethod
    def rows_drop(
        self, data: pl.DataFrame, filter: dict[str, Any]
    ) -> pl.DataFrame: ...

    @abstractmethod
    def rows_drop_nulls(
        self, data: pl.DataFrame, columns: list[str] | None
    ) -> pl.DataFrame: ...

    @abstractmethod
    def rows_flag(
        self,
        data: pl.DataFrame,
        filter: dict[str, Any],
        new_column: str,
        true_value: Any,  # noqa: ANN401
        false_value: Any,  # noqa: ANN401
    ) -> pl.DataFrame: ...

    @abstractmethod
    def rows_unique(
        self,
        data: pl.DataFrame,
        columns: list[str] | None,
        keep: Literal["first", "last", "any", "none"],
    ) -> pl.DataFrame: ...

    @abstractmethod
    def rows_deduplicate(
        self,
        data: pl.DataFrame,
        columns: list[str],
        sort_by: str,
        keep: Literal["first", "last"],
        descending: bool,  # noqa: FBT001
    ) -> pl.DataFrame: ...

    @abstractmethod
    def rows_sort(
        self,
        data: pl.DataFrame,
        by: list[str],
        descending: bool | Sequence[bool],  # noqa: FBT001
    ) -> pl.DataFrame: ...

    @abstractmethod
    def rows_head(self, data: pl.DataFrame, n: int) -> pl.DataFrame: ...

    @abstractmethod
    def rows_tail(self, data: pl.DataFrame, n: int) -> pl.DataFrame: ...

    @abstractmethod
    def rows_sample(
        self,
        data: pl.DataFrame,
        n: int | None,
        fraction: float | None,
        seed: int | None,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def rows_explode(
        self, data: pl.DataFrame, column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def rows_melt(
        self,
        data: pl.DataFrame,
        id_columns: list[str],
        value_columns: list[str],
        variable_name: str,
        value_name: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def rows_pivot(
        self,
        data: pl.DataFrame,
        index: list[str],
        columns: str,
        values: str,
        aggregate_function: PivotAgg,
    ) -> pl.DataFrame: ...

    # =========================================================================
    # String operations (10)
    # =========================================================================

    @abstractmethod
    def str_replace(
        self,
        data: pl.DataFrame,
        column: str,
        pattern: str,
        replacement: str,
        literal: bool,  # noqa: FBT001
    ) -> pl.DataFrame: ...

    @abstractmethod
    def str_slice(
        self,
        data: pl.DataFrame,
        column: str,
        offset: int,
        length: int | None,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def str_truncate(
        self,
        data: pl.DataFrame,
        column: str,
        max_length: int,
        suffix: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def str_split(
        self,
        data: pl.DataFrame,
        column: str,
        separator: str,
        new_columns: list[str] | None,
        keep_original: bool,  # noqa: FBT001
    ) -> pl.DataFrame: ...

    @abstractmethod
    def str_lower(
        self, data: pl.DataFrame, column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def str_upper(
        self, data: pl.DataFrame, column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def str_strip(
        self, data: pl.DataFrame, column: str, chars: str | None
    ) -> pl.DataFrame: ...

    @abstractmethod
    def str_pad(
        self,
        data: pl.DataFrame,
        column: str,
        length: int,
        fill_char: str,
        side: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def str_concat(
        self,
        data: pl.DataFrame,
        columns: list[str],
        new_column: str,
        separator: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def str_extract(
        self,
        data: pl.DataFrame,
        column: str,
        pattern: str,
        group_index: int,
        new_column: str,
    ) -> pl.DataFrame: ...

    # =========================================================================
    # Datetime operations (13)
    # =========================================================================

    @abstractmethod
    def dt_year(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_month(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_day(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_week(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_quarter(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_year_month(
        self, data: pl.DataFrame, column: str, new_column: str, fmt: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_quarter_year(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_calendar_week(
        self, data: pl.DataFrame, column: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_parse(
        self, data: pl.DataFrame, column: str, fmt: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_format(
        self, data: pl.DataFrame, column: str, fmt: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_diff_days(
        self,
        data: pl.DataFrame,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_age_years(
        self,
        data: pl.DataFrame,
        birth_column: str,
        reference_column: str | None,
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_truncate(
        self, data: pl.DataFrame, column: str, every: str, new_column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def dt_is_between(
        self,
        data: pl.DataFrame,
        column: str,
        start: str,
        end: str,
        new_column: str,
        closed: ClosedInterval,
    ) -> pl.DataFrame: ...

    # =========================================================================
    # Map operations (10)
    # =========================================================================

    @abstractmethod
    def map_values(
        self,
        data: pl.DataFrame,
        column: str,
        mapping: dict[Any, Any],
        default: Any,  # noqa: ANN401
        keep_unmapped: bool,  # noqa: FBT001
    ) -> pl.DataFrame: ...

    @abstractmethod
    def map_case(
        self,
        data: pl.DataFrame,
        column: str,
        cases: list[tuple[Any, Any]],
        default: Any,  # noqa: ANN401
        new_column: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def map_from_column(
        self,
        data: pl.DataFrame,
        column: str,
        lookup_column: str,
        value_column: str,
        new_column: str,
        default: Any,  # noqa: ANN401
    ) -> pl.DataFrame: ...

    @abstractmethod
    def map_discretize(
        self,
        data: pl.DataFrame,
        column: str,
        bins: list[float],
        labels: list[str] | None,
        new_column: str,
        right: bool,  # noqa: FBT001
    ) -> pl.DataFrame: ...

    @abstractmethod
    def map_onehot(
        self,
        data: pl.DataFrame,
        column: str,
        categories: list[Any] | None,
        prefix: str,
        drop: Any | None,  # noqa: ANN401
        drop_original: bool,  # noqa: FBT001
        unknown_value: str,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def map_ordinal(
        self,
        data: pl.DataFrame,
        column: str,
        categories: list[Any] | None,
        new_column: str,
        drop_original: bool,  # noqa: FBT001
        unknown_value: int,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def map_label(
        self,
        data: pl.DataFrame,
        column: str,
        categories: list[Any] | None,
        new_column: str,
        drop_original: bool,  # noqa: FBT001
        unknown_value: int,
    ) -> pl.DataFrame: ...

    @abstractmethod
    def map_bool_to_int(
        self, data: pl.DataFrame, column: str
    ) -> pl.DataFrame: ...

    @abstractmethod
    def map_null_to_value(
        self, data: pl.DataFrame, column: str, value: Any  # noqa: ANN401
    ) -> pl.DataFrame: ...

    @abstractmethod
    def map_value_to_null(
        self, data: pl.DataFrame, column: str, value: Any  # noqa: ANN401
    ) -> pl.DataFrame: ...
