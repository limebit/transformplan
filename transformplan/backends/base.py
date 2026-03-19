"""Abstract base class for TransformPlan backends.

This module defines the Backend ABC with all 86 operation methods that each
backend must implement. Using ABC (not typing.Protocol) gives runtime enforcement
that all methods are implemented, catching mistakes at instantiation.

Classes:
    Backend: Abstract base class with 86 abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Sequence, Union

Numeric = Union[int, float]
RankMethod = Literal["average", "min", "max", "dense", "ordinal", "random"]
FeatureRange = tuple[Numeric, Numeric]
FillNullStrategy = Literal["forward", "backward", "min", "max", "mean", "zero"]
ClosedInterval = Literal["left", "right", "both", "none"]
AggFunction = Literal["first", "sum", "mean", "median", "min", "max", "count"]


class Backend(ABC):
    """Abstract base class defining the operation interface for backends.

    Each backend must implement all 86 operations. Methods receive data
    and operation-specific parameters, and return transformed data.
    """

    # =========================================================================
    # Meta methods (4)
    # =========================================================================

    @abstractmethod
    def compute_hash(self, data: Any) -> str:
        """Deterministic, order-invariant hash of the data."""
        ...

    @abstractmethod
    def get_shape(self, data: Any) -> tuple[int, int]:
        """Return (rows, columns) shape."""
        ...

    @abstractmethod
    def get_columns(self, data: Any) -> list[str]:
        """Return list of column names."""
        ...

    @abstractmethod
    def get_schema(self, data: Any) -> dict[str, Any]:
        """Return column name to type mapping (native types for each backend)."""
        ...

    # =========================================================================
    # Type system methods (12)
    # =========================================================================

    @abstractmethod
    def is_numeric_type(self, dtype: Any) -> bool:
        """Check if dtype is numeric."""
        ...

    @abstractmethod
    def is_string_type(self, dtype: Any) -> bool:
        """Check if dtype is string/text."""
        ...

    @abstractmethod
    def is_datetime_type(self, dtype: Any) -> bool:
        """Check if dtype is date/datetime/time/duration."""
        ...

    @abstractmethod
    def is_boolean_type(self, dtype: Any) -> bool:
        """Check if dtype is boolean."""
        ...

    @abstractmethod
    def is_list_type(self, dtype: Any) -> bool:
        """Check if dtype is a list/array type."""
        ...

    @abstractmethod
    def float_type(self) -> Any:
        """Return the float/double type for this backend."""
        ...

    @abstractmethod
    def string_type(self) -> Any:
        """Return the string/text type for this backend."""
        ...

    @abstractmethod
    def integer_type(self) -> Any:
        """Return the integer type for this backend."""
        ...

    @abstractmethod
    def unsigned_int_type(self) -> Any:
        """Return the unsigned integer type for this backend."""
        ...

    @abstractmethod
    def boolean_type(self) -> Any:
        """Return the boolean type for this backend."""
        ...

    @abstractmethod
    def date_type(self) -> Any:
        """Return the date type for this backend."""
        ...

    @abstractmethod
    def type_name(self, dtype: Any) -> str:
        """Return a human-readable name for a dtype."""
        ...

    # =========================================================================
    # Column operations (13)
    # =========================================================================

    @abstractmethod
    def col_drop(self, data: Any, column: str) -> Any: ...

    @abstractmethod
    def col_rename(
        self, data: Any, column: str, new_name: str
    ) -> Any: ...

    @abstractmethod
    def col_cast(
        self, data: Any, column: str, dtype: type
    ) -> Any: ...

    @abstractmethod
    def col_reorder(
        self, data: Any, columns: list[str]
    ) -> Any: ...

    @abstractmethod
    def col_select(
        self, data: Any, columns: list[str]
    ) -> Any: ...

    @abstractmethod
    def col_duplicate(
        self, data: Any, column: str, new_name: str
    ) -> Any: ...

    @abstractmethod
    def col_fill_null(
        self,
        data: Any,
        column: str,
        value: Any,
        strategy: FillNullStrategy | None,
    ) -> Any: ...

    @abstractmethod
    def col_drop_null(
        self, data: Any, columns: list[str] | None
    ) -> Any: ...

    @abstractmethod
    def col_drop_zero(
        self, data: Any, column: str
    ) -> Any: ...

    @abstractmethod
    def col_add(
        self,
        data: Any,
        new_column: str,
        expr: str | None,
        value: Any,
    ) -> Any: ...

    @abstractmethod
    def col_add_uuid(
        self, data: Any, column: str, length: int
    ) -> Any: ...

    @abstractmethod
    def col_hash(
        self,
        data: Any,
        columns: list[str],
        new_column: str,
        salt: str,
    ) -> Any: ...

    @abstractmethod
    def col_coalesce(
        self, data: Any, columns: list[str], new_column: str
    ) -> Any: ...

    # =========================================================================
    # Math operations (26)
    # =========================================================================

    @abstractmethod
    def math_add(
        self, data: Any, column: str, value: Numeric
    ) -> Any: ...

    @abstractmethod
    def math_subtract(
        self, data: Any, column: str, value: Numeric
    ) -> Any: ...

    @abstractmethod
    def math_multiply(
        self, data: Any, column: str, value: Numeric
    ) -> Any: ...

    @abstractmethod
    def math_divide(
        self, data: Any, column: str, value: Numeric
    ) -> Any: ...

    @abstractmethod
    def math_clamp(
        self,
        data: Any,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
    ) -> Any: ...

    @abstractmethod
    def math_abs(self, data: Any, column: str) -> Any: ...

    @abstractmethod
    def math_round(
        self, data: Any, column: str, decimals: int
    ) -> Any: ...

    @abstractmethod
    def math_set_min(
        self, data: Any, column: str, min_value: Numeric
    ) -> Any: ...

    @abstractmethod
    def math_set_max(
        self, data: Any, column: str, max_value: Numeric
    ) -> Any: ...

    @abstractmethod
    def math_add_columns(
        self,
        data: Any,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def math_subtract_columns(
        self,
        data: Any,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def math_multiply_columns(
        self,
        data: Any,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def math_divide_columns(
        self,
        data: Any,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def math_percent_of(
        self,
        data: Any,
        column: str,
        total_column: str,
        new_column: str,
        multiply_by: float,
    ) -> Any: ...

    @abstractmethod
    def math_cumsum(
        self,
        data: Any,
        column: str,
        new_column: str,
        group_by: list[str] | None,
    ) -> Any: ...

    @abstractmethod
    def math_rank(
        self,
        data: Any,
        column: str,
        new_column: str,
        method: RankMethod,
        descending: bool,  # noqa: FBT001
        group_by: list[str] | None,
    ) -> Any: ...

    @abstractmethod
    def math_standardize(
        self,
        data: Any,
        column: str,
        mean: Numeric | None,
        std: Numeric | None,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def math_minmax(
        self,
        data: Any,
        column: str,
        min_val: Numeric | None,
        max_val: Numeric | None,
        feature_range: FeatureRange,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def math_robust_scale(
        self,
        data: Any,
        column: str,
        median: Numeric | None,
        iqr: Numeric | None,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def math_log(
        self,
        data: Any,
        column: str,
        base: Numeric | None,
        offset: Numeric,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def math_sqrt(
        self, data: Any, column: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def math_power(
        self,
        data: Any,
        column: str,
        exponent: Numeric,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def math_winsorize(
        self,
        data: Any,
        column: str,
        lower: Numeric | None,
        upper: Numeric | None,
        lower_value: Numeric | None,
        upper_value: Numeric | None,
        new_column: str,
    ) -> Any: ...

    # =========================================================================
    # Row operations (14)
    # =========================================================================

    @abstractmethod
    def rows_filter(
        self, data: Any, filter: dict[str, Any]
    ) -> Any: ...

    @abstractmethod
    def rows_drop(
        self, data: Any, filter: dict[str, Any]
    ) -> Any: ...

    @abstractmethod
    def rows_drop_nulls(
        self, data: Any, columns: list[str] | None
    ) -> Any: ...

    @abstractmethod
    def rows_flag(
        self,
        data: Any,
        filter: dict[str, Any],
        new_column: str,
        true_value: Any,
        false_value: Any,
    ) -> Any: ...

    @abstractmethod
    def rows_unique(
        self,
        data: Any,
        columns: list[str] | None,
        keep: Literal["first", "last", "any", "none"],
    ) -> Any: ...

    @abstractmethod
    def rows_deduplicate(
        self,
        data: Any,
        columns: list[str],
        sort_by: str,
        keep: Literal["first", "last"],
        descending: bool,  # noqa: FBT001
    ) -> Any: ...

    @abstractmethod
    def rows_sort(
        self,
        data: Any,
        by: list[str],
        descending: bool | Sequence[bool],  # noqa: FBT001
    ) -> Any: ...

    @abstractmethod
    def rows_head(self, data: Any, n: int) -> Any: ...

    @abstractmethod
    def rows_tail(self, data: Any, n: int) -> Any: ...

    @abstractmethod
    def rows_sample(
        self,
        data: Any,
        n: int | None,
        fraction: float | None,
        seed: int | None,
    ) -> Any: ...

    @abstractmethod
    def rows_explode(
        self, data: Any, column: str
    ) -> Any: ...

    @abstractmethod
    def rows_melt(
        self,
        data: Any,
        id_columns: list[str],
        value_columns: list[str],
        variable_name: str,
        value_name: str,
    ) -> Any: ...

    @abstractmethod
    def rows_pivot(
        self,
        data: Any,
        index: list[str],
        columns: str,
        values: str,
        aggregate_function: AggFunction,
    ) -> Any: ...

    # =========================================================================
    # String operations (10)
    # =========================================================================

    @abstractmethod
    def str_replace(
        self,
        data: Any,
        column: str,
        pattern: str,
        replacement: str,
        literal: bool,  # noqa: FBT001
    ) -> Any: ...

    @abstractmethod
    def str_slice(
        self,
        data: Any,
        column: str,
        offset: int,
        length: int | None,
    ) -> Any: ...

    @abstractmethod
    def str_truncate(
        self,
        data: Any,
        column: str,
        max_length: int,
        suffix: str,
    ) -> Any: ...

    @abstractmethod
    def str_split(
        self,
        data: Any,
        column: str,
        separator: str,
        new_columns: list[str] | None,
        keep_original: bool,  # noqa: FBT001
    ) -> Any: ...

    @abstractmethod
    def str_lower(
        self, data: Any, column: str
    ) -> Any: ...

    @abstractmethod
    def str_upper(
        self, data: Any, column: str
    ) -> Any: ...

    @abstractmethod
    def str_strip(
        self, data: Any, column: str, chars: str | None
    ) -> Any: ...

    @abstractmethod
    def str_pad(
        self,
        data: Any,
        column: str,
        length: int,
        fill_char: str,
        side: str,
    ) -> Any: ...

    @abstractmethod
    def str_concat(
        self,
        data: Any,
        columns: list[str],
        new_column: str,
        separator: str,
    ) -> Any: ...

    @abstractmethod
    def str_extract(
        self,
        data: Any,
        column: str,
        pattern: str,
        group_index: int,
        new_column: str,
    ) -> Any: ...

    # =========================================================================
    # Datetime operations (13)
    # =========================================================================

    @abstractmethod
    def dt_year(
        self, data: Any, column: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def dt_month(
        self, data: Any, column: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def dt_day(
        self, data: Any, column: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def dt_week(
        self, data: Any, column: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def dt_quarter(
        self, data: Any, column: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def dt_year_month(
        self, data: Any, column: str, new_column: str, fmt: str
    ) -> Any: ...

    @abstractmethod
    def dt_quarter_year(
        self, data: Any, column: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def dt_calendar_week(
        self, data: Any, column: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def dt_parse(
        self, data: Any, column: str, fmt: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def dt_format(
        self, data: Any, column: str, fmt: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def dt_diff_days(
        self,
        data: Any,
        column_a: str,
        column_b: str,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def dt_age_years(
        self,
        data: Any,
        birth_column: str,
        reference_column: str | None,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def dt_truncate(
        self, data: Any, column: str, every: str, new_column: str
    ) -> Any: ...

    @abstractmethod
    def dt_is_between(
        self,
        data: Any,
        column: str,
        start: str,
        end: str,
        new_column: str,
        closed: ClosedInterval,
    ) -> Any: ...

    # =========================================================================
    # Map operations (10)
    # =========================================================================

    @abstractmethod
    def map_values(
        self,
        data: Any,
        column: str,
        mapping: dict[Any, Any],
        default: Any,
        keep_unmapped: bool,  # noqa: FBT001
    ) -> Any: ...

    @abstractmethod
    def map_case(
        self,
        data: Any,
        column: str,
        cases: list[tuple[Any, Any]],
        default: Any,
        new_column: str,
    ) -> Any: ...

    @abstractmethod
    def map_from_column(
        self,
        data: Any,
        column: str,
        lookup_column: str,
        value_column: str,
        new_column: str,
        default: Any,
    ) -> Any: ...

    @abstractmethod
    def map_discretize(
        self,
        data: Any,
        column: str,
        bins: list[float],
        labels: list[str] | None,
        new_column: str,
        right: bool,  # noqa: FBT001
    ) -> Any: ...

    @abstractmethod
    def map_onehot(
        self,
        data: Any,
        column: str,
        categories: list[Any] | None,
        prefix: str,
        drop: Any | None,
        drop_original: bool,  # noqa: FBT001
        unknown_value: str,
    ) -> Any: ...

    @abstractmethod
    def map_ordinal(
        self,
        data: Any,
        column: str,
        categories: list[Any] | None,
        new_column: str,
        drop_original: bool,  # noqa: FBT001
        unknown_value: int,
    ) -> Any: ...

    @abstractmethod
    def map_label(
        self,
        data: Any,
        column: str,
        categories: list[Any] | None,
        new_column: str,
        drop_original: bool,  # noqa: FBT001
        unknown_value: int,
    ) -> Any: ...

    @abstractmethod
    def map_bool_to_int(
        self, data: Any, column: str
    ) -> Any: ...

    @abstractmethod
    def map_null_to_value(
        self, data: Any, column: str, value: Any
    ) -> Any: ...

    @abstractmethod
    def map_value_to_null(
        self, data: Any, column: str, value: Any
    ) -> Any: ...
