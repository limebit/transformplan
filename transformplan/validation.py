"""Schema validation for TransformPlan operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from .filters import Filter


# =============================================================================
# Type categories for validation
# =============================================================================

NUMERIC_TYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}

STRING_TYPES = {pl.Utf8, pl.String}

DATETIME_TYPES = {pl.Date, pl.Datetime, pl.Time, pl.Duration}

BOOLEAN_TYPES = {pl.Boolean}


def is_numeric(dtype: pl.DataType) -> bool:
    """Check if dtype is numeric."""
    return dtype in NUMERIC_TYPES or dtype.base_type() in NUMERIC_TYPES


def is_string(dtype: pl.DataType) -> bool:
    """Check if dtype is string."""
    return dtype in STRING_TYPES or dtype.base_type() in STRING_TYPES


def is_datetime(dtype: pl.DataType) -> bool:
    """Check if dtype is datetime-related."""
    return dtype in DATETIME_TYPES or dtype.base_type() in DATETIME_TYPES


def is_boolean(dtype: pl.DataType) -> bool:
    """Check if dtype is boolean."""
    return dtype in BOOLEAN_TYPES or dtype.base_type() in BOOLEAN_TYPES


def dtype_name(dtype: pl.DataType) -> str:
    """Get a readable name for a dtype."""
    return str(dtype).split("(")[0]


# =============================================================================
# Validation result classes
# =============================================================================


@dataclass
class ValidationError:
    """A single validation error."""

    step: int
    operation: str
    message: str

    def __str__(self) -> str:
        return f"Step {self.step} ({self.operation}): {self.message}"


class ValidationResult:
    """Result of schema validation."""

    def __init__(self) -> None:
        self._errors: list[ValidationError] = []

    def add_error(self, step: int, operation: str, message: str) -> None:
        self._errors.append(ValidationError(step, operation, message))

    @property
    def is_valid(self) -> bool:
        return len(self._errors) == 0

    @property
    def errors(self) -> list[ValidationError]:
        return self._errors

    def raise_if_invalid(self) -> None:
        """Raise ValidationError if validation failed."""
        if not self.is_valid:
            error_messages = "\n".join(f"  - {e}" for e in self._errors)
            raise SchemaValidationError(
                f"Schema validation failed with {len(self._errors)} error(s):\n{error_messages}"
            )

    def __repr__(self) -> str:
        if self.is_valid:
            return "ValidationResult(valid=True)"
        return f"ValidationResult(valid=False, errors={len(self._errors)})"


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    pass


# =============================================================================
# Schema tracker
# =============================================================================


class SchemaTracker:
    """Tracks schema changes through a pipeline for validation."""

    def __init__(self, schema: dict[str, pl.DataType]) -> None:
        self._schema = dict(schema)

    @property
    def columns(self) -> set[str]:
        return set(self._schema.keys())

    def has_column(self, name: str) -> bool:
        return name in self._schema

    def get_dtype(self, name: str) -> pl.DataType | None:
        return self._schema.get(name)

    def drop_column(self, name: str) -> None:
        self._schema.pop(name, None)

    def add_column(self, name: str, dtype: pl.DataType) -> None:
        self._schema[name] = dtype

    def rename_column(self, old_name: str, new_name: str) -> None:
        if old_name in self._schema:
            self._schema[new_name] = self._schema.pop(old_name)

    def set_dtype(self, name: str, dtype: pl.DataType) -> None:
        if name in self._schema:
            self._schema[name] = dtype

    def set_columns(self, columns: list[str]) -> None:
        """Keep only the specified columns in order."""
        self._schema = {col: self._schema[col] for col in columns if col in self._schema}


# =============================================================================
# Helper functions
# =============================================================================


def _check_column_exists(
    tracker: SchemaTracker,
    column: str,
    result: ValidationResult,
    step: int,
    op_name: str,
) -> bool:
    """Check if column exists, add error if not. Returns True if exists."""
    if not tracker.has_column(column):
        result.add_error(step, op_name, f"Column '{column}' does not exist")
        return False
    return True


def _check_column_numeric(
    tracker: SchemaTracker,
    column: str,
    result: ValidationResult,
    step: int,
    op_name: str,
) -> bool:
    """Check if column is numeric, add error if not. Returns True if numeric."""
    dtype = tracker.get_dtype(column)
    if dtype and not is_numeric(dtype):
        result.add_error(
            step, op_name, f"Column '{column}' is {dtype_name(dtype)}, expected numeric"
        )
        return False
    return True


def _check_column_string(
    tracker: SchemaTracker,
    column: str,
    result: ValidationResult,
    step: int,
    op_name: str,
) -> bool:
    """Check if column is string, add error if not. Returns True if string."""
    dtype = tracker.get_dtype(column)
    if dtype and not is_string(dtype):
        result.add_error(
            step, op_name, f"Column '{column}' is {dtype_name(dtype)}, expected string"
        )
        return False
    return True


def _check_column_datetime(
    tracker: SchemaTracker,
    column: str,
    result: ValidationResult,
    step: int,
    op_name: str,
) -> bool:
    """Check if column is datetime, add error if not. Returns True if datetime."""
    dtype = tracker.get_dtype(column)
    if dtype and not is_datetime(dtype):
        result.add_error(
            step, op_name, f"Column '{column}' is {dtype_name(dtype)}, expected date/datetime"
        )
        return False
    return True


# =============================================================================
# Column operation validators
# =============================================================================


def _validate_col_drop(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    if _check_column_exists(tracker, column, result, step, "col_drop"):
        tracker.drop_column(column)


def _validate_col_rename(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_name = params["new_name"]
    if not tracker.has_column(column):
        result.add_error(step, "col_rename", f"Column '{column}' does not exist")
    elif tracker.has_column(new_name):
        result.add_error(step, "col_rename", f"Column '{new_name}' already exists")
    else:
        tracker.rename_column(column, new_name)


def _validate_col_cast(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    target_dtype = params["dtype"]
    if _check_column_exists(tracker, column, result, step, "col_cast"):
        tracker.set_dtype(column, target_dtype)


def _validate_col_reorder(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params["columns"]
    missing = [col for col in columns if not tracker.has_column(col)]
    if missing:
        result.add_error(step, "col_reorder", f"Columns do not exist: {missing}")
    else:
        tracker.set_columns(columns)


def _validate_col_select(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params["columns"]
    missing = [col for col in columns if not tracker.has_column(col)]
    if missing:
        result.add_error(step, "col_select", f"Columns do not exist: {missing}")
    else:
        tracker.set_columns(columns)


def _validate_col_duplicate(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_name = params["new_name"]
    if not tracker.has_column(column):
        result.add_error(step, "col_duplicate", f"Column '{column}' does not exist")
    elif tracker.has_column(new_name):
        result.add_error(step, "col_duplicate", f"Column '{new_name}' already exists")
    else:
        tracker.add_column(new_name, tracker.get_dtype(column))


def _validate_col_fill_null(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    _check_column_exists(tracker, column, result, step, "col_fill_null")


def _validate_col_drop_null(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params.get("columns")
    if columns:
        missing = [col for col in columns if not tracker.has_column(col)]
        if missing:
            result.add_error(step, "col_drop_null", f"Columns do not exist: {missing}")


def _validate_col_drop_zero(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    if _check_column_exists(tracker, column, result, step, "col_drop_zero"):
        _check_column_numeric(tracker, column, result, step, "col_drop_zero")


def _validate_col_add(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    new_column = params["new_column"]
    expr = params.get("expr")
    if tracker.has_column(new_column):
        result.add_error(step, "col_add", f"Column '{new_column}' already exists")
    elif expr and not tracker.has_column(expr):
        result.add_error(step, "col_add", f"Source column '{expr}' does not exist")
    else:
        # Infer type from source or value
        if expr:
            tracker.add_column(new_column, tracker.get_dtype(expr))
        else:
            tracker.add_column(new_column, pl.Utf8)  # default to string for literals


def _validate_col_add_uuid(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    if tracker.has_column(column):
        result.add_error(step, "col_add_uuid", f"Column '{column}' already exists")
    else:
        tracker.add_column(column, pl.Utf8)


def _validate_col_hash(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params["columns"]
    new_column = params["new_column"]
    missing = [col for col in columns if not tracker.has_column(col)]
    if missing:
        result.add_error(step, "col_hash", f"Columns do not exist: {missing}")
    if tracker.has_column(new_column):
        result.add_error(step, "col_hash", f"Column '{new_column}' already exists")
    else:
        tracker.add_column(new_column, pl.Utf8)


def _validate_col_coalesce(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params["columns"]
    new_column = params["new_column"]
    missing = [col for col in columns if not tracker.has_column(col)]
    if missing:
        result.add_error(step, "col_coalesce", f"Columns do not exist: {missing}")
    else:
        # Result type is type of first column
        tracker.add_column(new_column, tracker.get_dtype(columns[0]))


# =============================================================================
# Math operation validators
# =============================================================================


def _validate_math_scalar(
    tracker: SchemaTracker,
    params: dict[str, Any],
    result: ValidationResult,
    step: int,
    op_name: str,
) -> None:
    """Validate math operation with scalar: column must exist and be numeric."""
    column = params["column"]
    if _check_column_exists(tracker, column, result, step, op_name):
        _check_column_numeric(tracker, column, result, step, op_name)


def _validate_math_columns(
    tracker: SchemaTracker,
    params: dict[str, Any],
    result: ValidationResult,
    step: int,
    op_name: str,
) -> None:
    """Validate math operation between columns: both must exist and be numeric."""
    column_a = params["column_a"]
    column_b = params["column_b"]
    new_column = params["new_column"]

    a_exists = _check_column_exists(tracker, column_a, result, step, op_name)
    b_exists = _check_column_exists(tracker, column_b, result, step, op_name)

    if a_exists:
        _check_column_numeric(tracker, column_a, result, step, op_name)
    if b_exists:
        _check_column_numeric(tracker, column_b, result, step, op_name)

    tracker.add_column(new_column, pl.Float64)


def _validate_math_cumsum(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_column = params["new_column"]
    group_by = params.get("group_by")

    if _check_column_exists(tracker, column, result, step, "math_cumsum"):
        _check_column_numeric(tracker, column, result, step, "math_cumsum")

    if group_by:
        missing = [col for col in group_by if not tracker.has_column(col)]
        if missing:
            result.add_error(step, "math_cumsum", f"Group-by columns do not exist: {missing}")

    if new_column != column:
        tracker.add_column(new_column, tracker.get_dtype(column))


def _validate_math_rank(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_column = params["new_column"]
    group_by = params.get("group_by")

    _check_column_exists(tracker, column, result, step, "math_rank")

    if group_by:
        missing = [col for col in group_by if not tracker.has_column(col)]
        if missing:
            result.add_error(step, "math_rank", f"Group-by columns do not exist: {missing}")

    tracker.add_column(new_column, pl.UInt32)


def _validate_math_percent_of(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    total_column = params["total_column"]
    new_column = params["new_column"]

    if _check_column_exists(tracker, column, result, step, "math_percent_of"):
        _check_column_numeric(tracker, column, result, step, "math_percent_of")
    if _check_column_exists(tracker, total_column, result, step, "math_percent_of"):
        _check_column_numeric(tracker, total_column, result, step, "math_percent_of")

    tracker.add_column(new_column, pl.Float64)


# =============================================================================
# String operation validators
# =============================================================================


def _validate_str_op(
    tracker: SchemaTracker,
    params: dict[str, Any],
    result: ValidationResult,
    step: int,
    op_name: str,
) -> None:
    """Validate string operation: column must exist and be string."""
    column = params["column"]
    if _check_column_exists(tracker, column, result, step, op_name):
        _check_column_string(tracker, column, result, step, op_name)


def _validate_str_split(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_columns = params.get("new_columns")

    if _check_column_exists(tracker, column, result, step, "str_split"):
        _check_column_string(tracker, column, result, step, "str_split")

    if new_columns:
        for new_col in new_columns:
            if tracker.has_column(new_col):
                result.add_error(step, "str_split", f"Column '{new_col}' already exists")
            else:
                tracker.add_column(new_col, pl.Utf8)
        if not params.get("keep_original", False):
            tracker.drop_column(column)


def _validate_str_concat(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params["columns"]
    new_column = params["new_column"]

    for col in columns:
        if _check_column_exists(tracker, col, result, step, "str_concat"):
            _check_column_string(tracker, col, result, step, "str_concat")

    tracker.add_column(new_column, pl.Utf8)


def _validate_str_extract(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_column = params["new_column"]

    if _check_column_exists(tracker, column, result, step, "str_extract"):
        _check_column_string(tracker, column, result, step, "str_extract")

    if new_column != column:
        tracker.add_column(new_column, pl.Utf8)


# =============================================================================
# Datetime operation validators
# =============================================================================


def _validate_dt_op(
    tracker: SchemaTracker,
    params: dict[str, Any],
    result: ValidationResult,
    step: int,
    op_name: str,
    output_dtype: pl.DataType = pl.Int32,
) -> None:
    """Validate datetime operation: column must exist and be datetime."""
    column = params["column"]
    new_column = params.get("new_column", column)

    if _check_column_exists(tracker, column, result, step, op_name):
        _check_column_datetime(tracker, column, result, step, op_name)

    if new_column != column:
        tracker.add_column(new_column, output_dtype)


def _validate_dt_parse(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_column = params.get("new_column", column)

    if _check_column_exists(tracker, column, result, step, "dt_parse"):
        _check_column_string(tracker, column, result, step, "dt_parse")

    tracker.set_dtype(new_column, pl.Date)


def _validate_dt_format(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_column = params.get("new_column", column)

    if _check_column_exists(tracker, column, result, step, "dt_format"):
        _check_column_datetime(tracker, column, result, step, "dt_format")

    if new_column != column:
        tracker.add_column(new_column, pl.Utf8)
    else:
        tracker.set_dtype(column, pl.Utf8)


def _validate_dt_diff_days(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column_a = params["column_a"]
    column_b = params["column_b"]
    new_column = params["new_column"]

    if _check_column_exists(tracker, column_a, result, step, "dt_diff_days"):
        _check_column_datetime(tracker, column_a, result, step, "dt_diff_days")
    if _check_column_exists(tracker, column_b, result, step, "dt_diff_days"):
        _check_column_datetime(tracker, column_b, result, step, "dt_diff_days")

    tracker.add_column(new_column, pl.Int64)


def _validate_dt_age_years(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    birth_column = params["birth_column"]
    reference_column = params.get("reference_column")
    new_column = params["new_column"]

    if _check_column_exists(tracker, birth_column, result, step, "dt_age_years"):
        _check_column_datetime(tracker, birth_column, result, step, "dt_age_years")

    if reference_column:
        if _check_column_exists(tracker, reference_column, result, step, "dt_age_years"):
            _check_column_datetime(tracker, reference_column, result, step, "dt_age_years")

    tracker.add_column(new_column, pl.Int64)


def _validate_dt_is_between(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_column = params["new_column"]

    if _check_column_exists(tracker, column, result, step, "dt_is_between"):
        _check_column_datetime(tracker, column, result, step, "dt_is_between")

    tracker.add_column(new_column, pl.Boolean)


# =============================================================================
# Row operation validators
# =============================================================================


def _validate_rows_drop_nulls(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params.get("columns")
    if columns:
        missing = [col for col in columns if not tracker.has_column(col)]
        if missing:
            result.add_error(step, "rows_drop_nulls", f"Columns do not exist: {missing}")


def _validate_rows_unique(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params.get("columns")
    if columns:
        missing = [col for col in columns if not tracker.has_column(col)]
        if missing:
            result.add_error(step, "rows_unique", f"Columns do not exist: {missing}")


def _validate_filter_columns(
    filter_dict: dict[str, Any], tracker: SchemaTracker, result: ValidationResult, step: int, op_name: str
) -> list[str]:
    """Recursively validate columns and types from a filter dict."""
    missing = []
    filter_type = filter_dict.get("type")

    if filter_type in ("and", "or"):
        missing.extend(_validate_filter_columns(filter_dict["left"], tracker, result, step, op_name))
        missing.extend(_validate_filter_columns(filter_dict["right"], tracker, result, step, op_name))
    elif filter_type == "not":
        missing.extend(_validate_filter_columns(filter_dict["operand"], tracker, result, step, op_name))
    elif "column" in filter_dict:
        column = filter_dict["column"]
        if not tracker.has_column(column):
            missing.append(column)
        else:
            # Type checking for comparison operators
            dtype = tracker.get_dtype(column)

            # Numeric comparisons
            if filter_type in ("gt", "ge", "lt", "le", "between"):
                if dtype and not is_numeric(dtype) and not is_datetime(dtype):
                    result.add_error(
                        step, op_name,
                        f"Column '{column}' is {dtype_name(dtype)}, cannot use numeric comparison"
                    )

            # String operations
            if filter_type in ("str_contains", "str_starts_with", "str_ends_with"):
                if dtype and not is_string(dtype):
                    result.add_error(
                        step, op_name,
                        f"Column '{column}' is {dtype_name(dtype)}, cannot use string filter"
                    )

    return missing


def _validate_rows_filter(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    filter_dict = params.get("filter", {})
    missing = _validate_filter_columns(filter_dict, tracker, result, step, "rows_filter")
    if missing:
        unique_missing = list(dict.fromkeys(missing))
        result.add_error(step, "rows_filter", f"Columns do not exist: {unique_missing}")


def _validate_rows_drop(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    filter_dict = params.get("filter", {})
    missing = _validate_filter_columns(filter_dict, tracker, result, step, "rows_drop")
    if missing:
        unique_missing = list(dict.fromkeys(missing))
        result.add_error(step, "rows_drop", f"Columns do not exist: {unique_missing}")


def _validate_rows_flag(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    filter_dict = params.get("filter", {})
    new_column = params["new_column"]

    missing = _validate_filter_columns(filter_dict, tracker, result, step, "rows_flag")
    if missing:
        unique_missing = list(dict.fromkeys(missing))
        result.add_error(step, "rows_flag", f"Columns do not exist: {unique_missing}")

    if tracker.has_column(new_column):
        result.add_error(step, "rows_flag", f"Column '{new_column}' already exists")
    else:
        tracker.add_column(new_column, pl.Boolean)


def _validate_rows_sort(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    by = params["by"]
    missing = [col for col in by if not tracker.has_column(col)]
    if missing:
        result.add_error(step, "rows_sort", f"Columns do not exist: {missing}")


def _validate_rows_deduplicate(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params["columns"]
    sort_by = params["sort_by"]

    missing = [col for col in columns if not tracker.has_column(col)]
    if missing:
        result.add_error(step, "rows_deduplicate", f"Columns do not exist: {missing}")

    if not tracker.has_column(sort_by):
        result.add_error(step, "rows_deduplicate", f"Sort column '{sort_by}' does not exist")


def _validate_rows_explode(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    if _check_column_exists(tracker, column, result, step, "rows_explode"):
        dtype = tracker.get_dtype(column)
        if dtype and not isinstance(dtype, pl.List):
            result.add_error(step, "rows_explode", f"Column '{column}' is {dtype_name(dtype)}, expected List")


def _validate_rows_melt(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    id_columns = params["id_columns"]
    value_columns = params["value_columns"]

    missing_id = [col for col in id_columns if not tracker.has_column(col)]
    missing_val = [col for col in value_columns if not tracker.has_column(col)]

    if missing_id:
        result.add_error(step, "rows_melt", f"ID columns do not exist: {missing_id}")
    if missing_val:
        result.add_error(step, "rows_melt", f"Value columns do not exist: {missing_val}")


def _validate_rows_pivot(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    index = params["index"]
    columns = params["columns"]
    values = params["values"]

    missing = [col for col in index if not tracker.has_column(col)]
    if missing:
        result.add_error(step, "rows_pivot", f"Index columns do not exist: {missing}")

    if not tracker.has_column(columns):
        result.add_error(step, "rows_pivot", f"Pivot column '{columns}' does not exist")

    if not tracker.has_column(values):
        result.add_error(step, "rows_pivot", f"Values column '{values}' does not exist")


# =============================================================================
# Map operation validators
# =============================================================================


def _validate_map_values(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    _check_column_exists(tracker, column, result, step, "map_values")


def _validate_map_discretize(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_column = params["new_column"]

    if _check_column_exists(tracker, column, result, step, "map_discretize"):
        _check_column_numeric(tracker, column, result, step, "map_discretize")

    if new_column != column:
        tracker.add_column(new_column, pl.Utf8)
    else:
        tracker.set_dtype(column, pl.Utf8)


def _validate_map_from_column(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    lookup_column = params["lookup_column"]
    value_column = params["value_column"]
    new_column = params["new_column"]

    _check_column_exists(tracker, column, result, step, "map_from_column")
    _check_column_exists(tracker, lookup_column, result, step, "map_from_column")
    _check_column_exists(tracker, value_column, result, step, "map_from_column")

    if new_column != column:
        tracker.add_column(new_column, tracker.get_dtype(value_column))


# =============================================================================
# Validator registry
# =============================================================================

_VALIDATORS: dict[str, Any] = {
    # Column ops
    "col_drop": _validate_col_drop,
    "col_rename": _validate_col_rename,
    "col_cast": _validate_col_cast,
    "col_reorder": _validate_col_reorder,
    "col_select": _validate_col_select,
    "col_duplicate": _validate_col_duplicate,
    "col_fill_null": _validate_col_fill_null,
    "col_drop_null": _validate_col_drop_null,
    "col_drop_zero": _validate_col_drop_zero,
    "col_add": _validate_col_add,
    "col_add_uuid": _validate_col_add_uuid,
    "col_hash": _validate_col_hash,
    "col_coalesce": _validate_col_coalesce,
    # Math ops
    "math_add": lambda t, p, r, s: _validate_math_scalar(t, p, r, s, "math_add"),
    "math_subtract": lambda t, p, r, s: _validate_math_scalar(t, p, r, s, "math_subtract"),
    "math_multiply": lambda t, p, r, s: _validate_math_scalar(t, p, r, s, "math_multiply"),
    "math_divide": lambda t, p, r, s: _validate_math_scalar(t, p, r, s, "math_divide"),
    "math_clamp": lambda t, p, r, s: _validate_math_scalar(t, p, r, s, "math_clamp"),
    "math_abs": lambda t, p, r, s: _validate_math_scalar(t, p, r, s, "math_abs"),
    "math_round": lambda t, p, r, s: _validate_math_scalar(t, p, r, s, "math_round"),
    "math_set_min": lambda t, p, r, s: _validate_math_scalar(t, p, r, s, "math_set_min"),
    "math_set_max": lambda t, p, r, s: _validate_math_scalar(t, p, r, s, "math_set_max"),
    "math_add_columns": lambda t, p, r, s: _validate_math_columns(t, p, r, s, "math_add_columns"),
    "math_subtract_columns": lambda t, p, r, s: _validate_math_columns(t, p, r, s, "math_subtract_columns"),
    "math_multiply_columns": lambda t, p, r, s: _validate_math_columns(t, p, r, s, "math_multiply_columns"),
    "math_divide_columns": lambda t, p, r, s: _validate_math_columns(t, p, r, s, "math_divide_columns"),
    "math_cumsum": _validate_math_cumsum,
    "math_rank": _validate_math_rank,
    "math_percent_of": _validate_math_percent_of,
    # String ops
    "str_replace": lambda t, p, r, s: _validate_str_op(t, p, r, s, "str_replace"),
    "str_slice": lambda t, p, r, s: _validate_str_op(t, p, r, s, "str_slice"),
    "str_truncate": lambda t, p, r, s: _validate_str_op(t, p, r, s, "str_truncate"),
    "str_lower": lambda t, p, r, s: _validate_str_op(t, p, r, s, "str_lower"),
    "str_upper": lambda t, p, r, s: _validate_str_op(t, p, r, s, "str_upper"),
    "str_strip": lambda t, p, r, s: _validate_str_op(t, p, r, s, "str_strip"),
    "str_pad": lambda t, p, r, s: _validate_str_op(t, p, r, s, "str_pad"),
    "str_split": _validate_str_split,
    "str_concat": _validate_str_concat,
    "str_extract": _validate_str_extract,
    # Datetime ops
    "dt_year": lambda t, p, r, s: _validate_dt_op(t, p, r, s, "dt_year"),
    "dt_month": lambda t, p, r, s: _validate_dt_op(t, p, r, s, "dt_month"),
    "dt_day": lambda t, p, r, s: _validate_dt_op(t, p, r, s, "dt_day"),
    "dt_week": lambda t, p, r, s: _validate_dt_op(t, p, r, s, "dt_week"),
    "dt_quarter": lambda t, p, r, s: _validate_dt_op(t, p, r, s, "dt_quarter"),
    "dt_year_month": lambda t, p, r, s: _validate_dt_op(t, p, r, s, "dt_year_month", pl.Utf8),
    "dt_quarter_year": lambda t, p, r, s: _validate_dt_op(t, p, r, s, "dt_quarter_year", pl.Utf8),
    "dt_calendar_week": lambda t, p, r, s: _validate_dt_op(t, p, r, s, "dt_calendar_week", pl.Utf8),
    "dt_truncate": lambda t, p, r, s: _validate_dt_op(t, p, r, s, "dt_truncate", pl.Date),
    "dt_parse": _validate_dt_parse,
    "dt_format": _validate_dt_format,
    "dt_diff_days": _validate_dt_diff_days,
    "dt_age_years": _validate_dt_age_years,
    "dt_is_between": _validate_dt_is_between,
    # Row ops
    "rows_drop_nulls": _validate_rows_drop_nulls,
    "rows_unique": _validate_rows_unique,
    "rows_filter": _validate_rows_filter,
    "rows_drop": _validate_rows_drop,
    "rows_flag": _validate_rows_flag,
    "rows_sort": _validate_rows_sort,
    "rows_deduplicate": _validate_rows_deduplicate,
    "rows_explode": _validate_rows_explode,
    "rows_melt": _validate_rows_melt,
    "rows_pivot": _validate_rows_pivot,
    # Map ops
    "map_values": _validate_map_values,
    "map_discretize": _validate_map_discretize,
    "map_from_column": _validate_map_from_column,
}


def validate_schema(
    operations: list[tuple[Any, dict[str, Any]]], schema: dict[str, pl.DataType]
) -> ValidationResult:
    """Validate all operations against the given schema.

    Args:
        operations: List of (method, params) tuples from TransformPlan.
        schema: Initial DataFrame schema.

    Returns:
        ValidationResult with any errors found.
    """
    result = ValidationResult()
    tracker = SchemaTracker(schema)

    for step, (method, params) in enumerate(operations, start=1):
        op_name = method.__name__.lstrip("_")
        validator = _VALIDATORS.get(op_name)
        if validator:
            validator(tracker, params, result, step)

    return result
