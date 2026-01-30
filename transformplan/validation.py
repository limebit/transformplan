"""Schema validation for TransformPlan operations.

This module provides schema validation and dry-run preview capabilities for
TransformPlan pipelines. It validates operations against DataFrame schemas
before execution, catching errors like missing columns or type mismatches.

Classes:
    ValidationResult: Container for validation errors.
    ValidationError: Single validation error with step and message.
    SchemaValidationError: Exception raised when validation fails.
    DryRunResult: Preview showing what each operation will do.
    DryRunStep: Single step in a dry run with schema changes.
    SchemaTracker: Internal tracker for schema changes through a pipeline.

Functions:
    validate_schema: Validate operations against a schema.
    dry_run_schema: Preview operations without executing.

Type Checking Functions:
    is_numeric: Check if dtype is numeric.
    is_string: Check if dtype is string.
    is_datetime: Check if dtype is datetime-related.
    is_boolean: Check if dtype is boolean.

Example:
    >>> from transformplan import TransformPlan
    >>>
    >>> plan = TransformPlan().col_drop("nonexistent")
    >>> result = plan.validate(df)
    >>> if not result.is_valid:
    ...     for error in result.errors:
    ...         print(error)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import polars as pl

# =============================================================================
# Type categories for validation
# =============================================================================

NUMERIC_TYPES = {
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

STRING_TYPES = {pl.Utf8(), pl.String()}

DATETIME_TYPES = {pl.Date(), pl.Datetime(), pl.Time(), pl.Duration()}

BOOLEAN_TYPES = {pl.Boolean()}


def is_numeric(dtype: pl.DataType) -> bool:
    """Check if dtype is numeric.

    Returns:
        True if dtype is numeric, False otherwise.
    """
    return dtype in NUMERIC_TYPES or dtype.base_type()() in NUMERIC_TYPES


def is_string(dtype: pl.DataType) -> bool:
    """Check if dtype is string.

    Returns:
        True if dtype is string, False otherwise.
    """
    return dtype in STRING_TYPES or dtype.base_type()() in STRING_TYPES


def is_datetime(dtype: pl.DataType) -> bool:
    """Check if dtype is datetime-related.

    Returns:
        True if dtype is datetime-related, False otherwise.
    """
    return dtype in DATETIME_TYPES or dtype.base_type()() in DATETIME_TYPES


def is_boolean(dtype: pl.DataType) -> bool:
    """Check if dtype is boolean.

    Returns:
        True if dtype is boolean, False otherwise.
    """
    return dtype in BOOLEAN_TYPES or dtype.base_type()() in BOOLEAN_TYPES


def dtype_name(dtype: pl.DataType) -> str:
    """Get a readable name for a dtype.

    Returns:
        String representation of the dtype.
    """
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
        """Return error message string.

        Returns:
            Formatted error message.
        """
        return f"Step {self.step} ({self.operation}): {self.message}"


class ValidationResult:
    """Result of schema validation."""

    def __init__(self) -> None:
        """Initialize an empty validation result."""
        self._errors: list[ValidationError] = []

    def add_error(self, step: int, operation: str, message: str) -> None:
        """Add a validation error."""
        self._errors.append(ValidationError(step, operation, message))

    @property
    def is_valid(self) -> bool:
        """Check if validation passed.

        Returns:
            True if no errors, False otherwise.
        """
        return len(self._errors) == 0

    @property
    def errors(self) -> list[ValidationError]:
        """Get list of validation errors.

        Returns:
            List of ValidationError instances.
        """
        return self._errors

    def raise_if_invalid(self) -> None:
        """Raise SchemaValidationError if validation failed.

        Raises:
            SchemaValidationError: If validation failed with errors.
        """
        if not self.is_valid:
            error_messages = "\n".join(f"  - {e}" for e in self._errors)
            msg = f"Schema validation failed with {len(self._errors)} error(s):\n{error_messages}"
            raise SchemaValidationError(msg)

    def __repr__(self) -> str:
        """Return string representation of validation result.

        Returns:
            Human-readable representation.
        """
        if self.is_valid:
            return "ValidationResult(valid=True)"
        return f"ValidationResult(valid=False, errors={len(self._errors)})"


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    pass


# =============================================================================
# Dry run result
# =============================================================================


@dataclass
class DryRunStep:
    """A single step in a dry run."""

    step: int
    operation: str
    params: dict[str, Any]
    schema_before: dict[str, str]
    schema_after: dict[str, str]
    columns_added: list[str]
    columns_removed: list[str]
    columns_modified: list[str]
    error: str | None = None


class DryRunResult:
    """Result of a dry run showing what a pipeline will do."""

    def __init__(
        self,
        input_schema: dict[str, pl.DataType],
        steps: list[DryRunStep],
        validation: ValidationResult,
    ) -> None:
        """Initialize DryRunResult.

        Args:
            input_schema: Initial schema as column name to dtype mapping.
            steps: List of dry run steps.
            validation: Validation result with any errors.
        """
        self._input_schema = input_schema
        self._steps = steps
        self._validation = validation

    @property
    def is_valid(self) -> bool:
        """Whether the pipeline passed validation.

        Returns:
            True if validation passed, False otherwise.
        """
        return self._validation.is_valid

    @property
    def errors(self) -> list[ValidationError]:
        """Validation errors.

        Returns:
            List of validation errors.
        """
        return self._validation.errors

    @property
    def steps(self) -> list[DryRunStep]:
        """List of dry run steps.

        Returns:
            List of DryRunStep instances.
        """
        return self._steps

    @property
    def input_schema(self) -> dict[str, pl.DataType]:
        """Input schema.

        Returns:
            Dictionary mapping column names to dtypes.
        """
        return self._input_schema

    @property
    def output_schema(self) -> dict[str, str]:
        """Predicted output schema after all operations.

        Returns:
            Dictionary mapping column names to dtype names.
        """
        if self._steps:
            return self._steps[-1].schema_after
        return {k: dtype_name(v) for k, v in self._input_schema.items()}

    @property
    def input_columns(self) -> list[str]:
        """Input column names.

        Returns:
            List of input column names.
        """
        return list(self._input_schema.keys())

    @property
    def output_columns(self) -> list[str]:
        """Predicted output column names.

        Returns:
            List of predicted output column names.
        """
        return list(self.output_schema.keys())

    def summary(self, *, show_params: bool = True, show_schema: bool = False) -> str:  # noqa: C901
        """Generate a human-readable summary.

        Args:
            show_params: Whether to show operation parameters.
            show_schema: Whether to show full schema at each step.

        Returns:
            Formatted string.
        """
        lines = []

        # Header
        lines.extend(("=" * 70, "DRY RUN PREVIEW", "=" * 70))

        # Validation status
        if self.is_valid:
            lines.append("✓ Validation: PASSED")
        else:
            lines.append(f"✗ Validation: FAILED ({len(self.errors)} errors)")
            lines.extend(f"  - {err}" for err in self.errors)

        lines.extend(["-" * 70, f"Input: {len(self._input_schema)} columns"])
        if show_schema:
            for col, dtype in self._input_schema.items():
                lines.append(f"  {col}: {dtype_name(dtype)}")

        lines.extend(
            [
                "-" * 70,
                "",
                f"{'#':<4} {'Operation':<20} {'Columns':<15} {'Changes':<30}",
                "-" * 70,
            ]
        )

        for step in self._steps:
            step_num = str(step.step)
            op = step.operation
            col_count = len(step.schema_after)

            # Build changes string
            changes = []
            if step.columns_added:
                changes.append(f"+{step.columns_added}")
            if step.columns_removed:
                changes.append(f"-{step.columns_removed}")
            if step.columns_modified:
                changes.append(f"~{step.columns_modified}")
            changes_str = " ".join(changes) if changes else "-"

            # Error marker
            err_marker = " ✗" if step.error else ""

            lines.append(
                f"{step_num:<4} {op:<20} {col_count:<15} {changes_str:<30}{err_marker}"
            )

            # Params
            if show_params and step.params:
                params_str = _format_params_short(step.params)
                lines.append(f"     └─ {params_str}")

            # Error detail
            if step.error:
                lines.append(f"     └─ ERROR: {step.error}")

            # Full schema
            if show_schema:
                lines.append(f"     Schema: {step.schema_after}")

        lines.extend(["=" * 70, f"Output: {len(self.output_schema)} columns"])
        if show_schema:
            for col, dtype in self.output_schema.items():
                lines.append(f"  {col}: {dtype}")

        return "\n".join(lines)

    def print(self, *, show_params: bool = True, show_schema: bool = False) -> None:
        """Print the dry run summary."""
        print(self.summary(show_params=show_params, show_schema=show_schema))  # noqa: T201

    def __repr__(self) -> str:
        """Return string representation of dry run result.

        Returns:
            Human-readable representation.
        """
        status = "valid" if self.is_valid else f"invalid ({len(self.errors)} errors)"
        return f"DryRunResult({len(self._steps)} steps, {status})"


def _format_params_short(params: dict[str, Any], max_length: int = 55) -> str:
    """Format params dict as a short string.

    Returns:
        Formatted string representation of params.
    """
    parts = []
    for key, value in params.items():
        if isinstance(value, dict) and "type" in value:
            # Filter - just show type
            parts.append(f"{key}=<filter>")
        elif isinstance(value, list) and len(value) > 3:
            parts.append(f"{key}=[...{len(value)} items]")
        elif isinstance(value, str) and len(value) > 20:
            parts.append(f"{key}='{value[:17]}...'")
        else:
            parts.append(f"{key}={value!r}")

    result = ", ".join(parts)
    if len(result) > max_length:
        result = result[: max_length - 3] + "..."
    return result


# =============================================================================
# Schema tracker
# =============================================================================


class SchemaTracker:
    """Tracks schema changes through a pipeline for validation."""

    def __init__(self, schema: dict[str, pl.DataType]) -> None:
        """Initialize tracker with a schema.

        Args:
            schema: Initial schema as column name to dtype mapping.
        """
        self._schema = dict(schema)

    @property
    def columns(self) -> set[str]:
        """Get set of column names.

        Returns:
            Set of column names.
        """
        return set(self._schema.keys())

    def has_column(self, name: str) -> bool:
        """Check if column exists.

        Returns:
            True if column exists, False otherwise.
        """
        return name in self._schema

    def get_dtype(self, name: str) -> pl.DataType | None:
        """Get dtype for a column.

        Returns:
            DataType or None if column doesn't exist.
        """
        return self._schema.get(name)

    def drop_column(self, name: str) -> None:
        """Remove a column from the schema."""
        self._schema.pop(name, None)

    def add_column(self, name: str, dtype: pl.DataType | None) -> None:
        """Add a column to the schema."""
        if dtype is not None:
            self._schema[name] = dtype

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column in the schema."""
        if old_name in self._schema:
            self._schema[new_name] = self._schema.pop(old_name)

    def set_dtype(self, name: str, dtype: pl.DataType) -> None:
        """Change the dtype of an existing column."""
        if name in self._schema:
            self._schema[name] = dtype

    def set_columns(self, columns: list[str]) -> None:
        """Keep only the specified columns in order."""
        self._schema = {
            col: self._schema[col] for col in columns if col in self._schema
        }


# Type alias for validator functions
ValidatorFunc = Callable[[SchemaTracker, dict[str, Any], ValidationResult, int], None]


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
    """Check if column exists, add error if not.

    Returns:
        True if column exists, False otherwise.
    """
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
    """Check if column is numeric, add error if not.

    Returns:
        True if column is numeric, False otherwise.
    """
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
    """Check if column is string, add error if not.

    Returns:
        True if column is string, False otherwise.
    """
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
    """Check if column is datetime, add error if not.

    Returns:
        True if column is datetime, False otherwise.
    """
    dtype = tracker.get_dtype(column)
    if dtype and not is_datetime(dtype):
        result.add_error(
            step,
            op_name,
            f"Column '{column}' is {dtype_name(dtype)}, expected date/datetime",
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
            tracker.add_column(new_column, pl.Utf8())  # default to string for literals


def _validate_col_add_uuid(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    if tracker.has_column(column):
        result.add_error(step, "col_add_uuid", f"Column '{column}' already exists")
    else:
        tracker.add_column(column, pl.Utf8())


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
        tracker.add_column(new_column, pl.Utf8())


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

    tracker.add_column(new_column, pl.Float64())


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
            result.add_error(
                step, "math_cumsum", f"Group-by columns do not exist: {missing}"
            )

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
            result.add_error(
                step, "math_rank", f"Group-by columns do not exist: {missing}"
            )

    tracker.add_column(new_column, pl.UInt32())


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

    tracker.add_column(new_column, pl.Float64())


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
                result.add_error(
                    step, "str_split", f"Column '{new_col}' already exists"
                )
            else:
                tracker.add_column(new_col, pl.Utf8())
        if not params.get("keep_original"):
            tracker.drop_column(column)


def _validate_str_concat(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params["columns"]
    new_column = params["new_column"]

    for col in columns:
        if _check_column_exists(tracker, col, result, step, "str_concat"):
            _check_column_string(tracker, col, result, step, "str_concat")

    tracker.add_column(new_column, pl.Utf8())


def _validate_str_extract(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_column = params["new_column"]

    if _check_column_exists(tracker, column, result, step, "str_extract"):
        _check_column_string(tracker, column, result, step, "str_extract")

    if new_column != column:
        tracker.add_column(new_column, pl.Utf8())


# =============================================================================
# Datetime operation validators
# =============================================================================


def _validate_dt_op(
    tracker: SchemaTracker,
    params: dict[str, Any],
    result: ValidationResult,
    step: int,
    op_name: str,
    output_dtype: pl.DataType | None = None,
) -> None:
    """Validate datetime operation: column must exist and be datetime."""
    if output_dtype is None:
        output_dtype = pl.Int32()
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

    tracker.set_dtype(new_column, pl.Date())


def _validate_dt_format(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_column = params.get("new_column", column)

    if _check_column_exists(tracker, column, result, step, "dt_format"):
        _check_column_datetime(tracker, column, result, step, "dt_format")

    if new_column != column:
        tracker.add_column(new_column, pl.Utf8())
    else:
        tracker.set_dtype(column, pl.Utf8())


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

    tracker.add_column(new_column, pl.Int64())


def _validate_dt_age_years(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    birth_column = params["birth_column"]
    reference_column = params.get("reference_column")
    new_column = params["new_column"]

    if _check_column_exists(tracker, birth_column, result, step, "dt_age_years"):
        _check_column_datetime(tracker, birth_column, result, step, "dt_age_years")

    if reference_column and _check_column_exists(
        tracker, reference_column, result, step, "dt_age_years"
    ):
        _check_column_datetime(tracker, reference_column, result, step, "dt_age_years")

    tracker.add_column(new_column, pl.Int64())


def _validate_dt_is_between(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    new_column = params["new_column"]

    if _check_column_exists(tracker, column, result, step, "dt_is_between"):
        _check_column_datetime(tracker, column, result, step, "dt_is_between")

    tracker.add_column(new_column, pl.Boolean())


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
            result.add_error(
                step, "rows_drop_nulls", f"Columns do not exist: {missing}"
            )


def _validate_rows_unique(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params.get("columns")
    if columns:
        missing = [col for col in columns if not tracker.has_column(col)]
        if missing:
            result.add_error(step, "rows_unique", f"Columns do not exist: {missing}")


def _validate_filter_columns(
    filter_dict: dict[str, Any],
    tracker: SchemaTracker,
    result: ValidationResult,
    step: int,
    op_name: str,
) -> list[str]:
    """Recursively validate columns and types from a filter dict.

    Returns:
        List of missing column names.
    """
    missing = []
    filter_type = filter_dict.get("type")

    if filter_type in ("and", "or"):
        missing.extend(
            _validate_filter_columns(
                filter_dict["left"], tracker, result, step, op_name
            )
        )
        missing.extend(
            _validate_filter_columns(
                filter_dict["right"], tracker, result, step, op_name
            )
        )
    elif filter_type == "not":
        missing.extend(
            _validate_filter_columns(
                filter_dict["operand"], tracker, result, step, op_name
            )
        )
    elif "column" in filter_dict:
        column = filter_dict["column"]
        if not tracker.has_column(column):
            missing.append(column)
        else:
            # Type checking for comparison operators
            dtype = tracker.get_dtype(column)

            # Numeric comparisons
            if (
                filter_type in ("gt", "ge", "lt", "le", "between")
                and dtype
                and not is_numeric(dtype)
                and not is_datetime(dtype)
            ):
                result.add_error(
                    step,
                    op_name,
                    f"Column '{column}' is {dtype_name(dtype)}, cannot use numeric comparison",
                )

            # String operations
            if (
                filter_type in ("str_contains", "str_starts_with", "str_ends_with")
                and dtype
                and not is_string(dtype)
            ):
                result.add_error(
                    step,
                    op_name,
                    f"Column '{column}' is {dtype_name(dtype)}, cannot use string filter",
                )

    return missing


def _validate_rows_filter(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    filter_dict = params.get("filter", {})
    missing = _validate_filter_columns(
        filter_dict, tracker, result, step, "rows_filter"
    )
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
        tracker.add_column(new_column, pl.Boolean())


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
        result.add_error(
            step, "rows_deduplicate", f"Sort column '{sort_by}' does not exist"
        )


def _validate_rows_explode(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    if _check_column_exists(tracker, column, result, step, "rows_explode"):
        dtype = tracker.get_dtype(column)
        if dtype and not isinstance(dtype, pl.List):
            result.add_error(
                step,
                "rows_explode",
                f"Column '{column}' is {dtype_name(dtype)}, expected List",
            )


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
        result.add_error(
            step, "rows_melt", f"Value columns do not exist: {missing_val}"
        )


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
        tracker.add_column(new_column, pl.Utf8())
    else:
        tracker.set_dtype(column, pl.Utf8())


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
# Encoding operation validators
# =============================================================================


def _resolve_drop_category(
    drop: Any | None,  # noqa: ANN401
    categories: list[Any],
    result: ValidationResult,
    step: int,
    op_name: str,
) -> tuple[Any | None, bool]:
    """Resolve the category to drop for one-hot encoding.

    Literal values take precedence over keywords "first"/"last".

    Returns:
        Tuple of (drop_category, is_valid). If is_valid is False, an error was added.
    """
    if drop is None:
        return None, True
    if not categories:
        return None, True
    # Literal values take precedence over keywords
    if drop in categories:
        return drop, True
    if drop == "first":
        return categories[0], True
    if drop == "last":
        return categories[-1], True
    # Value not in categories and not a keyword
    result.add_error(
        step, op_name, f"Drop value '{drop}' not in categories list"
    )
    return None, False


def _validate_enc_onehot(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    categories = params.get("categories")
    prefix = params["prefix"]
    drop = params.get("drop")
    drop_original = params["drop_original"]

    if not _check_column_exists(tracker, column, result, step, "enc_onehot"):
        return

    if categories is not None:
        # Check for duplicate categories
        if len(categories) != len(set(categories)):
            result.add_error(step, "enc_onehot", "Duplicate values in categories list")
            return

        # Determine which category to drop (if any)
        drop_category, is_valid = _resolve_drop_category(
            drop, categories, result, step, "enc_onehot"
        )
        if not is_valid:
            return

        # Check for column name collisions and update schema
        for cat in categories:
            if cat == drop_category:
                continue
            new_col = f"{prefix}_{cat}"
            if tracker.has_column(new_col):
                result.add_error(
                    step, "enc_onehot", f"Column '{new_col}' already exists"
                )
                return
            tracker.add_column(new_col, pl.Int64())

    # If categories is None, we can't fully validate the output schema
    # The validator will only check that the source column exists

    if drop_original:
        tracker.drop_column(column)


def _validate_enc_ordinal(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    categories = params.get("categories")
    new_column = params["new_column"]
    drop_original = params["drop_original"]

    if not _check_column_exists(tracker, column, result, step, "enc_ordinal"):
        return

    # Check for duplicate categories
    if categories is not None and len(categories) != len(set(categories)):
        result.add_error(step, "enc_ordinal", "Duplicate values in categories list")
        return

    # Update schema
    if new_column != column:
        tracker.add_column(new_column, pl.Int64())
        if drop_original:
            tracker.drop_column(column)
    else:
        tracker.set_dtype(column, pl.Int64())


def _validate_enc_label(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    categories = params.get("categories")
    new_column = params["new_column"]
    drop_original = params["drop_original"]

    if not _check_column_exists(tracker, column, result, step, "enc_label"):
        return

    # Check for duplicate categories
    if categories is not None and len(categories) != len(set(categories)):
        result.add_error(step, "enc_label", "Duplicate values in categories list")
        return

    # Update schema
    if new_column != column:
        tracker.add_column(new_column, pl.Int64())
        if drop_original:
            tracker.drop_column(column)
    else:
        tracker.set_dtype(column, pl.Int64())


# =============================================================================
# Validator registry
# =============================================================================

_VALIDATORS: dict[str, ValidatorFunc] = {
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
    "math_add": partial(_validate_math_scalar, op_name="math_add"),
    "math_subtract": partial(_validate_math_scalar, op_name="math_subtract"),
    "math_multiply": partial(_validate_math_scalar, op_name="math_multiply"),
    "math_divide": partial(_validate_math_scalar, op_name="math_divide"),
    "math_clamp": partial(_validate_math_scalar, op_name="math_clamp"),
    "math_abs": partial(_validate_math_scalar, op_name="math_abs"),
    "math_round": partial(_validate_math_scalar, op_name="math_round"),
    "math_set_min": partial(_validate_math_scalar, op_name="math_set_min"),
    "math_set_max": partial(_validate_math_scalar, op_name="math_set_max"),
    "math_add_columns": partial(_validate_math_columns, op_name="math_add_columns"),
    "math_subtract_columns": partial(
        _validate_math_columns, op_name="math_subtract_columns"
    ),
    "math_multiply_columns": partial(
        _validate_math_columns, op_name="math_multiply_columns"
    ),
    "math_divide_columns": partial(
        _validate_math_columns, op_name="math_divide_columns"
    ),
    "math_cumsum": _validate_math_cumsum,
    "math_rank": _validate_math_rank,
    "math_percent_of": _validate_math_percent_of,
    # String ops
    "str_replace": partial(_validate_str_op, op_name="str_replace"),
    "str_slice": partial(_validate_str_op, op_name="str_slice"),
    "str_truncate": partial(_validate_str_op, op_name="str_truncate"),
    "str_lower": partial(_validate_str_op, op_name="str_lower"),
    "str_upper": partial(_validate_str_op, op_name="str_upper"),
    "str_strip": partial(_validate_str_op, op_name="str_strip"),
    "str_pad": partial(_validate_str_op, op_name="str_pad"),
    "str_split": _validate_str_split,
    "str_concat": _validate_str_concat,
    "str_extract": _validate_str_extract,
    # Datetime ops
    "dt_year": partial(_validate_dt_op, op_name="dt_year"),
    "dt_month": partial(_validate_dt_op, op_name="dt_month"),
    "dt_day": partial(_validate_dt_op, op_name="dt_day"),
    "dt_week": partial(_validate_dt_op, op_name="dt_week"),
    "dt_quarter": partial(_validate_dt_op, op_name="dt_quarter"),
    "dt_year_month": partial(
        _validate_dt_op, op_name="dt_year_month", output_dtype=pl.Utf8()
    ),
    "dt_quarter_year": partial(
        _validate_dt_op, op_name="dt_quarter_year", output_dtype=pl.Utf8()
    ),
    "dt_calendar_week": partial(
        _validate_dt_op, op_name="dt_calendar_week", output_dtype=pl.Utf8()
    ),
    "dt_truncate": partial(
        _validate_dt_op, op_name="dt_truncate", output_dtype=pl.Date()
    ),
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
    # Encoding ops
    "enc_onehot": _validate_enc_onehot,
    "enc_ordinal": _validate_enc_ordinal,
    "enc_label": _validate_enc_label,
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


def dry_run_schema(
    operations: list[tuple[Any, dict[str, Any]]], schema: dict[str, pl.DataType]
) -> DryRunResult:
    """Perform a dry run showing what each operation will do.

    Args:
        operations: List of (method, params) tuples from TransformPlan.
        schema: Initial DataFrame schema.

    Returns:
        DryRunResult with step-by-step preview and validation.
    """
    validation_result = ValidationResult()
    tracker = SchemaTracker(schema)
    steps: list[DryRunStep] = []

    for step_num, (method, params) in enumerate(operations, start=1):
        op_name = method.__name__.lstrip("_")

        # Capture schema before
        schema_before = {k: dtype_name(v) for k, v in tracker._schema.items()}
        cols_before = set(tracker._schema.keys())

        # Run validation (which also updates tracker)
        step_errors_before = len(validation_result.errors)
        validator = _VALIDATORS.get(op_name)
        if validator:
            validator(tracker, params, validation_result, step_num)

        # Capture schema after
        schema_after = {k: dtype_name(v) for k, v in tracker._schema.items()}
        cols_after = set(tracker._schema.keys())

        # Calculate changes
        columns_added = list(cols_after - cols_before)
        columns_removed = list(cols_before - cols_after)

        # Detect type modifications (columns that exist in both but changed type)
        columns_modified = [
            col
            for col in cols_before & cols_after
            if schema_before.get(col) != schema_after.get(col)
        ]

        # Check if this step had an error
        step_error = None
        if len(validation_result.errors) > step_errors_before:
            step_error = str(validation_result.errors[-1].message)

        steps.append(
            DryRunStep(
                step=step_num,
                operation=op_name,
                params=params,
                schema_before=schema_before,
                schema_after=schema_after,
                columns_added=columns_added,
                columns_removed=columns_removed,
                columns_modified=columns_modified,
                error=step_error,
            )
        )

    return DryRunResult(
        input_schema=schema,
        steps=steps,
        validation=validation_result,
    )
