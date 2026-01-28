"""Schema validation for TransformPlan operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from .filters import Filter


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

    def set_columns(self, columns: list[str]) -> None:
        """Keep only the specified columns in order."""
        self._schema = {col: self._schema[col] for col in columns if col in self._schema}


# =============================================================================
# Validators for each operation type
# =============================================================================


def _validate_col_drop(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    column = params["column"]
    if not tracker.has_column(column):
        result.add_error(step, "col_drop", f"Column '{column}' does not exist")
    else:
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
    if not tracker.has_column(column):
        result.add_error(step, "col_cast", f"Column '{column}' does not exist")


def _validate_col_reorder(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    columns = params["columns"]
    missing = [col for col in columns if not tracker.has_column(col)]
    if missing:
        result.add_error(step, "col_reorder", f"Columns do not exist: {missing}")
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


def _validate_math_op(
    tracker: SchemaTracker,
    params: dict[str, Any],
    result: ValidationResult,
    step: int,
    op_name: str,
) -> None:
    column = params["column"]
    if not tracker.has_column(column):
        result.add_error(step, op_name, f"Column '{column}' does not exist")


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
    filter_dict: dict[str, Any], tracker: SchemaTracker
) -> list[str]:
    """Recursively extract and validate columns from a filter dict."""
    missing = []
    filter_type = filter_dict.get("type")

    if filter_type in ("and", "or"):
        missing.extend(_validate_filter_columns(filter_dict["left"], tracker))
        missing.extend(_validate_filter_columns(filter_dict["right"], tracker))
    elif filter_type == "not":
        missing.extend(_validate_filter_columns(filter_dict["operand"], tracker))
    elif "column" in filter_dict:
        column = filter_dict["column"]
        if not tracker.has_column(column):
            missing.append(column)

    return missing


def _validate_rows_filter(
    tracker: SchemaTracker, params: dict[str, Any], result: ValidationResult, step: int
) -> None:
    filter_dict = params.get("filter", {})
    missing = _validate_filter_columns(filter_dict, tracker)
    if missing:
        unique_missing = list(dict.fromkeys(missing))  # preserve order, remove dupes
        result.add_error(step, "rows_filter", f"Columns do not exist: {unique_missing}")


# =============================================================================
# Validator registry
# =============================================================================

_VALIDATORS: dict[str, Any] = {
    "col_drop": _validate_col_drop,
    "col_rename": _validate_col_rename,
    "col_cast": _validate_col_cast,
    "col_reorder": _validate_col_reorder,
    "col_duplicate": _validate_col_duplicate,
    "math_add": lambda t, p, r, s: _validate_math_op(t, p, r, s, "math_add"),
    "math_subtract": lambda t, p, r, s: _validate_math_op(t, p, r, s, "math_subtract"),
    "math_multiply": lambda t, p, r, s: _validate_math_op(t, p, r, s, "math_multiply"),
    "math_divide": lambda t, p, r, s: _validate_math_op(t, p, r, s, "math_divide"),
    "math_clamp": lambda t, p, r, s: _validate_math_op(t, p, r, s, "math_clamp"),
    "rows_drop_nulls": _validate_rows_drop_nulls,
    "rows_unique": _validate_rows_unique,
    "rows_filter": _validate_rows_filter,
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
