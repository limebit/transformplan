"""Tests for validation (validation.py)."""

import polars as pl
import pytest

from transformplan import Col, TransformPlan
from transformplan.validation import (
    DryRunResult,
    SchemaTracker,
    SchemaValidationError,
    ValidationError,
    ValidationResult,
    is_boolean,
    is_datetime,
    is_numeric,
    is_string,
)


class TestTypeChecking:
    """Tests for type checking functions."""

    def test_is_numeric(self) -> None:
        """Test is_numeric function."""
        assert is_numeric(pl.Int64()) is True
        assert is_numeric(pl.Float64()) is True
        assert is_numeric(pl.Int8()) is True
        assert is_numeric(pl.UInt32()) is True
        assert is_numeric(pl.Utf8()) is False
        assert is_numeric(pl.Boolean()) is False

    def test_is_string(self) -> None:
        """Test is_string function."""
        assert is_string(pl.Utf8()) is True
        assert is_string(pl.String()) is True
        assert is_string(pl.Int64()) is False
        assert is_string(pl.Boolean()) is False

    def test_is_datetime(self) -> None:
        """Test is_datetime function."""
        assert is_datetime(pl.Date()) is True
        assert is_datetime(pl.Datetime()) is True
        assert is_datetime(pl.Time()) is True
        assert is_datetime(pl.Duration()) is True
        assert is_datetime(pl.Int64()) is False
        assert is_datetime(pl.Utf8()) is False

    def test_is_boolean(self) -> None:
        """Test is_boolean function."""
        assert is_boolean(pl.Boolean()) is True
        assert is_boolean(pl.Int64()) is False
        assert is_boolean(pl.Utf8()) is False


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_empty_result_is_valid(self) -> None:
        """Test that empty result is valid."""
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_result_with_error_is_invalid(self) -> None:
        """Test that result with error is invalid."""
        result = ValidationResult()
        result.add_error(1, "col_drop", "Column does not exist")
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_add_multiple_errors(self) -> None:
        """Test adding multiple errors."""
        result = ValidationResult()
        result.add_error(1, "op1", "Error 1")
        result.add_error(2, "op2", "Error 2")
        assert len(result.errors) == 2

    def test_raise_if_invalid(self) -> None:
        """Test raise_if_invalid method."""
        result = ValidationResult()
        result.add_error(1, "col_drop", "Column does not exist")
        with pytest.raises(SchemaValidationError):
            result.raise_if_invalid()

    def test_raise_if_invalid_no_error(self) -> None:
        """Test that raise_if_invalid does nothing when valid."""
        result = ValidationResult()
        result.raise_if_invalid()  # Should not raise

    def test_repr_valid(self) -> None:
        """Test repr for valid result."""
        result = ValidationResult()
        assert "valid=True" in repr(result)

    def test_repr_invalid(self) -> None:
        """Test repr for invalid result."""
        result = ValidationResult()
        result.add_error(1, "op", "msg")
        assert "valid=False" in repr(result)


class TestValidationError:
    """Tests for ValidationError class."""

    def test_str(self) -> None:
        """Test string representation."""
        error = ValidationError(1, "col_drop", "Column 'x' does not exist")
        s = str(error)
        assert "Step 1" in s
        assert "col_drop" in s
        assert "Column 'x' does not exist" in s


class TestSchemaTracker:
    """Tests for SchemaTracker class."""

    def test_has_column(self) -> None:
        """Test has_column method."""
        tracker = SchemaTracker({"a": pl.Int64(), "b": pl.Utf8()})
        assert tracker.has_column("a") is True
        assert tracker.has_column("c") is False

    def test_get_dtype(self) -> None:
        """Test get_dtype method."""
        tracker = SchemaTracker({"a": pl.Int64(), "b": pl.Utf8()})
        assert tracker.get_dtype("a") == pl.Int64()
        assert tracker.get_dtype("c") is None

    def test_drop_column(self) -> None:
        """Test drop_column method."""
        tracker = SchemaTracker({"a": pl.Int64(), "b": pl.Utf8()})
        tracker.drop_column("a")
        assert tracker.has_column("a") is False
        assert tracker.has_column("b") is True

    def test_add_column(self) -> None:
        """Test add_column method."""
        tracker = SchemaTracker({"a": pl.Int64()})
        tracker.add_column("b", pl.Utf8())
        assert tracker.has_column("b") is True
        assert tracker.get_dtype("b") == pl.Utf8()

    def test_rename_column(self) -> None:
        """Test rename_column method."""
        tracker = SchemaTracker({"a": pl.Int64()})
        tracker.rename_column("a", "b")
        assert tracker.has_column("a") is False
        assert tracker.has_column("b") is True
        assert tracker.get_dtype("b") == pl.Int64()

    def test_set_dtype(self) -> None:
        """Test set_dtype method."""
        tracker = SchemaTracker({"a": pl.Int64()})
        tracker.set_dtype("a", pl.Float64())
        assert tracker.get_dtype("a") == pl.Float64()

    def test_set_columns(self) -> None:
        """Test set_columns method."""
        tracker = SchemaTracker({"a": pl.Int64(), "b": pl.Utf8(), "c": pl.Boolean()})
        tracker.set_columns(["b", "a"])
        assert tracker.has_column("a") is True
        assert tracker.has_column("b") is True
        assert tracker.has_column("c") is False

    def test_columns_property(self) -> None:
        """Test columns property."""
        tracker = SchemaTracker({"a": pl.Int64(), "b": pl.Utf8()})
        assert tracker.columns == {"a", "b"}


class TestValidateSchema:
    """Tests for validate_schema function."""

    def test_validate_col_drop(self, basic_df: pl.DataFrame) -> None:
        """Test validation of col_drop."""
        plan = TransformPlan().col_drop("age")
        result = plan.validate(basic_df)
        assert result.is_valid

    def test_validate_col_drop_nonexistent(self, basic_df: pl.DataFrame) -> None:
        """Test validation catches nonexistent column."""
        plan = TransformPlan().col_drop("nonexistent")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "does not exist" in str(result.errors[0])

    def test_validate_math_on_non_numeric(self, basic_df: pl.DataFrame) -> None:
        """Test validation catches math on non-numeric column."""
        plan = TransformPlan().math_add("name", 10)
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "expected numeric" in str(result.errors[0])

    def test_validate_string_on_non_string(self, basic_df: pl.DataFrame) -> None:
        """Test validation catches string op on non-string column."""
        plan = TransformPlan().str_lower("age")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "expected string" in str(result.errors[0])

    def test_validate_datetime_on_non_datetime(self, basic_df: pl.DataFrame) -> None:
        """Test validation catches datetime op on non-datetime column."""
        plan = TransformPlan().dt_year("name")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "expected date/datetime" in str(result.errors[0])

    def test_validate_tracks_schema_changes(self, basic_df: pl.DataFrame) -> None:
        """Test that validation tracks schema changes through pipeline."""
        # This should be valid: rename then use new name
        plan = TransformPlan().col_rename("name", "full_name").col_drop("full_name")
        result = plan.validate(basic_df)
        assert result.is_valid


class TestDryRunSchema:
    """Tests for dry_run_schema function."""

    def test_dry_run_basic(self, basic_df: pl.DataFrame) -> None:
        """Test basic dry run."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        assert isinstance(result, DryRunResult)
        assert result.is_valid
        assert len(result.steps) == 1

    def test_dry_run_invalid(self, basic_df: pl.DataFrame) -> None:
        """Test dry run with invalid plan."""
        plan = TransformPlan().col_drop("nonexistent")
        result = plan.dry_run(basic_df)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_dry_run_shows_columns_added(self, basic_df: pl.DataFrame) -> None:
        """Test that dry run shows added columns."""
        plan = TransformPlan().col_add("new_col", value="x")
        result = plan.dry_run(basic_df)
        assert "new_col" in result.steps[0].columns_added

    def test_dry_run_shows_columns_removed(self, basic_df: pl.DataFrame) -> None:
        """Test that dry run shows removed columns."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        assert "age" in result.steps[0].columns_removed

    def test_dry_run_shows_columns_modified(self, basic_df: pl.DataFrame) -> None:
        """Test that dry run shows modified columns."""
        plan = TransformPlan().col_cast("age", pl.Float64)
        result = plan.dry_run(basic_df)
        assert "age" in result.steps[0].columns_modified

    def test_dry_run_output_schema(self, basic_df: pl.DataFrame) -> None:
        """Test dry run output schema prediction."""
        plan = TransformPlan().col_drop("age").col_add("new_col", value="x")
        result = plan.dry_run(basic_df)
        assert "age" not in result.output_columns
        assert "new_col" in result.output_columns


class TestDryRunResult:
    """Tests for DryRunResult class."""

    def test_is_valid_property(self, basic_df: pl.DataFrame) -> None:
        """Test is_valid property."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        assert result.is_valid is True

    def test_errors_property(self, basic_df: pl.DataFrame) -> None:
        """Test errors property."""
        plan = TransformPlan().col_drop("nonexistent")
        result = plan.dry_run(basic_df)
        assert len(result.errors) > 0

    def test_steps_property(self, basic_df: pl.DataFrame) -> None:
        """Test steps property."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        assert len(result.steps) == 1

    def test_input_schema_property(self, basic_df: pl.DataFrame) -> None:
        """Test input_schema property."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        assert "age" in result.input_columns

    def test_output_schema_property(self, basic_df: pl.DataFrame) -> None:
        """Test output_schema property."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        assert "age" not in result.output_columns

    def test_summary(self, basic_df: pl.DataFrame) -> None:
        """Test summary method."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        summary = result.summary()
        assert "DRY RUN PREVIEW" in summary

    def test_print(self, basic_df: pl.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        """Test print method."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        result.print()
        captured = capsys.readouterr()
        assert "DRY RUN PREVIEW" in captured.out

    def test_repr(self, basic_df: pl.DataFrame) -> None:
        """Test repr."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        assert "DryRunResult" in repr(result)


class TestSchemaValidationError:
    """Tests for SchemaValidationError exception."""

    def test_exception_message(self, basic_df: pl.DataFrame) -> None:
        """Test exception message contains error details."""
        plan = TransformPlan().col_drop("nonexistent")
        try:
            plan.process(basic_df)
            pytest.fail("Should have raised SchemaValidationError")
        except SchemaValidationError as e:
            assert "validation failed" in str(e).lower()
            assert "nonexistent" in str(e) or "does not exist" in str(e)


class TestFilterValidation:
    """Tests for filter expression validation."""

    def test_validate_filter_column_exists(self, basic_df: pl.DataFrame) -> None:
        """Test filter validation checks column existence."""
        plan = TransformPlan().rows_filter(Col("nonexistent") > 10)
        result = plan.validate(basic_df)
        assert not result.is_valid
        # Check that the error message indicates the column doesn't exist
        assert "nonexistent" in str(result.errors[0])

    def test_validate_filter_numeric_comparison(self, basic_df: pl.DataFrame) -> None:
        """Test filter validation checks numeric types for comparison."""
        plan = TransformPlan().rows_filter(Col("name") > 10)
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "cannot use numeric comparison" in str(result.errors[0])

    def test_validate_filter_string_operation(self, basic_df: pl.DataFrame) -> None:
        """Test filter validation checks string types for string ops."""
        plan = TransformPlan().rows_filter(Col("age").str_contains("foo"))
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "cannot use string filter" in str(result.errors[0])
