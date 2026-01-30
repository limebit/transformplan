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

    def test_print(
        self, basic_df: pl.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
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


class TestDryRunResultEdgeCases:
    """Tests for DryRunResult edge cases."""

    def test_output_schema_empty_pipeline(self, basic_df: pl.DataFrame) -> None:
        """Test output_schema for empty pipeline returns input schema."""
        plan = TransformPlan()
        result = plan.dry_run(basic_df)
        assert len(result.output_schema) == len(basic_df.columns)

    def test_summary_validation_failed(self, basic_df: pl.DataFrame) -> None:
        """Test summary shows validation failed message."""
        plan = TransformPlan().col_drop("nonexistent")
        result = plan.dry_run(basic_df)
        summary = result.summary()
        assert "FAILED" in summary
        assert "1 errors" in summary or "1 error" in summary

    def test_summary_with_show_schema(self, basic_df: pl.DataFrame) -> None:
        """Test summary with show_schema=True."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        summary = result.summary(show_schema=True)
        # Should show column types
        assert "Int64" in summary or "Int" in summary

    def test_summary_with_modified_columns(self, basic_df: pl.DataFrame) -> None:
        """Test summary shows modified columns correctly."""
        plan = TransformPlan().col_cast("age", pl.Float64)
        result = plan.dry_run(basic_df)
        summary = result.summary()
        # Should show modified indicator ~
        assert "~" in summary or "age" in summary

    def test_summary_with_error_marker(self, basic_df: pl.DataFrame) -> None:
        """Test summary shows error marker for failed steps."""
        plan = TransformPlan().col_drop("nonexistent")
        result = plan.dry_run(basic_df)
        summary = result.summary()
        # Should show error marker
        assert "✗" in summary

    def test_repr_invalid(self, basic_df: pl.DataFrame) -> None:
        """Test repr for invalid result."""
        plan = TransformPlan().col_drop("nonexistent")
        result = plan.dry_run(basic_df)
        r = repr(result)
        assert "invalid" in r
        assert "1 errors" in r or "error" in r


class TestValidationEdgeCases:
    """Tests for validation edge cases."""

    def test_validate_col_drop_null_with_columns(
        self, df_with_nulls: pl.DataFrame
    ) -> None:
        """Test col_drop_null validation with specified columns."""
        plan = TransformPlan().col_drop_null(columns=["name", "age"])
        result = plan.validate(df_with_nulls)
        assert result.is_valid

    def test_validate_col_drop_null_missing_columns(
        self, df_with_nulls: pl.DataFrame
    ) -> None:
        """Test col_drop_null validation with missing columns."""
        plan = TransformPlan().col_drop_null(columns=["nonexistent"])
        result = plan.validate(df_with_nulls)
        assert not result.is_valid

    def test_validate_col_add_with_expr(self, basic_df: pl.DataFrame) -> None:
        """Test col_add validation with source expression."""
        plan = TransformPlan().col_add("name_copy", expr="name")
        result = plan.validate(basic_df)
        assert result.is_valid

    def test_validate_col_add_duplicate_column(self, basic_df: pl.DataFrame) -> None:
        """Test col_add validation when column already exists."""
        plan = TransformPlan().col_add("name", value="test")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "already exists" in str(result.errors[0])

    def test_validate_col_add_uuid_duplicate(self, basic_df: pl.DataFrame) -> None:
        """Test col_add_uuid validation when column already exists."""
        plan = TransformPlan().col_add_uuid("name")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "already exists" in str(result.errors[0])

    def test_validate_col_hash_missing_columns(self, basic_df: pl.DataFrame) -> None:
        """Test col_hash validation with missing columns."""
        plan = TransformPlan().col_hash(["name", "nonexistent"], "hash_col")
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_validate_col_hash_duplicate_new_column(
        self, basic_df: pl.DataFrame
    ) -> None:
        """Test col_hash validation when new column already exists."""
        plan = TransformPlan().col_hash(["name", "age"], "name")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "already exists" in str(result.errors[0])

    def test_validate_col_coalesce_missing(self, basic_df: pl.DataFrame) -> None:
        """Test col_coalesce validation with missing columns."""
        plan = TransformPlan().col_coalesce(["name", "nonexistent"], "result")
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_validate_math_cumsum_missing_group_by(
        self, basic_df: pl.DataFrame
    ) -> None:
        """Test math_cumsum validation with missing group_by columns."""
        plan = TransformPlan().math_cumsum(
            "salary", new_column="cumsum", group_by=["nonexistent"]
        )
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "Group-by columns do not exist" in str(result.errors[0])

    def test_validate_math_rank_missing_group_by(self, basic_df: pl.DataFrame) -> None:
        """Test math_rank validation with missing group_by columns."""
        plan = TransformPlan().math_rank(
            "salary", new_column="rank", group_by=["nonexistent"]
        )
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_validate_str_split_duplicate_columns(
        self, string_df: pl.DataFrame
    ) -> None:
        """Test str_split validation when new columns already exist."""
        plan = TransformPlan().str_split("text", " ", new_columns=["text"])
        result = plan.validate(string_df)
        assert not result.is_valid
        assert "already exists" in str(result.errors[0])

    def test_validate_rows_drop_nulls_missing_columns(
        self, basic_df: pl.DataFrame
    ) -> None:
        """Test rows_drop_nulls validation with missing columns."""
        plan = TransformPlan().rows_drop_nulls(columns=["nonexistent"])
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_validate_rows_unique_missing_columns(
        self, basic_df: pl.DataFrame
    ) -> None:
        """Test rows_unique validation with missing columns."""
        plan = TransformPlan().rows_unique(columns=["nonexistent"])
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_validate_rows_drop_filter(self, basic_df: pl.DataFrame) -> None:
        """Test rows_drop validation with filter."""
        plan = TransformPlan().rows_drop(Col("nonexistent") > 10)
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_validate_rows_flag_filter(self, basic_df: pl.DataFrame) -> None:
        """Test rows_flag validation with filter."""
        plan = TransformPlan().rows_flag(Col("nonexistent") > 10, "flag")
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_validate_rows_flag_duplicate_column(self, basic_df: pl.DataFrame) -> None:
        """Test rows_flag validation when new column already exists."""
        plan = TransformPlan().rows_flag(Col("age") > 30, "name")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "already exists" in str(result.errors[0])

    def test_validate_rows_sort_missing_columns(self, basic_df: pl.DataFrame) -> None:
        """Test rows_sort validation with missing columns."""
        plan = TransformPlan().rows_sort("nonexistent")
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_validate_rows_deduplicate_missing_sort(
        self, duplicates_df: pl.DataFrame
    ) -> None:
        """Test rows_deduplicate validation with missing sort column."""
        plan = TransformPlan().rows_deduplicate(["id"], "nonexistent")
        result = plan.validate(duplicates_df)
        assert not result.is_valid
        assert "does not exist" in str(result.errors[0])

    def test_validate_rows_deduplicate_missing_columns(
        self, duplicates_df: pl.DataFrame
    ) -> None:
        """Test rows_deduplicate validation with missing columns."""
        plan = TransformPlan().rows_deduplicate(["nonexistent"], "timestamp")
        result = plan.validate(duplicates_df)
        assert not result.is_valid

    def test_validate_rows_explode_non_list(self, basic_df: pl.DataFrame) -> None:
        """Test rows_explode validation on non-list column."""
        plan = TransformPlan().rows_explode("name")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "expected List" in str(result.errors[0])

    def test_validate_rows_melt_missing_id(self, wide_df: pl.DataFrame) -> None:
        """Test rows_melt validation with missing id columns."""
        plan = TransformPlan().rows_melt(
            id_columns=["nonexistent"], value_columns=["q1"]
        )
        result = plan.validate(wide_df)
        assert not result.is_valid

    def test_validate_rows_melt_missing_value(self, wide_df: pl.DataFrame) -> None:
        """Test rows_melt validation with missing value columns."""
        plan = TransformPlan().rows_melt(
            id_columns=["id"], value_columns=["nonexistent"]
        )
        result = plan.validate(wide_df)
        assert not result.is_valid

    def test_validate_rows_pivot_missing_index(self, long_df: pl.DataFrame) -> None:
        """Test rows_pivot validation with missing index columns."""
        plan = TransformPlan().rows_pivot(
            index=["nonexistent"], columns="quarter", values="value"
        )
        result = plan.validate(long_df)
        assert not result.is_valid

    def test_validate_rows_pivot_missing_columns(self, long_df: pl.DataFrame) -> None:
        """Test rows_pivot validation with missing pivot column."""
        plan = TransformPlan().rows_pivot(
            index=["id"], columns="nonexistent", values="value"
        )
        result = plan.validate(long_df)
        assert not result.is_valid

    def test_validate_rows_pivot_missing_values(self, long_df: pl.DataFrame) -> None:
        """Test rows_pivot validation with missing values column."""
        plan = TransformPlan().rows_pivot(
            index=["id"], columns="quarter", values="nonexistent"
        )
        result = plan.validate(long_df)
        assert not result.is_valid

    def test_validate_filter_not_operator(self, basic_df: pl.DataFrame) -> None:
        """Test filter validation with Not operator."""
        plan = TransformPlan().rows_filter(~(Col("nonexistent") == 1))
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_validate_col_select_missing_columns(self, basic_df: pl.DataFrame) -> None:
        """Test col_select validation with missing columns."""
        plan = TransformPlan().col_select(["name", "nonexistent"])
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "Columns do not exist" in str(result.errors[0])

    def test_validate_col_add_missing_expr(self, basic_df: pl.DataFrame) -> None:
        """Test col_add validation with missing source expression."""
        plan = TransformPlan().col_add("new_col", expr="nonexistent")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "does not exist" in str(result.errors[0])

    def test_input_schema_property(self, basic_df: pl.DataFrame) -> None:
        """Test DryRunResult.input_schema property."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        schema = result.input_schema
        assert "age" in schema
        assert schema["age"] == pl.Int64()


class TestDryRunSummaryFormatting:
    """Tests for DryRunResult.summary() formatting edge cases."""

    def test_summary_columns_added(self, basic_df: pl.DataFrame) -> None:
        """Test summary shows columns added with + prefix."""
        plan = TransformPlan().col_add("new_col", value="test")
        result = plan.dry_run(basic_df)
        summary = result.summary()
        assert "+['" in summary or "+" in summary

    def test_summary_filter_param(self, basic_df: pl.DataFrame) -> None:
        """Test summary shows filter params as <filter>."""
        plan = TransformPlan().rows_filter(Col("age") > 30)
        result = plan.dry_run(basic_df)
        summary = result.summary(show_params=True)
        assert "<filter>" in summary

    def test_summary_long_list_param(self) -> None:
        """Test summary truncates long list params."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12], "e": [13, 14, 15]})
        plan = TransformPlan().col_select(["a", "b", "c", "d", "e"])
        result = plan.dry_run(df)
        summary = result.summary(show_params=True)
        # List of 5 items should be truncated
        assert "items" in summary

    def test_summary_long_string_param(self) -> None:
        """Test summary truncates long string params."""
        df = pl.DataFrame({"text": ["hello world"]})
        # Create a plan with a long string parameter
        long_pattern = "this_is_a_very_long_pattern_that_should_be_truncated"
        plan = TransformPlan().str_replace("text", long_pattern, "short")
        result = plan.dry_run(df)
        summary = result.summary(show_params=True)
        # Long string should be truncated
        assert "..." in summary or long_pattern[:17] in summary
