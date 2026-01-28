"""Integration tests for end-to-end TransformPlan usage."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from transformplan import Col, TransformPlan
from transformplan.validation import SchemaValidationError


class TestFullPipeline:
    """Tests for full pipeline execution."""

    def test_complex_pipeline(self, basic_df: pl.DataFrame) -> None:
        """Test a complex multi-step pipeline."""
        plan = (
            TransformPlan()
            .col_rename("name", "full_name")
            .col_add("bonus", value=1000)  # Use int to match numeric type
            .col_cast("bonus", pl.Float64)  # Cast to float for math operation
            .math_add_columns("salary", "bonus", "total_comp")
            .rows_filter(Col("age") >= 30)
            .rows_sort("total_comp", descending=True)
            .col_select(["id", "full_name", "total_comp"])
        )

        result, protocol = plan.process(basic_df)

        # Verify result
        assert result.columns == ["id", "full_name", "total_comp"]
        # ages 25,30,35,40,45 - filtering >= 30 leaves 4 rows
        assert len(result) == 4
        assert all(tc > 50000 for tc in result["total_comp"].to_list())

        # Verify protocol
        assert len(protocol) == 7
        assert protocol.input_hash != protocol.output_hash

    def test_data_cleaning_pipeline(self, df_with_nulls: pl.DataFrame) -> None:
        """Test a data cleaning pipeline."""
        plan = (
            TransformPlan()
            .col_fill_null("name", value="Unknown")
            .col_fill_null("age", strategy="forward")
            .rows_drop_nulls("salary")
            .col_add("status", value="cleaned")
        )

        result, _ = plan.process(df_with_nulls)

        # Verify cleaning
        assert result["name"].null_count() == 0
        assert "status" in result.columns

    def test_aggregation_pipeline(self, numeric_df: pl.DataFrame) -> None:
        """Test an aggregation pipeline."""
        plan = (
            TransformPlan()
            .math_add_columns("a", "b", "sum_ab")
            .math_multiply_columns("a", "c", "product_ac")
            .math_cumsum("a", "cumsum_a")
            .math_rank("sum_ab", "rank")
        )

        result, _ = plan.process(numeric_df)

        assert "sum_ab" in result.columns
        assert "product_ac" in result.columns
        assert "cumsum_a" in result.columns
        assert "rank" in result.columns


class TestValidationThenExecution:
    """Tests for validation-then-execution workflow."""

    def test_validate_before_process(self, basic_df: pl.DataFrame) -> None:
        """Test validating before processing."""
        plan = TransformPlan().col_drop("age").math_multiply("salary", 1.1)

        validation = plan.validate(basic_df)
        assert validation.is_valid

        result, _ = plan.process(basic_df)
        assert "age" not in result.columns

    def test_dry_run_before_process(self, basic_df: pl.DataFrame) -> None:
        """Test dry run before processing."""
        plan = (
            TransformPlan()
            .col_drop("age")
            .col_add("new_col", value="test")
            .math_multiply("salary", 1.1)
        )

        dry_run = plan.dry_run(basic_df)
        assert dry_run.is_valid
        assert "age" not in dry_run.output_columns
        assert "new_col" in dry_run.output_columns

        # Now execute
        result, _ = plan.process(basic_df)
        assert "age" not in result.columns
        assert "new_col" in result.columns


class TestSerializationRoundtrip:
    """Tests for serialization roundtrip."""

    def test_json_roundtrip_produces_same_result(self, basic_df: pl.DataFrame) -> None:
        """Test that JSON roundtrip produces identical results."""
        original_plan = (
            TransformPlan()
            .col_drop("active")
            .col_rename("name", "full_name")
            .math_multiply("salary", 1.1)
            .rows_filter(Col("age") >= 30)
        )

        json_str = original_plan.to_json()
        restored_plan = TransformPlan.from_json(json_str)

        original_result, _ = original_plan.process(basic_df)
        restored_result, _ = restored_plan.process(basic_df)

        assert original_result.equals(restored_result)

    def test_file_roundtrip(self, basic_df: pl.DataFrame) -> None:
        """Test saving and loading from file."""
        plan = TransformPlan().col_drop("age").math_multiply("salary", 2)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            plan.to_json(path)
            restored = TransformPlan.from_json(path)

            original_result, _ = plan.process(basic_df)
            restored_result, _ = restored.process(basic_df)

            assert original_result.equals(restored_result)
        finally:
            path.unlink()

    def test_protocol_roundtrip(self, basic_df: pl.DataFrame) -> None:
        """Test protocol serialization roundtrip."""
        plan = TransformPlan().col_drop("age")
        _, original_protocol = plan.process(basic_df)
        original_protocol.set_metadata(author="test", version="1.0")

        from transformplan.protocol import Protocol

        json_str = original_protocol.to_json()
        restored_protocol = Protocol.from_json(json_str)

        assert restored_protocol.input_hash == original_protocol.input_hash
        assert restored_protocol.output_hash == original_protocol.output_hash
        assert restored_protocol.metadata["author"] == "test"


class TestErrorHandlingInPipeline:
    """Tests for error handling in pipelines."""

    def test_validation_error_stops_execution(self, basic_df: pl.DataFrame) -> None:
        """Test that validation error prevents execution."""
        plan = (
            TransformPlan()
            .col_drop("age")
            .col_drop("nonexistent")  # This should fail validation
            .math_multiply("salary", 2)
        )

        with pytest.raises(SchemaValidationError):
            plan.process(basic_df)

    def test_validation_detects_chained_errors(self, basic_df: pl.DataFrame) -> None:
        """Test that validation catches errors in chained operations."""
        plan = (
            TransformPlan()
            .col_rename("name", "full_name")
            .col_drop("name")  # name doesn't exist anymore
        )

        validation = plan.validate(basic_df)
        assert not validation.is_valid

    def test_can_skip_validation_for_performance(self, basic_df: pl.DataFrame) -> None:
        """Test that validation can be skipped."""
        plan = TransformPlan().col_drop("age")

        # With validation (default)
        result1, _ = plan.process(basic_df)

        # Without validation
        result2, _ = plan.process(basic_df, validate=False)

        assert result1.equals(result2)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self, empty_df: pl.DataFrame) -> None:
        """Test pipeline on empty DataFrame."""
        plan = TransformPlan().col_add("new_col", value="test")

        result, protocol = plan.process(empty_df)
        assert "new_col" in result.columns
        assert len(result) == 0
        assert len(protocol) == 1

    def test_single_row_dataframe(self, single_row_df: pl.DataFrame) -> None:
        """Test pipeline on single-row DataFrame."""
        plan = TransformPlan().math_multiply("value", 2).col_add("flag", value=True)

        result, _ = plan.process(single_row_df)
        assert result["value"][0] == 200.0
        assert result["flag"][0] is True

    def test_empty_plan(self, basic_df: pl.DataFrame) -> None:
        """Test that empty plan returns unchanged DataFrame."""
        plan = TransformPlan()

        result, protocol = plan.process(basic_df)
        assert result.equals(basic_df)
        assert len(protocol) == 0

    def test_many_operations(self, basic_df: pl.DataFrame) -> None:
        """Test pipeline with many operations."""
        plan = TransformPlan()
        for i in range(20):
            plan = plan.col_add(f"col_{i}", value=i)

        result, protocol = plan.process(basic_df)
        assert len(result.columns) == len(basic_df.columns) + 20
        assert len(protocol) == 20


class TestReproducibility:
    """Tests for reproducibility."""

    def test_same_input_same_output_hash(self, basic_df: pl.DataFrame) -> None:
        """Test that same input produces same output hash."""
        plan = TransformPlan().col_drop("age").math_multiply("salary", 1.1)

        _, protocol1 = plan.process(basic_df)
        _, protocol2 = plan.process(basic_df)

        assert protocol1.input_hash == protocol2.input_hash
        assert protocol1.output_hash == protocol2.output_hash

    def test_restored_plan_produces_same_hash(self, basic_df: pl.DataFrame) -> None:
        """Test that restored plan produces same output hash."""
        original_plan = TransformPlan().col_drop("age").math_multiply("salary", 1.1)

        _, original_protocol = original_plan.process(basic_df)

        restored_plan = TransformPlan.from_json(original_plan.to_json())
        _, restored_protocol = restored_plan.process(basic_df)

        assert original_protocol.output_hash == restored_protocol.output_hash


class TestComplexFilters:
    """Tests for complex filter expressions."""

    def test_nested_and_or_filters(self, basic_df: pl.DataFrame) -> None:
        """Test nested AND/OR filter expressions."""
        plan = TransformPlan().rows_filter(
            ((Col("age") >= 30) & (Col("age") <= 40)) | (Col("name") == "Alice")
        )

        result, _ = plan.process(basic_df)
        # Should include: Alice (age 25), Bob (age 30), Charlie (age 35), David (age 40)
        assert len(result) == 4

    def test_complex_filter_serialization(self, basic_df: pl.DataFrame) -> None:
        """Test that complex filters survive serialization."""
        plan = TransformPlan().rows_filter(
            ((Col("age") >= 30) & (Col("active") == True)) | (Col("salary") > 80000)  # noqa: E712
        )

        json_str = plan.to_json()
        restored = TransformPlan.from_json(json_str)

        original_result, _ = plan.process(basic_df)
        restored_result, _ = restored.process(basic_df)

        assert original_result.equals(restored_result)


class TestProtocolAuditTrail:
    """Tests for protocol audit trail."""

    def test_protocol_captures_all_steps(self, basic_df: pl.DataFrame) -> None:
        """Test that protocol captures all steps."""
        plan = (
            TransformPlan()
            .col_drop("active")
            .col_rename("name", "full_name")
            .math_add("age", 1)
        )

        _, protocol = plan.process(basic_df)
        d = protocol.to_dict()

        assert len(d["steps"]) == 3
        ops = [s["operation"] for s in d["steps"]]
        assert ops == ["col_drop", "col_rename", "math_add"]

    def test_protocol_tracks_shape_changes(self, basic_df: pl.DataFrame) -> None:
        """Test that protocol tracks shape changes."""
        plan = TransformPlan().rows_filter(Col("age") >= 30)

        _, protocol = plan.process(basic_df)
        d = protocol.to_dict()

        step = d["steps"][0]
        assert step["rows_changed"] > 0  # Some rows were filtered

    def test_protocol_metadata_preserved(self, basic_df: pl.DataFrame) -> None:
        """Test that protocol metadata is preserved."""
        plan = TransformPlan().col_drop("age")

        _, protocol = plan.process(basic_df)
        protocol.set_metadata(
            author="test_user",
            project="test_project",
            version="1.0.0",
        )

        from transformplan.protocol import Protocol

        restored = Protocol.from_json(protocol.to_json())

        assert restored.metadata["author"] == "test_user"
        assert restored.metadata["project"] == "test_project"
        assert restored.metadata["version"] == "1.0.0"
