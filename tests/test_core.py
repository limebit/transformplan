"""Tests for core execution logic (core.py)."""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from transformplan import Col, TransformPlan
from transformplan.validation import SchemaValidationError


class TestTransformPlanBasic:
    """Tests for basic TransformPlan functionality."""

    def test_empty_plan(self, basic_df: pl.DataFrame) -> None:
        """Test that empty plan returns unchanged DataFrame."""
        plan = TransformPlan()
        result, protocol = plan.process(basic_df)
        assert result.equals(basic_df)
        assert len(protocol) == 0

    def test_len(self, basic_df: pl.DataFrame) -> None:
        """Test __len__ returns number of operations."""
        plan = TransformPlan().col_drop("age").col_rename("name", "full_name")
        assert len(plan) == 2

    def test_repr(self) -> None:
        """Test __repr__ output."""
        plan = TransformPlan().col_drop("x")
        assert "TransformPlan" in repr(plan)
        assert "1 operations" in repr(plan)


class TestMethodChaining:
    """Tests for method chaining behavior."""

    def test_chaining_returns_self(self, basic_df: pl.DataFrame) -> None:
        """Test that methods return self for chaining."""
        plan = TransformPlan()
        result = plan.col_drop("age")
        assert result is plan

    def test_multiple_operations_execute_in_order(self, basic_df: pl.DataFrame) -> None:
        """Test that operations execute in order."""
        # Rename then drop should work
        plan = TransformPlan().col_rename("name", "full_name").col_drop("full_name")
        result, _ = plan.process(basic_df)
        assert "name" not in result.columns
        assert "full_name" not in result.columns

    def test_order_matters(self, basic_df: pl.DataFrame) -> None:
        """Test that order of operations matters."""
        # Drop then rename should fail validation
        plan = TransformPlan().col_drop("name").col_rename("name", "full_name")
        validation = plan.validate(basic_df)
        assert not validation.is_valid


class TestProcess:
    """Tests for process() method."""

    def test_process_returns_tuple(self, basic_df: pl.DataFrame) -> None:
        """Test that process returns (DataFrame, Protocol) tuple."""
        plan = TransformPlan().col_drop("age")
        result = plan.process(basic_df)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pl.DataFrame)

    def test_process_with_validation(self, basic_df: pl.DataFrame) -> None:
        """Test that process validates by default."""
        plan = TransformPlan().col_drop("nonexistent")
        with pytest.raises(SchemaValidationError):
            plan.process(basic_df)

    def test_process_skip_validation(self, basic_df: pl.DataFrame) -> None:
        """Test that validation can be skipped."""
        plan = TransformPlan().col_drop("age")
        result, _ = plan.process(basic_df, validate=False)
        assert "age" not in result.columns


class TestValidate:
    """Tests for validate() method."""

    def test_validate_valid_plan(self, basic_df: pl.DataFrame) -> None:
        """Test validation of valid plan."""
        plan = TransformPlan().col_drop("age")
        result = plan.validate(basic_df)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_invalid_plan(self, basic_df: pl.DataFrame) -> None:
        """Test validation of invalid plan."""
        plan = TransformPlan().col_drop("nonexistent")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_multiple_errors(self, basic_df: pl.DataFrame) -> None:
        """Test validation catches multiple errors."""
        plan = TransformPlan().col_drop("nonexistent1").col_drop("nonexistent2")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert len(result.errors) == 2


class TestDryRun:
    """Tests for dry_run() method."""

    def test_dry_run_valid_plan(self, basic_df: pl.DataFrame) -> None:
        """Test dry run of valid plan."""
        plan = TransformPlan().col_drop("age").col_rename("name", "full_name")
        result = plan.dry_run(basic_df)
        assert result.is_valid
        assert len(result.steps) == 2

    def test_dry_run_shows_schema_changes(self, basic_df: pl.DataFrame) -> None:
        """Test that dry run shows schema changes."""
        plan = TransformPlan().col_drop("age")
        result = plan.dry_run(basic_df)
        step = result.steps[0]
        assert "age" in step.columns_removed

    def test_dry_run_output_schema(self, basic_df: pl.DataFrame) -> None:
        """Test that dry run predicts output schema."""
        plan = TransformPlan().col_drop("age").col_add("new_col", value="x")
        result = plan.dry_run(basic_df)
        assert "age" not in result.output_columns
        assert "new_col" in result.output_columns


class TestSerialization:
    """Tests for serialization methods."""

    def test_to_dict(self, basic_df: pl.DataFrame) -> None:
        """Test serialization to dict."""
        plan = TransformPlan().col_drop("age").col_rename("name", "full_name")
        d = plan.to_dict()
        assert "version" in d
        assert "steps" in d
        assert len(d["steps"]) == 2

    def test_from_dict(self, basic_df: pl.DataFrame) -> None:
        """Test deserialization from dict."""
        d = {
            "version": "1.0",
            "steps": [
                {"operation": "col_drop", "params": {"column": "age"}},
                {
                    "operation": "col_rename",
                    "params": {"column": "name", "new_name": "full_name"},
                },
            ],
        }
        plan = TransformPlan.from_dict(d)
        assert len(plan) == 2
        result, _ = plan.process(basic_df)
        assert "age" not in result.columns
        assert "full_name" in result.columns

    def test_to_json(self, basic_df: pl.DataFrame) -> None:
        """Test serialization to JSON string."""
        plan = TransformPlan().col_drop("age")
        json_str = plan.to_json()
        assert isinstance(json_str, str)
        d = json.loads(json_str)
        assert "steps" in d

    def test_to_json_file(self, basic_df: pl.DataFrame) -> None:
        """Test serialization to JSON file."""
        plan = TransformPlan().col_drop("age")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            plan.to_json(path)
            assert path.exists()
            content = json.loads(path.read_text())
            assert "steps" in content
        finally:
            path.unlink()

    def test_from_json_string(self, basic_df: pl.DataFrame) -> None:
        """Test deserialization from JSON string."""
        json_str = '{"version": "1.0", "steps": [{"operation": "col_drop", "params": {"column": "age"}}]}'
        plan = TransformPlan.from_json(json_str)
        assert len(plan) == 1

    def test_from_json_file(self, basic_df: pl.DataFrame) -> None:
        """Test deserialization from JSON file."""
        json_str = '{"version": "1.0", "steps": [{"operation": "col_drop", "params": {"column": "age"}}]}'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_str)
            path = Path(f.name)
        try:
            plan = TransformPlan.from_json(path)
            assert len(plan) == 1
        finally:
            path.unlink()

    def test_roundtrip_serialization(self, basic_df: pl.DataFrame) -> None:
        """Test that serialization round trip preserves plan."""
        original_plan = (
            TransformPlan()
            .col_drop("age")
            .col_rename("name", "full_name")
            .rows_filter(Col("id") > 2)
        )
        json_str = original_plan.to_json()
        restored_plan = TransformPlan.from_json(json_str)

        original_result, _ = original_plan.process(basic_df)
        restored_result, _ = restored_plan.process(basic_df)
        assert original_result.equals(restored_result)


class TestPipe:
    """Tests for pipe() composition."""

    def test_pipe_keeps_the_chain(self, basic_df: pl.DataFrame) -> None:
        """Test that pipe applies a function without breaking the chain."""

        def rename_block(
            plan: TransformPlan, source: str, target: str
        ) -> TransformPlan:
            return plan.col_rename(source, target)

        plan = (
            TransformPlan()
            .col_drop("age")
            .pipe(rename_block, "name", "full_name")
            .col_drop("salary")
        )
        result, _ = plan.process(basic_df)
        assert "full_name" in result.columns
        assert len(plan) == 3

    def test_pipe_forwards_keyword_arguments(self, basic_df: pl.DataFrame) -> None:
        """Test that keyword arguments reach the piped function."""

        def drop_it(plan: TransformPlan, *, column: str) -> TransformPlan:
            return plan.col_drop(column)

        plan = TransformPlan().pipe(drop_it, column="age")
        result, _ = plan.process(basic_df)
        assert "age" not in result.columns

    def test_pipe_leaves_protocol_unchanged(self, basic_df: pl.DataFrame) -> None:
        """Test that pipe itself registers no step."""

        def noop(plan: TransformPlan) -> TransformPlan:
            return plan

        plan = TransformPlan().col_drop("age").pipe(noop)
        _, protocol = plan.process(basic_df)
        assert len(protocol) == 1


class TestExtend:
    """Tests for extend() and plan addition."""

    def test_extend_appends_steps(self, basic_df: pl.DataFrame) -> None:
        """Test that extend appends another plan's steps."""
        block = TransformPlan().col_drop("salary")
        plan = TransformPlan().col_drop("age").extend(block)
        result, _ = plan.process(basic_df)
        assert "age" not in result.columns
        assert "salary" not in result.columns
        assert len(plan) == 2

    def test_extend_deep_copies_params(self) -> None:
        """Test that a reused block does not share mutable state."""
        block = TransformPlan().col_fill_null("name", value="X")

        first = TransformPlan().extend(block)
        second = TransformPlan().extend(block)

        first._operations[0][1]["value"] = "CHANGED"

        assert second._operations[0][1]["value"] == "X"
        assert block._operations[0][1]["value"] == "X"

    def test_extend_is_serializable(self, basic_df: pl.DataFrame) -> None:
        """Test that extended steps survive a JSON round trip."""
        block = TransformPlan().col_drop("salary")
        plan = TransformPlan().col_drop("age").extend(block)
        restored = TransformPlan.from_json(plan.to_json())
        assert len(restored) == 2

        expected, _ = plan.process(basic_df)
        actual, _ = restored.process(basic_df)
        assert actual.equals(expected)

    def test_add_leaves_operands_untouched(self) -> None:
        """Test that plan_a + plan_b builds a new plan."""
        left = TransformPlan().col_drop("age")
        right = TransformPlan().col_drop("salary")
        combined = left + right

        assert len(left) == 1
        assert len(right) == 1
        assert len(combined) == 2

    def test_extend_inherits_target_section(self) -> None:
        """Test that a block without sections adopts the target's section."""
        block = TransformPlan().col_drop("salary")
        plan = TransformPlan().section("Cleanup").extend(block)
        assert plan.to_dict()["steps"][0]["section"] == "Cleanup"

    def test_extend_keeps_block_section(self) -> None:
        """Test that a block's own section wins over the target's."""
        block = TransformPlan().section("Block").col_drop("salary")
        plan = TransformPlan().section("Target").extend(block)
        assert plan.to_dict()["steps"][0]["section"] == "Block"


class TestSectionAndReason:
    """Tests for section() and because() metadata."""

    def test_section_labels_following_steps(self) -> None:
        """Test that steps inherit the current section."""
        plan = (
            TransformPlan()
            .section("Filters")
            .col_drop("age")
            .col_drop("salary")
            .section("Types")
            .col_cast("id", "Float64")
        )
        sections = [s.get("section") for s in plan.to_dict()["steps"]]
        assert sections == ["Filters", "Filters", "Types"]

    def test_section_none_ends_the_section(self) -> None:
        """Test that section(None) stops labelling."""
        plan = TransformPlan().section("A").col_drop("age").section(None).col_drop("id")
        steps = plan.to_dict()["steps"]
        assert steps[0]["section"] == "A"
        assert "section" not in steps[1]

    def test_because_annotates_the_last_step(self) -> None:
        """Test that because attaches a reason."""
        plan = TransformPlan().col_drop("age").because("not needed downstream")
        assert plan.to_dict()["steps"][0]["reason"] == "not needed downstream"

    def test_because_annotates_every_step_of_a_sequence(self) -> None:
        """Test that a sequence call gets the reason on all its steps."""
        plan = TransformPlan().col_drop(["age", "salary"]).because("unused")
        reasons = [s.get("reason") for s in plan.to_dict()["steps"]]
        assert reasons == ["unused", "unused"]

    def test_because_without_operation_raises(self) -> None:
        """Test that because before any operation is an error."""
        with pytest.raises(ValueError, match="must follow an operation"):
            TransformPlan().because("nothing to explain")

    def test_metadata_survives_round_trip(self) -> None:
        """Test that section and reason survive a JSON round trip."""
        plan = TransformPlan().section("Cleanup").col_drop("age").because("not needed")
        restored = TransformPlan.from_json(plan.to_json())
        step = restored.to_dict()["steps"][0]
        assert step["section"] == "Cleanup"
        assert step["reason"] == "not needed"

    def test_plain_plan_serializes_without_metadata_keys(self) -> None:
        """Test that plans without metadata are unchanged in JSON."""
        step = TransformPlan().col_drop("age").to_dict()["steps"][0]
        assert set(step) == {"operation", "params"}


class TestToPython:
    """Tests for to_python() method."""

    def test_to_python_basic(self) -> None:
        """Test Python code generation."""
        plan = TransformPlan().col_drop("age").col_rename("name", "full_name")
        code = plan.to_python()
        assert "TransformPlan()" in code
        assert "col_drop" in code
        assert "col_rename" in code

    def test_to_python_custom_variable(self) -> None:
        """Test Python code generation with custom variable name."""
        plan = TransformPlan().col_drop("x")
        code = plan.to_python(variable_name="my_plan")
        assert "my_plan = (" in code

    def test_to_python_is_executable_with_cast(self) -> None:
        """Test that generated code compiles when a dtype is involved."""
        plan = TransformPlan().col_cast("age", pl.Float64)
        code = plan.to_python()
        assert 'dtype="Float64"' in code
        compile(code, "<generated>", "exec")

    def test_to_python_renders_section_and_reason(self) -> None:
        """Test that metadata is reproduced in generated code."""
        plan = TransformPlan().section("Cleanup").col_drop("age").because("not needed")
        code = plan.to_python()
        assert '.section("Cleanup")' in code
        assert '.because("not needed")' in code
        compile(code, "<generated>", "exec")

    def test_to_python_with_filter(self, basic_df: pl.DataFrame) -> None:
        """Test Python code generation with filter."""
        plan = TransformPlan().rows_filter(Col("age") >= 30)
        code = plan.to_python()
        assert "rows_filter" in code
        assert "Col" in code


class TestUnknownOperation:
    """Tests for handling unknown operations."""

    def test_from_dict_unknown_operation(self) -> None:
        """Test that unknown operation raises error."""
        d = {
            "version": "1.0",
            "steps": [{"operation": "unknown_op", "params": {}}],
        }
        with pytest.raises(ValueError, match="Unknown operation"):
            TransformPlan.from_dict(d)


class TestFormatParamsAsPython:
    """Tests for _format_params_as_python() method."""

    def test_format_params_none_value_skipped(self) -> None:
        """Test that None values are skipped in params formatting."""
        plan = TransformPlan()
        # col_add with no expr (None) should skip the None value
        plan.col_add("new_col", value="test")
        code = plan.to_python()
        # The output should not have expr=None in it
        assert "expr=None" not in code
        assert 'new_column="new_col"' in code

    def test_format_params_string_value(self) -> None:
        """Test that string values are formatted with quotes."""
        plan = TransformPlan().col_rename("old", "new")
        code = plan.to_python()
        assert 'column="old"' in code
        assert 'new_name="new"' in code

    def test_format_params_bool_value(self) -> None:
        """Test that boolean values are formatted correctly."""
        plan = TransformPlan().rows_sort("age", descending=True)
        code = plan.to_python()
        assert "descending=True" in code

    def test_format_params_int_value(self) -> None:
        """Test that int values are formatted correctly."""
        plan = TransformPlan().math_add("value", 10)
        code = plan.to_python()
        assert "value=10" in code

    def test_format_params_float_value(self) -> None:
        """Test that float values are formatted correctly."""
        plan = TransformPlan().math_multiply("price", 1.5)
        code = plan.to_python()
        assert "value=1.5" in code

    def test_format_params_list_value(self) -> None:
        """Test that list values are formatted correctly."""
        plan = TransformPlan().col_select(["a", "b", "c"])
        code = plan.to_python()
        assert "['a', 'b', 'c']" in code

    def test_format_params_dict_value(self) -> None:
        """Test that dict values are formatted correctly."""
        plan = TransformPlan().map_values("status", {"A": "Active", "B": "Blocked"})
        code = plan.to_python()
        assert "mapping=" in code
        assert "'A'" in code


class TestFormatFilterAsPython:
    """Tests for _format_filter_as_python() method."""

    def test_format_filter_and(self) -> None:
        """Test Python code generation for And filter."""
        plan = TransformPlan().rows_filter((Col("age") >= 30) & (Col("id") > 1))
        code = plan.to_python()
        assert "&" in code
        assert 'Col("age")' in code or "Col(" in code

    def test_format_filter_or(self) -> None:
        """Test Python code generation for Or filter."""
        plan = TransformPlan().rows_filter((Col("age") >= 30) | (Col("id") == 1))
        code = plan.to_python()
        assert "|" in code

    def test_format_filter_not(self) -> None:
        """Test Python code generation for Not filter."""
        plan = TransformPlan().rows_filter(~(Col("active") == True))  # noqa: E712
        code = plan.to_python()
        assert "~" in code

    def test_format_filter_is_in(self) -> None:
        """Test Python code generation for IsIn filter."""
        plan = TransformPlan().rows_filter(Col("name").is_in(["Alice", "Bob"]))
        code = plan.to_python()
        assert ".is_in(" in code
        assert "['Alice', 'Bob']" in code

    def test_format_filter_is_null(self) -> None:
        """Test Python code generation for IsNull filter."""
        plan = TransformPlan().rows_filter(Col("name").is_null())
        code = plan.to_python()
        assert ".is_null()" in code

    def test_format_filter_is_not_null(self) -> None:
        """Test Python code generation for IsNotNull filter."""
        plan = TransformPlan().rows_filter(Col("name").is_not_null())
        code = plan.to_python()
        assert ".is_not_null()" in code

    def test_format_filter_between(self) -> None:
        """Test Python code generation for Between filter."""
        plan = TransformPlan().rows_filter(Col("age").between(25, 40))
        code = plan.to_python()
        assert ".between(" in code
        assert "25" in code
        assert "40" in code

    def test_format_filter_str_contains(self) -> None:
        """Test Python code generation for StrContains filter."""
        df = pl.DataFrame({"email": ["test@example.com"]})
        plan = TransformPlan().rows_filter(Col("email").str_contains("@example"))
        plan.validate(df)  # Just validate to ensure it's valid
        code = plan.to_python()
        assert ".str_contains(" in code
        assert "@example" in code

    def test_format_filter_str_starts_with(self) -> None:
        """Test Python code generation for StrStartsWith filter."""
        df = pl.DataFrame({"code": ["PRD-001"]})
        plan = TransformPlan().rows_filter(Col("code").str_starts_with("PRD"))
        plan.validate(df)
        code = plan.to_python()
        assert ".str_starts_with(" in code
        assert "'PRD'" in code

    def test_format_filter_str_ends_with(self) -> None:
        """Test Python code generation for StrEndsWith filter."""
        df = pl.DataFrame({"file": ["data.csv"]})
        plan = TransformPlan().rows_filter(Col("file").str_ends_with(".csv"))
        plan.validate(df)
        code = plan.to_python()
        assert ".str_ends_with(" in code
        assert "'.csv'" in code

    def test_format_filter_unknown_fallback(self) -> None:
        """Test Python code generation for unknown filter type (fallback)."""
        plan = TransformPlan()
        # Call the private method directly with an unknown filter type
        result = plan._format_filter_as_python(
            {"type": "unknown_filter", "column": "x"}
        )
        assert "Filter.from_dict(" in result


class TestFormatParamsEdgeCases:
    """Tests for _format_params_as_python edge cases."""

    def test_format_params_custom_type(self) -> None:
        """Test formatting with custom/unusual type (else branch)."""
        plan = TransformPlan()
        # Create a tuple value which falls into the else branch
        # Use a frozenset or any non-standard type
        result = plan._format_params_as_python({"custom": frozenset([1, 2, 3])})
        assert "custom=" in result
        assert "frozenset" in result
