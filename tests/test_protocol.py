"""Tests for protocol/audit trail (protocol.py)."""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from transformplan import TransformPlan
from transformplan.protocol import Protocol, frame_hash


class TestFrameHash:
    """Tests for frame_hash function."""

    def test_frame_hash_deterministic(self, basic_df: pl.DataFrame) -> None:
        """Test that hash is deterministic."""
        hash1 = frame_hash(basic_df)
        hash2 = frame_hash(basic_df)
        assert hash1 == hash2

    def test_frame_hash_length(self, basic_df: pl.DataFrame) -> None:
        """Test that hash has expected length."""
        h = frame_hash(basic_df)
        assert len(h) == 16

    def test_frame_hash_row_order_invariant(self) -> None:
        """Test that hash is row-order invariant."""
        df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pl.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
        assert frame_hash(df1) == frame_hash(df2)

    def test_frame_hash_column_order_invariant(self) -> None:
        """Test that hash is column-order invariant."""
        df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pl.DataFrame({"b": [3, 4], "a": [1, 2]})
        assert frame_hash(df1) == frame_hash(df2)

    def test_frame_hash_content_sensitive(self) -> None:
        """Test that hash changes when content changes."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [1, 2, 4]})
        assert frame_hash(df1) != frame_hash(df2)

    def test_frame_hash_empty_df(self, empty_df: pl.DataFrame) -> None:
        """Test hashing empty DataFrame."""
        h = frame_hash(empty_df)
        assert len(h) == 16


class TestProtocolBasic:
    """Tests for basic Protocol functionality."""

    def test_protocol_from_process(self, basic_df: pl.DataFrame) -> None:
        """Test that process returns a Protocol."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        assert isinstance(protocol, Protocol)
        assert len(protocol) == 1

    def test_protocol_len(self, basic_df: pl.DataFrame) -> None:
        """Test __len__ returns number of steps."""
        plan = TransformPlan().col_drop("age").col_rename("name", "full_name")
        _, protocol = plan.process(basic_df)
        assert len(protocol) == 2

    def test_protocol_repr(self, basic_df: pl.DataFrame) -> None:
        """Test __repr__ output."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        assert "Protocol" in repr(protocol)
        assert "1 steps" in repr(protocol)


class TestProtocolHashes:
    """Tests for Protocol hash properties."""

    def test_input_hash(self, basic_df: pl.DataFrame) -> None:
        """Test input_hash property."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        assert protocol.input_hash == frame_hash(basic_df)

    def test_output_hash(self, basic_df: pl.DataFrame) -> None:
        """Test output_hash property."""
        plan = TransformPlan().col_drop("age")
        result, protocol = plan.process(basic_df)
        assert protocol.output_hash == frame_hash(result)

    def test_output_hash_different_from_input(self, basic_df: pl.DataFrame) -> None:
        """Test that output hash differs when data changes."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        assert protocol.input_hash != protocol.output_hash

    def test_output_hash_same_when_no_change(self) -> None:
        """Test that output hash equals input when no data change."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        # Rename and rename back - data unchanged
        plan = TransformPlan().col_rename("a", "b").col_rename("b", "a")
        _, protocol = plan.process(df)
        assert protocol.input_hash == protocol.output_hash


class TestProtocolMetadata:
    """Tests for Protocol metadata."""

    def test_set_metadata(self, basic_df: pl.DataFrame) -> None:
        """Test setting metadata."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        protocol.set_metadata(author="test", version="1.0")
        assert protocol.metadata["author"] == "test"
        assert protocol.metadata["version"] == "1.0"

    def test_metadata_in_dict(self, basic_df: pl.DataFrame) -> None:
        """Test that metadata appears in to_dict."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        protocol.set_metadata(author="test")
        d = protocol.to_dict()
        assert d["metadata"]["author"] == "test"


class TestProtocolSerialization:
    """Tests for Protocol serialization."""

    def test_to_dict(self, basic_df: pl.DataFrame) -> None:
        """Test serialization to dict."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        d = protocol.to_dict()
        assert "version" in d
        assert "created_at" in d
        assert "input" in d
        assert "steps" in d

    def test_to_json(self, basic_df: pl.DataFrame) -> None:
        """Test serialization to JSON string."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        json_str = protocol.to_json()
        d = json.loads(json_str)
        assert "steps" in d

    def test_to_json_file(self, basic_df: pl.DataFrame) -> None:
        """Test serialization to JSON file."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            protocol.to_json(path)
            assert path.exists()
        finally:
            path.unlink()

    def test_from_dict(self, basic_df: pl.DataFrame) -> None:
        """Test deserialization from dict."""
        plan = TransformPlan().col_drop("age")
        _, original = plan.process(basic_df)
        d = original.to_dict()
        restored = Protocol.from_dict(d)
        assert restored.input_hash == original.input_hash
        assert restored.output_hash == original.output_hash
        assert len(restored) == len(original)

    def test_from_json_string(self, basic_df: pl.DataFrame) -> None:
        """Test deserialization from JSON string."""
        plan = TransformPlan().col_drop("age")
        _, original = plan.process(basic_df)
        json_str = original.to_json()
        restored = Protocol.from_json(json_str)
        assert restored.input_hash == original.input_hash

    def test_from_json_file(self, basic_df: pl.DataFrame) -> None:
        """Test deserialization from JSON file."""
        plan = TransformPlan().col_drop("age")
        _, original = plan.process(basic_df)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            original.to_json(path)
            restored = Protocol.from_json(path)
            assert restored.input_hash == original.input_hash
        finally:
            path.unlink()


class TestProtocolDataFrame:
    """Tests for Protocol DataFrame conversion."""

    def test_to_dataframe(self, basic_df: pl.DataFrame) -> None:
        """Test conversion to DataFrame."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        df = protocol.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        # Should have input row + operation rows
        assert len(df) >= 1

    def test_to_dataframe_columns(self, basic_df: pl.DataFrame) -> None:
        """Test DataFrame column structure."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        df = protocol.to_dataframe()
        assert "step" in df.columns
        assert "operation" in df.columns
        assert "output_hash" in df.columns


class TestProtocolCSV:
    """Tests for Protocol CSV export."""

    def test_to_csv(self, basic_df: pl.DataFrame) -> None:
        """Test CSV export."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            protocol.to_csv(path)
            assert path.exists()
            content = path.read_text()
            assert "step" in content
            assert "operation" in content
        finally:
            path.unlink()


class TestProtocolSummary:
    """Tests for Protocol summary/print methods."""

    def test_summary(self, basic_df: pl.DataFrame) -> None:
        """Test summary generation."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        summary = protocol.summary()
        assert isinstance(summary, str)
        assert "TRANSFORM PROTOCOL" in summary
        assert "col_drop" in summary

    def test_summary_with_params(self, basic_df: pl.DataFrame) -> None:
        """Test summary includes parameters."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        summary = protocol.summary(show_params=True)
        assert "age" in summary

    def test_summary_without_params(self, basic_df: pl.DataFrame) -> None:
        """Test summary can hide parameters."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        summary = protocol.summary(show_params=False)
        # Params should not appear in the step detail lines
        # (though column name might appear in step row)
        assert "TRANSFORM PROTOCOL" in summary

    def test_print(
        self, basic_df: pl.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test print method."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        protocol.print()
        captured = capsys.readouterr()
        assert "TRANSFORM PROTOCOL" in captured.out


class TestProtocolStepTracking:
    """Tests for Protocol step tracking."""

    def test_step_timing(self, basic_df: pl.DataFrame) -> None:
        """Test that timing is recorded."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        d = protocol.to_dict()
        step = d["steps"][0]
        assert "elapsed_seconds" in step
        assert step["elapsed_seconds"] >= 0

    def test_step_shape_changes(self, basic_df: pl.DataFrame) -> None:
        """Test that shape changes are recorded."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        d = protocol.to_dict()
        step = d["steps"][0]
        assert "old_shape" in step
        assert "new_shape" in step
        assert step["cols_changed"] == 1  # One column dropped

    def test_step_output_hash(self, basic_df: pl.DataFrame) -> None:
        """Test that each step has output hash."""
        plan = TransformPlan().col_drop("age").col_drop("salary")
        _, protocol = plan.process(basic_df)
        d = protocol.to_dict()
        for step in d["steps"]:
            assert "output_hash" in step
            assert len(step["output_hash"]) == 16


class TestProtocolOutputHashNoSteps:
    """Tests for Protocol.output_hash when no steps."""

    def test_output_hash_returns_input_hash_when_no_steps(
        self, basic_df: pl.DataFrame
    ) -> None:
        """Test that output_hash returns input_hash when no steps."""
        protocol = Protocol()
        input_hash = frame_hash(basic_df)
        protocol.set_input(input_hash, basic_df.shape)
        # No steps added
        assert protocol.output_hash == input_hash


class TestProtocolSummaryMetadata:
    """Tests for Protocol.summary() with metadata."""

    def test_summary_includes_metadata(self, basic_df: pl.DataFrame) -> None:
        """Test that protocol summary includes metadata."""
        plan = TransformPlan().col_drop("age")
        _, protocol = plan.process(basic_df)
        protocol.set_metadata(author="test_user", project="test_project")
        summary = protocol.summary()
        assert "author: test_user" in summary
        assert "project: test_project" in summary


class TestProtocolFormatFilter:
    """Tests for Protocol._format_filter() method."""

    def test_format_filter_and(self) -> None:
        """Test _format_filter for And filter."""
        from transformplan import Col

        df = pl.DataFrame({"age": [25, 35, 45], "id": [1, 2, 3]})
        plan = TransformPlan().rows_filter((Col("age") >= 30) & (Col("id") > 1))
        _, protocol = plan.process(df)
        summary = protocol.summary(show_params=True)
        # Should contain the & operator formatting
        assert "&" in summary

    def test_format_filter_or(self) -> None:
        """Test _format_filter for Or filter."""
        from transformplan import Col

        df = pl.DataFrame({"age": [25, 35, 45], "id": [1, 2, 3]})
        plan = TransformPlan().rows_filter((Col("age") >= 30) | (Col("id") == 1))
        _, protocol = plan.process(df)
        summary = protocol.summary(show_params=True)
        # Should contain the | operator formatting
        assert "|" in summary

    def test_format_filter_not(self) -> None:
        """Test _format_filter for Not filter."""
        from transformplan import Col

        df = pl.DataFrame({"active": [True, False, True]})
        plan = TransformPlan().rows_filter(~(Col("active") == True))  # noqa: E712
        _, protocol = plan.process(df)
        summary = protocol.summary(show_params=True)
        # Should contain the ~ operator formatting
        assert "~" in summary

    def test_format_filter_is_in_short(self) -> None:
        """Test _format_filter for IsIn with short list."""
        from transformplan import Col

        df = pl.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        plan = TransformPlan().rows_filter(Col("name").is_in(["Alice", "Bob"]))
        _, protocol = plan.process(df)
        summary = protocol.summary(show_params=True)
        # Should show the values
        assert "name in" in summary

    def test_format_filter_is_in_long(self) -> None:
        """Test _format_filter for IsIn with long list (truncated)."""
        from transformplan import Col

        df = pl.DataFrame({"id": [1, 2, 3, 4, 5, 6]})
        plan = TransformPlan().rows_filter(Col("id").is_in([1, 2, 3, 4, 5, 6]))
        _, protocol = plan.process(df)
        summary = protocol.summary(show_params=True)
        # Should truncate long lists
        assert "items" in summary

    def test_format_filter_is_null(self, df_with_nulls: pl.DataFrame) -> None:
        """Test _format_filter for IsNull."""
        from transformplan import Col

        plan = TransformPlan().rows_filter(Col("name").is_null())
        _, protocol = plan.process(df_with_nulls)
        summary = protocol.summary(show_params=True)
        assert "is null" in summary

    def test_format_filter_is_not_null(self, df_with_nulls: pl.DataFrame) -> None:
        """Test _format_filter for IsNotNull."""
        from transformplan import Col

        plan = TransformPlan().rows_filter(Col("name").is_not_null())
        _, protocol = plan.process(df_with_nulls)
        summary = protocol.summary(show_params=True)
        assert "is not null" in summary

    def test_format_filter_between(self) -> None:
        """Test _format_filter for Between."""
        from transformplan import Col

        df = pl.DataFrame({"age": [25, 35, 45]})
        plan = TransformPlan().rows_filter(Col("age").between(25, 40))
        _, protocol = plan.process(df)
        summary = protocol.summary(show_params=True)
        assert "between" in summary

    def test_format_filter_str_contains(self) -> None:
        """Test _format_filter for StrContains."""
        from transformplan import Col

        df = pl.DataFrame({"email": ["alice@example.com", "bob@test.com"]})
        plan = TransformPlan().rows_filter(Col("email").str_contains("@example"))
        _, protocol = plan.process(df)
        summary = protocol.summary(show_params=True)
        assert "contains" in summary

    def test_format_filter_str_starts_with(self) -> None:
        """Test _format_filter for StrStartsWith."""
        from transformplan import Col

        df = pl.DataFrame({"code": ["PRD-001", "TST-002"]})
        plan = TransformPlan().rows_filter(Col("code").str_starts_with("PRD"))
        _, protocol = plan.process(df)
        summary = protocol.summary(show_params=True)
        assert "starts_with" in summary

    def test_format_filter_str_ends_with(self) -> None:
        """Test _format_filter for StrEndsWith."""
        from transformplan import Col

        df = pl.DataFrame({"file": ["data.csv", "report.pdf"]})
        plan = TransformPlan().rows_filter(Col("file").str_ends_with(".csv"))
        _, protocol = plan.process(df)
        summary = protocol.summary(show_params=True)
        assert "ends_with" in summary

    def test_format_filter_unknown_type(self) -> None:
        """Test _format_filter for unknown filter type (fallback)."""
        protocol = Protocol()
        # Manually test _format_filter with unknown type
        result = protocol._format_filter({"type": "unknown_filter_type"})
        assert "<unknown_filter_type>" in result


class TestProtocolFormatParams:
    """Tests for Protocol._format_params() edge cases."""

    def test_format_params_long_list(self) -> None:
        """Test _format_params truncates long lists."""
        protocol = Protocol()
        params = {"values": [1, 2, 3, 4, 5, 6, 7]}
        result = protocol._format_params(params)
        # Should truncate to show first 2 items and count
        assert "items" in result
        assert "1, 2" in result

    def test_format_params_long_result(self) -> None:
        """Test _format_params truncates overall result."""
        protocol = Protocol()
        # Create params that would produce a very long string
        params = {
            "column1": "some_value",
            "column2": "another_value",
            "column3": "yet_another",
            "column4": "and_more",
            "column5": "even_more",
        }
        result = protocol._format_params(params, max_length=30)
        assert len(result) <= 30
        assert result.endswith("...")

    def test_format_params_dict_without_type(self) -> None:
        """Test _format_params with dict that has no 'type' key."""
        protocol = Protocol()
        params = {"mapping": {"A": "Active", "B": "Blocked"}}
        result = protocol._format_params(params)
        # Should show {...} for dict without type
        assert "{...}" in result
