"""Tests for map operations (ops/map.py)."""

import polars as pl
import pytest

from transformplan import TransformPlan


class TestMapValues:
    """Tests for map_values operation."""

    def test_map_values_basic(self, map_df: pl.DataFrame) -> None:
        """Test basic value mapping."""
        plan = TransformPlan().map_values(
            "status", {"A": "Active", "B": "Blocked", "C": "Closed"}
        )
        result, _ = plan.process(map_df)
        expected = ["Active", "Blocked", "Active", "Closed", "Blocked"]
        assert result["status"].to_list() == expected

    def test_map_values_keep_unmapped(self, map_df: pl.DataFrame) -> None:
        """Test keeping unmapped values."""
        plan = TransformPlan().map_values("status", {"A": "Active"}, keep_unmapped=True)
        result, _ = plan.process(map_df)
        # B and C should stay as is
        assert "B" in result["status"].to_list()
        assert "C" in result["status"].to_list()

    def test_map_values_replace_unmapped(self, map_df: pl.DataFrame) -> None:
        """Test replacing unmapped values with default."""
        plan = TransformPlan().map_values(
            "status", {"A": "Active"}, default="Unknown", keep_unmapped=False
        )
        result, _ = plan.process(map_df)
        assert "B" not in result["status"].to_list()
        assert "Unknown" in result["status"].to_list()

    def test_map_values_empty_mapping(self, map_df: pl.DataFrame) -> None:
        """Test with empty mapping."""
        plan = TransformPlan().map_values("status", {})
        result, _ = plan.process(map_df)
        # Should be unchanged
        assert result["status"].to_list() == map_df["status"].to_list()

    def test_map_values_numeric(self) -> None:
        """Test mapping numeric values."""
        df = pl.DataFrame({"code": [1, 2, 3, 1, 2]})
        plan = TransformPlan().map_values("code", {1: "One", 2: "Two", 3: "Three"})
        result, _ = plan.process(df)
        assert result["code"].to_list() == ["One", "Two", "Three", "One", "Two"]


class TestMapDiscretize:
    """Tests for map_discretize operation."""

    def test_map_discretize_basic(self, map_df: pl.DataFrame) -> None:
        """Test basic discretization."""
        plan = TransformPlan().map_discretize(
            "score",
            bins=[60, 70, 80, 90],
            labels=["F", "D", "C", "B", "A"],
            new_column="grade",
        )
        result, _ = plan.process(map_df)
        assert "grade" in result.columns
        # score: 85, 72, 91, 68, 79
        # 85 -> (80,90] -> B
        # 72 -> (70,80] -> C
        # 91 -> (90,inf] -> A
        # 68 -> (60,70] -> D
        # 79 -> (70,80] -> C
        assert result["grade"][0] == "B"
        assert result["grade"][2] == "A"

    def test_map_discretize_auto_labels(self, map_df: pl.DataFrame) -> None:
        """Test discretization with auto-generated labels."""
        plan = TransformPlan().map_discretize(
            "score", bins=[60, 80], new_column="bucket"
        )
        result, _ = plan.process(map_df)
        assert "bucket" in result.columns

    def test_map_discretize_in_place(self, map_df: pl.DataFrame) -> None:
        """Test discretization in place."""
        # bins create len(bins)+1 categories: (-inf, 50], (50, 75], (75, 100], (100, inf]
        # So we need 4 labels
        plan = TransformPlan().map_discretize(
            "score", bins=[50, 75, 100], labels=["Fail", "Low", "Medium", "High"]
        )
        result, _ = plan.process(map_df)
        assert result["score"].dtype == pl.Utf8


class TestMapBoolToInt:
    """Tests for map_bool_to_int operation."""

    def test_map_bool_to_int(self, basic_df: pl.DataFrame) -> None:
        """Test converting boolean to integer."""
        plan = TransformPlan().map_bool_to_int("active")
        result, _ = plan.process(basic_df)
        assert result["active"].dtype == pl.Int64
        # True -> 1, False -> 0
        expected = [1, 1, 0, 1, 0]
        assert result["active"].to_list() == expected


class TestMapNullToValue:
    """Tests for map_null_to_value operation."""

    def test_map_null_to_value_string(self, df_with_nulls: pl.DataFrame) -> None:
        """Test replacing nulls with string value."""
        plan = TransformPlan().map_null_to_value("name", "Unknown")
        result, _ = plan.process(df_with_nulls)
        assert result["name"].null_count() == 0
        assert "Unknown" in result["name"].to_list()

    def test_map_null_to_value_numeric(self, df_with_nulls: pl.DataFrame) -> None:
        """Test replacing nulls with numeric value."""
        plan = TransformPlan().map_null_to_value("age", 0)
        result, _ = plan.process(df_with_nulls)
        assert result["age"].null_count() == 0
        assert 0 in result["age"].to_list()


class TestMapValueToNull:
    """Tests for map_value_to_null operation."""

    def test_map_value_to_null_string(self) -> None:
        """Test replacing specific string with null."""
        df = pl.DataFrame({"status": ["active", "N/A", "inactive", "N/A"]})
        plan = TransformPlan().map_value_to_null("status", "N/A")
        result, _ = plan.process(df)
        assert result["status"].null_count() == 2

    def test_map_value_to_null_numeric(self) -> None:
        """Test replacing specific number with null."""
        df = pl.DataFrame({"value": [1, -1, 2, -1, 3]})
        plan = TransformPlan().map_value_to_null("value", -1)
        result, _ = plan.process(df)
        assert result["value"].null_count() == 2


class TestMapCase:
    """Tests for map_case operation."""

    def test_map_case_basic(self, map_df: pl.DataFrame) -> None:
        """Test basic case mapping."""
        plan = TransformPlan().map_case(
            "status",
            cases=[("A", "Active"), ("B", "Blocked"), ("C", "Closed")],
            default="Unknown",
        )
        result, _ = plan.process(map_df)
        assert "Active" in result["status"].to_list()
        assert "Blocked" in result["status"].to_list()
        assert "Closed" in result["status"].to_list()

    def test_map_case_new_column(self, map_df: pl.DataFrame) -> None:
        """Test case mapping to new column."""
        plan = TransformPlan().map_case(
            "status",
            cases=[("A", 1), ("B", 2), ("C", 3)],
            default=0,
            new_column="status_code",
        )
        result, _ = plan.process(map_df)
        assert "status_code" in result.columns
        assert "status" in result.columns  # Original preserved

    def test_map_case_empty_cases(self, map_df: pl.DataFrame) -> None:
        """Test case mapping with empty cases list."""
        plan = TransformPlan().map_case("status", cases=[], default="default")
        result, _ = plan.process(map_df)
        assert all(s == "default" for s in result["status"].to_list())

    def test_map_case_default_only(self, map_df: pl.DataFrame) -> None:
        """Test case mapping where only default matches."""
        plan = TransformPlan().map_case(
            "status", cases=[("X", "Found")], default="Not Found"
        )
        result, _ = plan.process(map_df)
        assert all(s == "Not Found" for s in result["status"].to_list())


class TestMapFromColumn:
    """Tests for map_from_column operation."""

    @pytest.mark.filterwarnings(
        "ignore:.*`default` parameter for `replace` is deprecated.*:DeprecationWarning"
    )
    def test_map_from_column_basic(self, map_df: pl.DataFrame) -> None:
        """Test mapping values using another column as lookup."""
        plan = TransformPlan().map_from_column(
            "lookup_key", "lookup_key", "lookup_value", "mapped_value"
        )
        result, _ = plan.process(map_df)
        assert "mapped_value" in result.columns

    @pytest.mark.filterwarnings(
        "ignore:.*`default` parameter for `replace` is deprecated.*:DeprecationWarning"
    )
    def test_map_from_column_with_default(self) -> None:
        """Test mapping with default for missing lookups."""
        df = pl.DataFrame(
            {
                "key": [1, 2, 3, 99],
                "lookup_key": [1, 2, 3, 3],
                "lookup_value": ["A", "B", "C", "C"],
            }
        )
        plan = TransformPlan().map_from_column(
            "key", "lookup_key", "lookup_value", "result", default="Unknown"
        )
        result, _ = plan.process(df)
        # 99 is not in lookup_key, should get default
        assert result["result"][3] == "Unknown"


class TestMapChaining:
    """Tests for chaining map operations."""

    def test_multiple_map_operations(self, map_df: pl.DataFrame) -> None:
        """Test chaining multiple map operations."""
        plan = (
            TransformPlan()
            .map_values("status", {"A": "Active", "B": "Blocked", "C": "Closed"})
            .map_discretize(
                "score",
                bins=[70, 85],
                labels=["Low", "Medium", "High"],
                new_column="score_band",
            )
        )
        result, _ = plan.process(map_df)
        assert "Active" in result["status"].to_list()
        assert "score_band" in result.columns

    def test_null_mapping_chain(self, df_with_nulls: pl.DataFrame) -> None:
        """Test chaining null-related map operations."""
        plan = (
            TransformPlan()
            .map_null_to_value("name", "N/A")
            .map_value_to_null("name", "N/A")
        )
        result, _ = plan.process(df_with_nulls)
        # Should have same null count as original
        assert result["name"].null_count() == df_with_nulls["name"].null_count()


class TestMapEdgeCases:
    """Tests for edge cases in map operations."""

    def test_map_values_with_none_key(self) -> None:
        """Test mapping with None in values."""
        df = pl.DataFrame({"status": ["A", None, "B"]})
        plan = TransformPlan().map_values("status", {"A": "Active", "B": "Blocked"})
        result, _ = plan.process(df)
        # None should pass through
        assert result["status"][1] is None

    def test_discretize_at_boundaries(self) -> None:
        """Test discretization at exact boundary values."""
        df = pl.DataFrame({"value": [0, 50, 100]})
        plan = TransformPlan().map_discretize(
            "value", bins=[50], labels=["Low", "High"], new_column="category"
        )
        result, _ = plan.process(df)
        # 0 -> Low (<=50), 50 -> Low (<=50), 100 -> High (>50)
        assert result["category"][0] == "Low"
        assert result["category"][1] == "Low"  # right=True by default, so 50 is in (50]
        assert result["category"][2] == "High"

    def test_map_values_type_coercion(self) -> None:
        """Test that mapping can change value types."""
        df = pl.DataFrame({"code": ["1", "2", "3"]})
        plan = TransformPlan().map_values(
            "code", {"1": "One", "2": "Two", "3": "Three"}
        )
        result, _ = plan.process(df)
        assert result["code"].dtype == pl.Utf8


class TestMapDiscretizeRightFalse:
    """Tests for map_discretize with right=False."""

    def test_map_discretize_right_false(self) -> None:
        """Test discretization with right=False (left-closed intervals)."""
        df = pl.DataFrame({"value": [0, 50, 100]})
        plan = TransformPlan().map_discretize(
            "value",
            bins=[50],
            labels=["Low", "High"],
            new_column="category",
            right=False,
        )
        result, _ = plan.process(df)
        # With right=False: [left, right)
        # 0 -> [-inf, 50) -> Low
        # 50 -> [50, inf) -> High (50 is at the boundary, goes to High)
        # 100 -> [50, inf) -> High
        assert result["category"][0] == "Low"
        assert result["category"][1] == "High"
        assert result["category"][2] == "High"

    def test_map_discretize_right_false_auto_labels(self) -> None:
        """Test discretization with right=False and auto-generated labels."""
        df = pl.DataFrame({"value": [10, 50, 90]})
        plan = TransformPlan().map_discretize(
            "value", bins=[30, 70], new_column="bucket", right=False
        )
        result, _ = plan.process(df)
        assert "bucket" in result.columns
        # Auto-labels should be like "[-inf, 30)", "[30, 70)", "[70, inf)"
        # 10 -> "[-inf, 30)"
        # 50 -> "[30, 70)"
        # 90 -> "[70, inf)"
        labels = result["bucket"].to_list()
        assert "[-inf, 30)" in labels[0]
        assert "[30, 70)" in labels[1] or "[30.0, 70.0)" in labels[1]
        assert "[70" in labels[2]

    def test_map_discretize_right_false_multiple_bins(self) -> None:
        """Test discretization with right=False and multiple bins."""
        df = pl.DataFrame({"score": [0, 60, 70, 80, 100]})
        plan = TransformPlan().map_discretize(
            "score",
            bins=[60, 70, 80, 90],
            labels=["F", "D", "C", "B", "A"],
            new_column="grade",
            right=False,
        )
        result, _ = plan.process(df)
        # With right=False (left-closed):
        # 0 -> [-inf, 60) -> F
        # 60 -> [60, 70) -> D
        # 70 -> [70, 80) -> C
        # 80 -> [80, 90) -> B
        # 100 -> [90, inf) -> A
        assert result["grade"][0] == "F"
        assert result["grade"][1] == "D"
        assert result["grade"][2] == "C"
        assert result["grade"][3] == "B"
        assert result["grade"][4] == "A"
