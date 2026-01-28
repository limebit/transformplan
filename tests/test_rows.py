"""Tests for row operations (ops/rows.py)."""

import polars as pl

from transformplan import Col, TransformPlan


class TestRowsDropNulls:
    """Tests for rows_drop_nulls operation."""

    def test_rows_drop_nulls_all_columns(self, df_with_nulls: pl.DataFrame) -> None:
        """Test dropping rows with nulls in any column."""
        plan = TransformPlan().rows_drop_nulls()
        result, _ = plan.process(df_with_nulls)
        # Check no nulls in any column
        assert all(result[col].null_count() == 0 for col in result.columns)
        assert len(result) < len(df_with_nulls)

    def test_rows_drop_nulls_specific_column(self, df_with_nulls: pl.DataFrame) -> None:
        """Test dropping rows with nulls in specific column."""
        plan = TransformPlan().rows_drop_nulls("name")
        result, _ = plan.process(df_with_nulls)
        assert result["name"].null_count() == 0
        # Other columns may still have nulls
        assert len(result) == 3  # Original has 2 null names

    def test_rows_drop_nulls_multiple_columns(self, df_with_nulls: pl.DataFrame) -> None:
        """Test dropping rows with nulls in multiple columns."""
        plan = TransformPlan().rows_drop_nulls(["name", "age"])
        result, _ = plan.process(df_with_nulls)
        assert result["name"].null_count() == 0
        assert result["age"].null_count() == 0


class TestRowsUnique:
    """Tests for rows_unique operation."""

    def test_rows_unique_all_columns(self) -> None:
        """Test keeping unique rows based on all columns."""
        # Add some exact duplicates
        df = pl.DataFrame(
            {
                "a": [1, 1, 2, 2, 3],
                "b": [10, 10, 20, 20, 30],
            }
        )
        plan = TransformPlan().rows_unique()
        result, _ = plan.process(df)
        assert len(result) == 3

    def test_rows_unique_specific_column(self, duplicates_df: pl.DataFrame) -> None:
        """Test keeping unique rows based on specific column."""
        plan = TransformPlan().rows_unique("id")
        result, _ = plan.process(duplicates_df)
        # id has values [1, 1, 2, 2, 3] -> 3 unique
        assert len(result) == 3

    def test_rows_unique_keep_first(self) -> None:
        """Test keeping first occurrence."""
        df = pl.DataFrame(
            {
                "key": [1, 1, 1],
                "value": ["a", "b", "c"],
            }
        )
        plan = TransformPlan().rows_unique("key", keep="first")
        result, _ = plan.process(df)
        assert result["value"][0] == "a"

    def test_rows_unique_keep_last(self) -> None:
        """Test keeping last occurrence."""
        df = pl.DataFrame(
            {
                "key": [1, 1, 1],
                "value": ["a", "b", "c"],
            }
        )
        plan = TransformPlan().rows_unique("key", keep="last")
        result, _ = plan.process(df)
        assert result["value"][0] == "c"


class TestRowsFilter:
    """Tests for rows_filter operation."""

    def test_rows_filter_eq(self, basic_df: pl.DataFrame) -> None:
        """Test filtering with equality."""
        plan = TransformPlan().rows_filter(Col("name") == "Alice")
        result, _ = plan.process(basic_df)
        assert len(result) == 1
        assert result["name"][0] == "Alice"

    def test_rows_filter_gt(self, basic_df: pl.DataFrame) -> None:
        """Test filtering with greater than."""
        plan = TransformPlan().rows_filter(Col("age") > 30)
        result, _ = plan.process(basic_df)
        assert all(age > 30 for age in result["age"].to_list())

    def test_rows_filter_combined(self, basic_df: pl.DataFrame) -> None:
        """Test filtering with combined conditions."""
        plan = TransformPlan().rows_filter(
            (Col("age") >= 30) & (Col("active") == True)  # noqa: E712
        )
        result, _ = plan.process(basic_df)
        assert all(age >= 30 for age in result["age"].to_list())
        assert all(active for active in result["active"].to_list())

    def test_rows_filter_or_condition(self, basic_df: pl.DataFrame) -> None:
        """Test filtering with OR condition."""
        plan = TransformPlan().rows_filter(
            (Col("name") == "Alice") | (Col("name") == "Bob")
        )
        result, _ = plan.process(basic_df)
        assert len(result) == 2
        assert set(result["name"].to_list()) == {"Alice", "Bob"}

    def test_rows_filter_is_in(self, basic_df: pl.DataFrame) -> None:
        """Test filtering with is_in."""
        plan = TransformPlan().rows_filter(Col("name").is_in(["Alice", "Charlie"]))
        result, _ = plan.process(basic_df)
        assert len(result) == 2

    def test_rows_filter_nonexistent_column_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that filtering on nonexistent column fails."""
        plan = TransformPlan().rows_filter(Col("nonexistent") == "value")
        result = plan.validate(basic_df)
        assert not result.is_valid


class TestRowsDrop:
    """Tests for rows_drop operation."""

    def test_rows_drop_basic(self, basic_df: pl.DataFrame) -> None:
        """Test dropping rows matching filter."""
        plan = TransformPlan().rows_drop(Col("name") == "Alice")
        result, _ = plan.process(basic_df)
        assert "Alice" not in result["name"].to_list()
        assert len(result) == len(basic_df) - 1

    def test_rows_drop_multiple(self, basic_df: pl.DataFrame) -> None:
        """Test dropping multiple rows."""
        plan = TransformPlan().rows_drop(Col("active") == False)  # noqa: E712
        result, _ = plan.process(basic_df)
        assert all(active for active in result["active"].to_list())


class TestRowsDeduplicate:
    """Tests for rows_deduplicate operation."""

    def test_rows_deduplicate_keep_first(self, duplicates_df: pl.DataFrame) -> None:
        """Test deduplication keeping first by sort order."""
        plan = TransformPlan().rows_deduplicate("id", sort_by="timestamp", keep="first")
        result, _ = plan.process(duplicates_df)
        assert len(result) == 3  # 3 unique ids

    def test_rows_deduplicate_keep_last(self, duplicates_df: pl.DataFrame) -> None:
        """Test deduplication keeping last by sort order."""
        plan = TransformPlan().rows_deduplicate("id", sort_by="timestamp", keep="last")
        result, _ = plan.process(duplicates_df)
        assert len(result) == 3

    def test_rows_deduplicate_descending(self, duplicates_df: pl.DataFrame) -> None:
        """Test deduplication with descending sort."""
        plan = TransformPlan().rows_deduplicate(
            "id", sort_by="timestamp", keep="first", descending=True
        )
        result, _ = plan.process(duplicates_df)
        # With descending sort, "first" gets the latest timestamp
        assert len(result) == 3


class TestRowsExplode:
    """Tests for rows_explode operation."""

    def test_rows_explode_basic(self, list_df: pl.DataFrame) -> None:
        """Test basic explode operation."""
        plan = TransformPlan().rows_explode("tags")
        result, _ = plan.process(list_df)
        # [a,b,c], [d,e], [f] = 6 total rows
        assert len(result) == 6
        assert "tags" in result.columns
        # Tags should be scalars now
        assert result["tags"].dtype == pl.Utf8

    def test_rows_explode_preserves_other_columns(self, list_df: pl.DataFrame) -> None:
        """Test that other columns are preserved."""
        plan = TransformPlan().rows_explode("tags")
        result, _ = plan.process(list_df)
        assert "id" in result.columns
        assert "name" in result.columns


class TestRowsMelt:
    """Tests for rows_melt operation."""

    def test_rows_melt_basic(self, wide_df: pl.DataFrame) -> None:
        """Test basic melt operation."""
        plan = TransformPlan().rows_melt(
            id_columns=["id", "name"],
            value_columns=["q1", "q2", "q3", "q4"],
        )
        result, _ = plan.process(wide_df)
        # 3 rows * 4 quarters = 12 rows
        assert len(result) == 12
        assert "variable" in result.columns
        assert "value" in result.columns

    def test_rows_melt_custom_names(self, wide_df: pl.DataFrame) -> None:
        """Test melt with custom variable/value names."""
        plan = TransformPlan().rows_melt(
            id_columns=["id"],
            value_columns=["q1", "q2"],
            variable_name="quarter",
            value_name="sales",
        )
        result, _ = plan.process(wide_df)
        assert "quarter" in result.columns
        assert "sales" in result.columns


class TestRowsSample:
    """Tests for rows_sample operation."""

    def test_rows_sample_n(self, basic_df: pl.DataFrame) -> None:
        """Test sampling n rows."""
        plan = TransformPlan().rows_sample(n=3, seed=42)
        result, _ = plan.process(basic_df)
        assert len(result) == 3

    def test_rows_sample_fraction(self, basic_df: pl.DataFrame) -> None:
        """Test sampling by fraction."""
        plan = TransformPlan().rows_sample(fraction=0.4, seed=42)
        result, _ = plan.process(basic_df)
        assert len(result) == 2  # 40% of 5

    def test_rows_sample_reproducible(self, basic_df: pl.DataFrame) -> None:
        """Test that sampling with seed is reproducible."""
        plan = TransformPlan().rows_sample(n=3, seed=42)
        result1, _ = plan.process(basic_df)
        result2, _ = plan.process(basic_df)
        assert result1.equals(result2)


class TestRowsHead:
    """Tests for rows_head operation."""

    def test_rows_head_default(self, basic_df: pl.DataFrame) -> None:
        """Test head with default n=5."""
        plan = TransformPlan().rows_head()
        result, _ = plan.process(basic_df)
        assert len(result) == 5

    def test_rows_head_custom_n(self, basic_df: pl.DataFrame) -> None:
        """Test head with custom n."""
        plan = TransformPlan().rows_head(n=2)
        result, _ = plan.process(basic_df)
        assert len(result) == 2
        assert result["id"].to_list() == [1, 2]

    def test_rows_head_exceeds_length(self, basic_df: pl.DataFrame) -> None:
        """Test head when n exceeds DataFrame length."""
        plan = TransformPlan().rows_head(n=100)
        result, _ = plan.process(basic_df)
        assert len(result) == len(basic_df)


class TestRowsTail:
    """Tests for rows_tail operation."""

    def test_rows_tail_default(self, basic_df: pl.DataFrame) -> None:
        """Test tail with default n=5."""
        plan = TransformPlan().rows_tail()
        result, _ = plan.process(basic_df)
        assert len(result) == 5

    def test_rows_tail_custom_n(self, basic_df: pl.DataFrame) -> None:
        """Test tail with custom n."""
        plan = TransformPlan().rows_tail(n=2)
        result, _ = plan.process(basic_df)
        assert len(result) == 2
        assert result["id"].to_list() == [4, 5]


class TestRowsSort:
    """Tests for rows_sort operation."""

    def test_rows_sort_single_column(self, basic_df: pl.DataFrame) -> None:
        """Test sorting by single column."""
        plan = TransformPlan().rows_sort("name")
        result, _ = plan.process(basic_df)
        assert result["name"].to_list() == sorted(basic_df["name"].to_list())

    def test_rows_sort_descending(self, basic_df: pl.DataFrame) -> None:
        """Test sorting in descending order."""
        plan = TransformPlan().rows_sort("age", descending=True)
        result, _ = plan.process(basic_df)
        assert result["age"].to_list() == sorted(basic_df["age"].to_list(), reverse=True)

    def test_rows_sort_multiple_columns(self) -> None:
        """Test sorting by multiple columns."""
        df = pl.DataFrame(
            {
                "group": ["B", "A", "A", "B"],
                "value": [1, 2, 1, 2],
            }
        )
        plan = TransformPlan().rows_sort(["group", "value"])
        result, _ = plan.process(df)
        assert result["group"].to_list() == ["A", "A", "B", "B"]
        assert result["value"].to_list() == [1, 2, 1, 2]


class TestRowsFlag:
    """Tests for rows_flag operation."""

    def test_rows_flag_boolean(self, basic_df: pl.DataFrame) -> None:
        """Test flagging with boolean values."""
        plan = TransformPlan().rows_flag(Col("age") >= 35, "is_senior")
        result, _ = plan.process(basic_df)
        assert "is_senior" in result.columns
        # Check flag values
        for i, age in enumerate(result["age"].to_list()):
            expected = age >= 35
            assert result["is_senior"][i] == expected

    def test_rows_flag_custom_values(self, basic_df: pl.DataFrame) -> None:
        """Test flagging with custom values."""
        plan = TransformPlan().rows_flag(
            Col("active") == True,  # noqa: E712
            "status",
            true_value="active",
            false_value="inactive",
        )
        result, _ = plan.process(basic_df)
        assert "status" in result.columns
        assert set(result["status"].to_list()) == {"active", "inactive"}


class TestRowsPivot:
    """Tests for rows_pivot operation."""

    def test_rows_pivot_basic(self, long_df: pl.DataFrame) -> None:
        """Test basic pivot operation."""
        plan = TransformPlan().rows_pivot(
            index="id", columns="quarter", values="value"
        )
        result, _ = plan.process(long_df)
        assert "Q1" in result.columns
        assert "Q2" in result.columns
        assert len(result) == 3  # 3 unique ids

    def test_rows_pivot_sum_aggregate(self) -> None:
        """Test pivot with sum aggregation."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "quarter": ["Q1", "Q1", "Q2", "Q2"],
                "value": [10, 20, 30, 40],
            }
        )
        plan = TransformPlan().rows_pivot(
            index="id", columns="quarter", values="value", aggregate_function="sum"
        )
        result, _ = plan.process(df)
        assert result["Q1"][0] == 30  # 10 + 20
        assert result["Q2"][0] == 70  # 30 + 40


class TestRowChaining:
    """Tests for chaining multiple row operations."""

    def test_filter_then_sort(self, basic_df: pl.DataFrame) -> None:
        """Test filtering then sorting."""
        plan = TransformPlan().rows_filter(Col("age") >= 30).rows_sort("age")
        result, _ = plan.process(basic_df)
        assert all(age >= 30 for age in result["age"].to_list())
        assert result["age"].to_list() == sorted(result["age"].to_list())

    def test_filter_then_head(self, basic_df: pl.DataFrame) -> None:
        """Test filtering then taking head."""
        plan = TransformPlan().rows_filter(Col("active") == True).rows_head(2)  # noqa: E712
        result, _ = plan.process(basic_df)
        assert len(result) == 2
        assert all(active for active in result["active"].to_list())


class TestEdgeCases:
    """Tests for edge cases in row operations."""

    def test_filter_empty_result(self, basic_df: pl.DataFrame) -> None:
        """Test filter that results in empty DataFrame."""
        plan = TransformPlan().rows_filter(Col("age") > 1000)
        result, _ = plan.process(basic_df)
        assert len(result) == 0
        assert result.columns == basic_df.columns

    def test_operations_on_empty_df(self, empty_df: pl.DataFrame) -> None:
        """Test operations on empty DataFrame."""
        plan = TransformPlan().rows_sort("id").rows_head(10)
        result, _ = plan.process(empty_df)
        assert len(result) == 0

    def test_operations_on_single_row(self, single_row_df: pl.DataFrame) -> None:
        """Test operations on single row DataFrame."""
        plan = TransformPlan().rows_unique()
        result, _ = plan.process(single_row_df)
        assert len(result) == 1
