"""Tests for math operations (ops/math.py)."""

import polars as pl

from transformplan import TransformPlan


class TestMathAdd:
    """Tests for math_add operation."""

    def test_math_add_integer(self, numeric_df: pl.DataFrame) -> None:
        """Test adding an integer."""
        plan = TransformPlan().math_add("a", 10)
        result, _ = plan.process(numeric_df)
        expected = [x + 10 for x in numeric_df["a"].to_list()]
        assert result["a"].to_list() == expected

    def test_math_add_float(self, numeric_df: pl.DataFrame) -> None:
        """Test adding a float."""
        plan = TransformPlan().math_add("a", 0.5)
        result, _ = plan.process(numeric_df)
        assert all(
            abs(r - (o + 0.5)) < 0.001 for r, o in zip(result["a"], numeric_df["a"])
        )

    def test_math_add_negative(self, numeric_df: pl.DataFrame) -> None:
        """Test adding a negative number."""
        plan = TransformPlan().math_add("a", -5)
        result, _ = plan.process(numeric_df)
        expected = [x - 5 for x in numeric_df["a"].to_list()]
        assert result["a"].to_list() == expected

    def test_math_add_nonexistent_raises(self, numeric_df: pl.DataFrame) -> None:
        """Test that adding to nonexistent column fails."""
        plan = TransformPlan().math_add("nonexistent", 10)
        result = plan.validate(numeric_df)
        assert not result.is_valid

    def test_math_add_non_numeric_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that adding to non-numeric column fails."""
        plan = TransformPlan().math_add("name", 10)
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "expected numeric" in str(result.errors[0])


class TestMathSubtract:
    """Tests for math_subtract operation."""

    def test_math_subtract_integer(self, numeric_df: pl.DataFrame) -> None:
        """Test subtracting an integer."""
        plan = TransformPlan().math_subtract("a", 1)
        result, _ = plan.process(numeric_df)
        expected = [x - 1 for x in numeric_df["a"].to_list()]
        assert result["a"].to_list() == expected

    def test_math_subtract_results_negative(self, numeric_df: pl.DataFrame) -> None:
        """Test subtraction resulting in negative values."""
        plan = TransformPlan().math_subtract("a", 10)
        result, _ = plan.process(numeric_df)
        assert any(x < 0 for x in result["a"].to_list())


class TestMathMultiply:
    """Tests for math_multiply operation."""

    def test_math_multiply_integer(self, numeric_df: pl.DataFrame) -> None:
        """Test multiplying by an integer."""
        plan = TransformPlan().math_multiply("a", 2)
        result, _ = plan.process(numeric_df)
        expected = [x * 2 for x in numeric_df["a"].to_list()]
        assert result["a"].to_list() == expected

    def test_math_multiply_float(self, numeric_df: pl.DataFrame) -> None:
        """Test multiplying by a float."""
        plan = TransformPlan().math_multiply("a", 1.5)
        result, _ = plan.process(numeric_df)
        assert all(
            abs(r - (o * 1.5)) < 0.001 for r, o in zip(result["a"], numeric_df["a"])
        )

    def test_math_multiply_by_zero(self, numeric_df: pl.DataFrame) -> None:
        """Test multiplying by zero."""
        plan = TransformPlan().math_multiply("a", 0)
        result, _ = plan.process(numeric_df)
        assert all(x == 0 for x in result["a"].to_list())


class TestMathDivide:
    """Tests for math_divide operation."""

    def test_math_divide_integer(self, numeric_df: pl.DataFrame) -> None:
        """Test dividing by an integer."""
        plan = TransformPlan().math_divide("b", 10)
        result, _ = plan.process(numeric_df)
        expected = [x / 10 for x in numeric_df["b"].to_list()]
        assert result["b"].to_list() == expected

    def test_math_divide_float(self, numeric_df: pl.DataFrame) -> None:
        """Test dividing by a float."""
        plan = TransformPlan().math_divide("b", 2.5)
        result, _ = plan.process(numeric_df)
        assert all(
            abs(r - (o / 2.5)) < 0.001 for r, o in zip(result["b"], numeric_df["b"])
        )


class TestMathClamp:
    """Tests for math_clamp operation."""

    def test_math_clamp_lower_only(self, numeric_df: pl.DataFrame) -> None:
        """Test clamping with only lower bound."""
        plan = TransformPlan().math_clamp("a", lower=3)
        result, _ = plan.process(numeric_df)
        assert all(x >= 3 for x in result["a"].to_list())

    def test_math_clamp_upper_only(self, numeric_df: pl.DataFrame) -> None:
        """Test clamping with only upper bound."""
        plan = TransformPlan().math_clamp("a", upper=3)
        result, _ = plan.process(numeric_df)
        assert all(x <= 3 for x in result["a"].to_list())

    def test_math_clamp_both_bounds(self, numeric_df: pl.DataFrame) -> None:
        """Test clamping with both bounds."""
        plan = TransformPlan().math_clamp("a", lower=2, upper=4)
        result, _ = plan.process(numeric_df)
        assert all(2 <= x <= 4 for x in result["a"].to_list())


class TestMathSetMin:
    """Tests for math_set_min operation."""

    def test_math_set_min(self, numeric_df: pl.DataFrame) -> None:
        """Test setting minimum value."""
        plan = TransformPlan().math_set_min("a", 3)
        result, _ = plan.process(numeric_df)
        assert all(x >= 3 for x in result["a"].to_list())
        # Values already >= 3 should be unchanged
        assert result["a"][2] == 3
        assert result["a"][3] == 4


class TestMathSetMax:
    """Tests for math_set_max operation."""

    def test_math_set_max(self, numeric_df: pl.DataFrame) -> None:
        """Test setting maximum value."""
        plan = TransformPlan().math_set_max("a", 3)
        result, _ = plan.process(numeric_df)
        assert all(x <= 3 for x in result["a"].to_list())
        # Values already <= 3 should be unchanged
        assert result["a"][0] == 1
        assert result["a"][1] == 2


class TestMathAbs:
    """Tests for math_abs operation."""

    def test_math_abs_positive(self, numeric_df: pl.DataFrame) -> None:
        """Test abs on positive values."""
        plan = TransformPlan().math_abs("a")
        result, _ = plan.process(numeric_df)
        assert result["a"].to_list() == numeric_df["a"].to_list()

    def test_math_abs_negative(self) -> None:
        """Test abs on negative values."""
        df = pl.DataFrame({"a": [-1, -2, 3, -4, 5]})
        plan = TransformPlan().math_abs("a")
        result, _ = plan.process(df)
        assert result["a"].to_list() == [1, 2, 3, 4, 5]


class TestMathRound:
    """Tests for math_round operation."""

    def test_math_round_default(self) -> None:
        """Test rounding to default (0 decimals)."""
        df = pl.DataFrame({"a": [1.4, 2.5, 3.6, 4.4, 5.5]})
        plan = TransformPlan().math_round("a")
        result, _ = plan.process(df)
        # Polars uses banker's rounding
        assert all(x == round(o) for x, o in zip(result["a"], df["a"]))

    def test_math_round_decimals(self) -> None:
        """Test rounding to specific decimals."""
        df = pl.DataFrame({"a": [1.234, 2.567, 3.891]})
        plan = TransformPlan().math_round("a", decimals=2)
        result, _ = plan.process(df)
        assert result["a"][0] == 1.23
        assert result["a"][1] == 2.57


class TestMathAddColumns:
    """Tests for math_add_columns operation."""

    def test_math_add_columns(self, numeric_df: pl.DataFrame) -> None:
        """Test adding two columns."""
        plan = TransformPlan().math_add_columns("a", "b", "sum")
        result, _ = plan.process(numeric_df)
        assert "sum" in result.columns
        expected = [a + b for a, b in zip(numeric_df["a"], numeric_df["b"])]
        assert result["sum"].to_list() == expected

    def test_math_add_columns_preserves_originals(
        self, numeric_df: pl.DataFrame
    ) -> None:
        """Test that original columns are preserved."""
        plan = TransformPlan().math_add_columns("a", "b", "sum")
        result, _ = plan.process(numeric_df)
        assert "a" in result.columns
        assert "b" in result.columns


class TestMathSubtractColumns:
    """Tests for math_subtract_columns operation."""

    def test_math_subtract_columns(self, numeric_df: pl.DataFrame) -> None:
        """Test subtracting two columns."""
        plan = TransformPlan().math_subtract_columns("b", "a", "diff")
        result, _ = plan.process(numeric_df)
        assert "diff" in result.columns
        expected = [b - a for a, b in zip(numeric_df["a"], numeric_df["b"])]
        assert result["diff"].to_list() == expected


class TestMathMultiplyColumns:
    """Tests for math_multiply_columns operation."""

    def test_math_multiply_columns(self, numeric_df: pl.DataFrame) -> None:
        """Test multiplying two columns."""
        plan = TransformPlan().math_multiply_columns("a", "b", "product")
        result, _ = plan.process(numeric_df)
        assert "product" in result.columns
        expected = [a * b for a, b in zip(numeric_df["a"], numeric_df["b"])]
        assert result["product"].to_list() == expected


class TestMathDivideColumns:
    """Tests for math_divide_columns operation."""

    def test_math_divide_columns(self, numeric_df: pl.DataFrame) -> None:
        """Test dividing two columns."""
        plan = TransformPlan().math_divide_columns("b", "a", "quotient")
        result, _ = plan.process(numeric_df)
        assert "quotient" in result.columns
        expected = [b / a for a, b in zip(numeric_df["a"], numeric_df["b"])]
        assert result["quotient"].to_list() == expected


class TestMathPercentOf:
    """Tests for math_percent_of operation."""

    def test_math_percent_of_default(self, numeric_df: pl.DataFrame) -> None:
        """Test percent calculation with default multiplier."""
        plan = TransformPlan().math_percent_of("a", "c", "percent")
        result, _ = plan.process(numeric_df)
        assert "percent" in result.columns
        expected = [a / c * 100 for a, c in zip(numeric_df["a"], numeric_df["c"])]
        assert result["percent"].to_list() == expected

    def test_math_percent_of_custom_multiplier(self, numeric_df: pl.DataFrame) -> None:
        """Test percent calculation with custom multiplier."""
        plan = TransformPlan().math_percent_of("a", "c", "ratio", multiply_by=1.0)
        result, _ = plan.process(numeric_df)
        expected = [a / c for a, c in zip(numeric_df["a"], numeric_df["c"])]
        assert result["ratio"].to_list() == expected


class TestMathCumsum:
    """Tests for math_cumsum operation."""

    def test_math_cumsum_basic(self, numeric_df: pl.DataFrame) -> None:
        """Test basic cumulative sum."""
        plan = TransformPlan().math_cumsum("a", "cumsum")
        result, _ = plan.process(numeric_df)
        assert "cumsum" in result.columns
        expected = [1, 3, 6, 10, 15]
        assert result["cumsum"].to_list() == expected

    def test_math_cumsum_in_place(self, numeric_df: pl.DataFrame) -> None:
        """Test cumulative sum in place."""
        plan = TransformPlan().math_cumsum("a")
        result, _ = plan.process(numeric_df)
        expected = [1, 3, 6, 10, 15]
        assert result["a"].to_list() == expected

    def test_math_cumsum_with_group_by(self) -> None:
        """Test cumulative sum with grouping."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B"],
                "value": [1, 2, 3, 10, 20],
            }
        )
        plan = TransformPlan().math_cumsum("value", "cumsum", group_by="group")
        result, _ = plan.process(df)
        assert result["cumsum"].to_list() == [1, 3, 6, 10, 30]


class TestMathRank:
    """Tests for math_rank operation."""

    def test_math_rank_ordinal(self, numeric_df: pl.DataFrame) -> None:
        """Test ordinal ranking."""
        plan = TransformPlan().math_rank("a", "rank")
        result, _ = plan.process(numeric_df)
        assert "rank" in result.columns
        assert result["rank"].to_list() == [1, 2, 3, 4, 5]

    def test_math_rank_descending(self, numeric_df: pl.DataFrame) -> None:
        """Test descending ranking."""
        plan = TransformPlan().math_rank("a", "rank", descending=True)
        result, _ = plan.process(numeric_df)
        assert result["rank"].to_list() == [5, 4, 3, 2, 1]

    def test_math_rank_with_group_by(self) -> None:
        """Test ranking with grouping."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B"],
                "value": [3, 1, 2, 20, 10],
            }
        )
        plan = TransformPlan().math_rank("value", "rank", group_by="group")
        result, _ = plan.process(df)
        # Within group A: 3->3, 1->1, 2->2
        # Within group B: 20->2, 10->1
        assert result["rank"].to_list() == [3, 1, 2, 2, 1]

    def test_math_rank_dense(self) -> None:
        """Test dense ranking with ties."""
        df = pl.DataFrame({"value": [1, 2, 2, 3]})
        plan = TransformPlan().math_rank("value", "rank", method="dense")
        result, _ = plan.process(df)
        assert result["rank"].to_list() == [1, 2, 2, 3]


class TestMathChaining:
    """Tests for chaining multiple math operations."""

    def test_multiple_math_operations(self, numeric_df: pl.DataFrame) -> None:
        """Test chaining multiple math operations."""
        plan = TransformPlan().math_multiply("a", 2).math_add("a", 10).math_round("a")
        result, _ = plan.process(numeric_df)
        # Each value: (x * 2) + 10
        expected = [(x * 2) + 10 for x in numeric_df["a"].to_list()]
        assert result["a"].to_list() == expected

    def test_math_with_column_operations(self, numeric_df: pl.DataFrame) -> None:
        """Test math operations combined with column operations."""
        plan = (
            TransformPlan().math_add_columns("a", "b", "sum").math_multiply("sum", 0.1)
        )
        result, _ = plan.process(numeric_df)
        expected = [(a + b) * 0.1 for a, b in zip(numeric_df["a"], numeric_df["b"])]
        assert all(abs(r - e) < 0.001 for r, e in zip(result["sum"], expected))
