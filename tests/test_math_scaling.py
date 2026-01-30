"""Tests for ML preprocessing operations (scaling, transforms, outlier handling)."""

import polars as pl

from transformplan import TransformPlan


class TestMathStandardize:
    """Tests for math_standardize operation (z-score)."""

    def test_standardize_with_explicit_params(self, numeric_df: pl.DataFrame) -> None:
        """Test standardization with explicit mean and std."""
        plan = TransformPlan().math_standardize("a", mean=3, std=1)
        result, _ = plan.process(numeric_df)
        # (value - mean) / std = (1-3)/1=-2, (2-3)/1=-1, etc.
        expected = [-2.0, -1.0, 0.0, 1.0, 2.0]
        assert result["a"].to_list() == expected

    def test_standardize_derive_from_data(self, numeric_df: pl.DataFrame) -> None:
        """Test standardization deriving params from data."""
        plan = TransformPlan().math_standardize("a")
        result, _ = plan.process(numeric_df)
        # Mean of [1,2,3,4,5] = 3, std ≈ 1.58
        # Result should have mean ≈ 0 and std ≈ 1
        values = result["a"].to_list()
        mean_val = sum(v for v in values if v is not None) / len(values)
        assert abs(mean_val) < 0.001
        # Check that the values approximate a standardized distribution
        # Original [1,2,3,4,5] -> z-scores should be approximately [-1.26, -0.63, 0, 0.63, 1.26]
        assert values[2] is not None
        assert abs(values[2]) < 0.001  # Middle value should be ~0

    def test_standardize_to_new_column(self, numeric_df: pl.DataFrame) -> None:
        """Test standardization to a new column."""
        plan = TransformPlan().math_standardize("a", mean=3, std=1, new_column="a_z")
        result, _ = plan.process(numeric_df)
        assert "a" in result.columns
        assert "a_z" in result.columns
        assert result["a"].to_list() == [1, 2, 3, 4, 5]
        assert result["a_z"].to_list() == [-2.0, -1.0, 0.0, 1.0, 2.0]

    def test_standardize_zero_std(self) -> None:
        """Test standardization when std is zero (constant values)."""
        df = pl.DataFrame({"a": [5.0, 5.0, 5.0, 5.0, 5.0]})
        plan = TransformPlan().math_standardize("a")
        result, _ = plan.process(df)
        # Should return zeros to avoid division by zero
        assert result["a"].to_list() == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_standardize_nonexistent_column(self, numeric_df: pl.DataFrame) -> None:
        """Test validation fails for nonexistent column."""
        plan = TransformPlan().math_standardize("nonexistent")
        result = plan.validate(numeric_df)
        assert not result.is_valid

    def test_standardize_non_numeric_column(self, basic_df: pl.DataFrame) -> None:
        """Test validation fails for non-numeric column."""
        plan = TransformPlan().math_standardize("name")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "expected numeric" in str(result.errors[0])


class TestMathMinmax:
    """Tests for math_minmax operation (min-max normalization)."""

    def test_minmax_default_range(self, numeric_df: pl.DataFrame) -> None:
        """Test min-max scaling to default [0, 1] range."""
        plan = TransformPlan().math_minmax("a", min_val=1, max_val=5)
        result, _ = plan.process(numeric_df)
        # (value - min) / (max - min) = (1-1)/4=0, (2-1)/4=0.25, etc.
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert result["a"].to_list() == expected

    def test_minmax_custom_range(self, numeric_df: pl.DataFrame) -> None:
        """Test min-max scaling to custom range."""
        plan = TransformPlan().math_minmax(
            "a", min_val=1, max_val=5, feature_range=(0, 10)
        )
        result, _ = plan.process(numeric_df)
        # 0 + (value - 1) * 10 / 4 = 0, 2.5, 5, 7.5, 10
        expected = [0.0, 2.5, 5.0, 7.5, 10.0]
        assert result["a"].to_list() == expected

    def test_minmax_derive_from_data(self, numeric_df: pl.DataFrame) -> None:
        """Test min-max scaling deriving params from data."""
        plan = TransformPlan().math_minmax("a")
        result, _ = plan.process(numeric_df)
        # Min=1, Max=5, so result should be [0, 0.25, 0.5, 0.75, 1.0]
        assert result["a"].min() == 0.0
        assert result["a"].max() == 1.0

    def test_minmax_to_new_column(self, numeric_df: pl.DataFrame) -> None:
        """Test min-max to a new column."""
        plan = TransformPlan().math_minmax(
            "a", min_val=1, max_val=5, new_column="a_norm"
        )
        result, _ = plan.process(numeric_df)
        assert "a" in result.columns
        assert "a_norm" in result.columns
        assert result["a"].to_list() == [1, 2, 3, 4, 5]

    def test_minmax_constant_values(self) -> None:
        """Test min-max when all values are the same."""
        df = pl.DataFrame({"a": [5.0, 5.0, 5.0, 5.0, 5.0]})
        plan = TransformPlan().math_minmax("a")
        result, _ = plan.process(df)
        # Should return midpoint of range when min == max
        assert all(x == 0.5 for x in result["a"].to_list())


class TestMathRobustScale:
    """Tests for math_robust_scale operation (median/IQR scaling)."""

    def test_robust_scale_with_explicit_params(self) -> None:
        """Test robust scaling with explicit median and IQR."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        plan = TransformPlan().math_robust_scale("a", median=3.0, iqr=2.0)
        result, _ = plan.process(df)
        # (value - median) / iqr = (1-3)/2=-1, (2-3)/2=-0.5, etc.
        expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
        assert result["a"].to_list() == expected

    def test_robust_scale_derive_from_data(self) -> None:
        """Test robust scaling deriving params from data."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        plan = TransformPlan().math_robust_scale("a")
        result, _ = plan.process(df)
        # Median of [1,2,3,4,5] = 3
        # Q1 = 2, Q3 = 4, IQR = 2
        # Result: (value - 3) / 2
        expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
        assert result["a"].to_list() == expected

    def test_robust_scale_to_new_column(self) -> None:
        """Test robust scaling to a new column."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        plan = TransformPlan().math_robust_scale(
            "a", median=3.0, iqr=2.0, new_column="a_robust"
        )
        result, _ = plan.process(df)
        assert "a" in result.columns
        assert "a_robust" in result.columns
        assert result["a"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_robust_scale_zero_iqr(self) -> None:
        """Test robust scale when IQR is zero."""
        df = pl.DataFrame({"a": [5.0, 5.0, 5.0, 5.0, 5.0]})
        plan = TransformPlan().math_robust_scale("a")
        result, _ = plan.process(df)
        # Should return zeros
        assert result["a"].to_list() == [0.0, 0.0, 0.0, 0.0, 0.0]


class TestMathLog:
    """Tests for math_log operation."""

    def test_log_natural(self) -> None:
        """Test natural log."""
        import math as pymath

        df = pl.DataFrame({"a": [1.0, pymath.e, pymath.e**2]})
        plan = TransformPlan().math_log("a")
        result, _ = plan.process(df)
        assert abs(result["a"][0] - 0.0) < 0.001
        assert abs(result["a"][1] - 1.0) < 0.001
        assert abs(result["a"][2] - 2.0) < 0.001

    def test_log_base_10(self) -> None:
        """Test log base 10."""
        df = pl.DataFrame({"a": [1.0, 10.0, 100.0, 1000.0]})
        plan = TransformPlan().math_log("a", base=10)
        result, _ = plan.process(df)
        expected = [0.0, 1.0, 2.0, 3.0]
        for r, e in zip(result["a"], expected):
            assert abs(r - e) < 0.001

    def test_log_with_offset(self) -> None:
        """Test log with offset for handling zeros."""
        df = pl.DataFrame({"a": [0.0, 1.0, 2.0]})
        plan = TransformPlan().math_log("a", offset=1)
        result, _ = plan.process(df)
        # log(0+1) = 0, log(1+1) = log(2), log(2+1) = log(3)
        import math as pymath

        assert abs(result["a"][0] - 0.0) < 0.001
        assert abs(result["a"][1] - pymath.log(2)) < 0.001
        assert abs(result["a"][2] - pymath.log(3)) < 0.001

    def test_log_to_new_column(self) -> None:
        """Test log transform to new column."""
        df = pl.DataFrame({"a": [1.0, 10.0, 100.0]})
        plan = TransformPlan().math_log("a", base=10, new_column="a_log")
        result, _ = plan.process(df)
        assert "a" in result.columns
        assert "a_log" in result.columns
        assert result["a"].to_list() == [1.0, 10.0, 100.0]

    def test_log_custom_base(self) -> None:
        """Test log with custom base."""
        df = pl.DataFrame({"a": [1.0, 2.0, 4.0, 8.0]})
        plan = TransformPlan().math_log("a", base=2)
        result, _ = plan.process(df)
        expected = [0.0, 1.0, 2.0, 3.0]
        for r, e in zip(result["a"], expected):
            assert abs(r - e) < 0.001


class TestMathSqrt:
    """Tests for math_sqrt operation."""

    def test_sqrt_basic(self) -> None:
        """Test basic square root."""
        df = pl.DataFrame({"a": [0.0, 1.0, 4.0, 9.0, 16.0]})
        plan = TransformPlan().math_sqrt("a")
        result, _ = plan.process(df)
        expected = [0.0, 1.0, 2.0, 3.0, 4.0]
        for r, e in zip(result["a"], expected):
            assert abs(r - e) < 0.001

    def test_sqrt_to_new_column(self) -> None:
        """Test sqrt to new column."""
        df = pl.DataFrame({"a": [1.0, 4.0, 9.0]})
        plan = TransformPlan().math_sqrt("a", new_column="a_sqrt")
        result, _ = plan.process(df)
        assert "a" in result.columns
        assert "a_sqrt" in result.columns
        assert result["a"].to_list() == [1.0, 4.0, 9.0]


class TestMathPower:
    """Tests for math_power operation."""

    def test_power_square(self) -> None:
        """Test squaring values."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        plan = TransformPlan().math_power("a", 2)
        result, _ = plan.process(df)
        expected = [1.0, 4.0, 9.0, 16.0, 25.0]
        assert result["a"].to_list() == expected

    def test_power_cube(self) -> None:
        """Test cubing values."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        plan = TransformPlan().math_power("a", 3)
        result, _ = plan.process(df)
        expected = [1.0, 8.0, 27.0]
        assert result["a"].to_list() == expected

    def test_power_sqrt_via_half(self) -> None:
        """Test square root via power 0.5."""
        df = pl.DataFrame({"a": [1.0, 4.0, 9.0, 16.0]})
        plan = TransformPlan().math_power("a", 0.5)
        result, _ = plan.process(df)
        expected = [1.0, 2.0, 3.0, 4.0]
        for r, e in zip(result["a"], expected):
            assert abs(r - e) < 0.001

    def test_power_to_new_column(self) -> None:
        """Test power to new column."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        plan = TransformPlan().math_power("a", 2, new_column="a_squared")
        result, _ = plan.process(df)
        assert "a" in result.columns
        assert "a_squared" in result.columns
        assert result["a"].to_list() == [1.0, 2.0, 3.0]


class TestMathWinsorize:
    """Tests for math_winsorize operation (outlier handling)."""

    def test_winsorize_percentile_based(self) -> None:
        """Test winsorization with percentile bounds."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        plan = TransformPlan().math_winsorize("a", lower=0.1, upper=0.9)
        result, _ = plan.process(df)
        # Values below 10th percentile clipped up, above 90th clipped down
        assert min(result["a"]) >= df["a"].quantile(0.1)
        assert max(result["a"]) <= df["a"].quantile(0.9)

    def test_winsorize_value_based(self) -> None:
        """Test winsorization with explicit bounds."""
        df = pl.DataFrame({"a": [1.0, 5.0, 10.0, 50.0, 100.0]})
        plan = TransformPlan().math_winsorize("a", lower_value=5.0, upper_value=50.0)
        result, _ = plan.process(df)
        assert result["a"].to_list() == [5.0, 5.0, 10.0, 50.0, 50.0]

    def test_winsorize_lower_only(self) -> None:
        """Test winsorization with only lower bound."""
        df = pl.DataFrame({"a": [1.0, 5.0, 10.0, 50.0, 100.0]})
        plan = TransformPlan().math_winsorize("a", lower_value=5.0)
        result, _ = plan.process(df)
        assert result["a"].to_list() == [5.0, 5.0, 10.0, 50.0, 100.0]

    def test_winsorize_upper_only(self) -> None:
        """Test winsorization with only upper bound."""
        df = pl.DataFrame({"a": [1.0, 5.0, 10.0, 50.0, 100.0]})
        plan = TransformPlan().math_winsorize("a", upper_value=50.0)
        result, _ = plan.process(df)
        assert result["a"].to_list() == [1.0, 5.0, 10.0, 50.0, 50.0]

    def test_winsorize_to_new_column(self) -> None:
        """Test winsorization to new column."""
        df = pl.DataFrame({"a": [1.0, 5.0, 10.0, 50.0, 100.0]})
        plan = TransformPlan().math_winsorize(
            "a", lower_value=5.0, upper_value=50.0, new_column="a_win"
        )
        result, _ = plan.process(df)
        assert "a" in result.columns
        assert "a_win" in result.columns
        assert result["a"].to_list() == [1.0, 5.0, 10.0, 50.0, 100.0]

    def test_winsorize_mixed_bounds(self) -> None:
        """Test winsorization with value for lower and percentile for upper."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]})
        plan = TransformPlan().math_winsorize("a", lower_value=2.0, upper=0.9)
        result, _ = plan.process(df)
        assert min(result["a"]) == 2.0
        # 90th percentile should clip the outlier
        assert max(result["a"]) < 100.0


class TestScalingEdgeCases:
    """Tests for edge cases across scaling operations."""

    def test_standardize_with_nulls(self) -> None:
        """Test standardization preserves nulls."""
        df = pl.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        plan = TransformPlan().math_standardize("a", mean=3.0, std=2.0)
        result, _ = plan.process(df)
        assert result["a"][1] is None
        assert result["a"][3] is None
        assert result["a"][0] == -1.0  # (1-3)/2
        assert result["a"][2] == 0.0  # (3-3)/2
        assert result["a"][4] == 1.0  # (5-3)/2

    def test_empty_dataframe(self, empty_df: pl.DataFrame) -> None:
        """Test operations on empty dataframe."""
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)})
        plan = TransformPlan().math_standardize("a", mean=0, std=1)
        result, _ = plan.process(df)
        assert len(result) == 0

    def test_single_row(self, single_row_df: pl.DataFrame) -> None:
        """Test operations on single row."""
        df = pl.DataFrame({"a": [5.0]})
        plan = TransformPlan().math_standardize("a")
        result, _ = plan.process(df)
        # Single value has std=0, should return 0
        assert result["a"][0] == 0.0


class TestScalingChaining:
    """Tests for chaining scaling operations."""

    def test_chain_multiple_transforms(self) -> None:
        """Test chaining multiple transform operations."""
        df = pl.DataFrame({"a": [1.0, 4.0, 9.0, 16.0, 25.0]})
        plan = (
            TransformPlan()
            .math_sqrt("a", new_column="a_sqrt")
            .math_standardize("a_sqrt", mean=3.0, std=1.0, new_column="a_z")
        )
        result, _ = plan.process(df)
        # sqrt gives [1, 2, 3, 4, 5], standardize gives [-2, -1, 0, 1, 2]
        assert "a" in result.columns
        assert "a_sqrt" in result.columns
        assert "a_z" in result.columns
        expected_z = [-2.0, -1.0, 0.0, 1.0, 2.0]
        for r, e in zip(result["a_z"], expected_z):
            assert abs(r - e) < 0.001

    def test_chain_with_winsorize(self) -> None:
        """Test chaining winsorize with standardize."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0]})  # 100 is outlier
        plan = (
            TransformPlan()
            .math_winsorize("a", upper_value=5.0, new_column="a_win")
            .math_standardize("a_win")
        )
        result, _ = plan.process(df)
        # After winsorize: [1, 2, 3, 4, 5]
        # After standardize: mean≈0, std≈1
        values = result["a_win"].to_list()
        mean_val = sum(v for v in values if v is not None) / len(values)
        assert abs(mean_val) < 0.001
        # Middle value should be ~0
        assert values[2] is not None
        assert abs(values[2]) < 0.001
