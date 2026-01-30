"""Tests for encoding operations (ops/encoding.py)."""

import polars as pl
import pytest

from transformplan import TransformPlan


@pytest.fixture
def encoding_df() -> pl.DataFrame:
    """DataFrame for encoding operations."""
    return pl.DataFrame(
        {
            "color": ["red", "green", "blue", "red", "green"],
            "size": ["small", "medium", "large", "medium", "small"],
            "department": ["HR", "Engineering", "Sales", "HR", "Engineering"],
        }
    )


class TestEncOnehot:
    """Tests for enc_onehot operation."""

    def test_enc_onehot_with_categories(self, encoding_df: pl.DataFrame) -> None:
        """Test one-hot encoding with explicit categories."""
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green", "blue"]
        )
        result, _ = plan.process(encoding_df)

        # Check columns created
        assert "color_red" in result.columns
        assert "color_green" in result.columns
        assert "color_blue" in result.columns

        # Original column dropped by default
        assert "color" not in result.columns

        # Check values (row 0 is "red")
        assert result["color_red"][0] == 1
        assert result["color_green"][0] == 0
        assert result["color_blue"][0] == 0

        # Row 1 is "green"
        assert result["color_red"][1] == 0
        assert result["color_green"][1] == 1
        assert result["color_blue"][1] == 0

    def test_enc_onehot_derive_categories(self, encoding_df: pl.DataFrame) -> None:
        """Test one-hot encoding deriving categories from data."""
        plan = TransformPlan().enc_onehot("color")
        result, _ = plan.process(encoding_df)

        # Should derive categories alphabetically: blue, green, red
        assert "color_blue" in result.columns
        assert "color_green" in result.columns
        assert "color_red" in result.columns

    def test_enc_onehot_custom_prefix(self, encoding_df: pl.DataFrame) -> None:
        """Test one-hot encoding with custom prefix."""
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green"], prefix="c"
        )
        result, _ = plan.process(encoding_df)

        assert "c_red" in result.columns
        assert "c_green" in result.columns

    def test_enc_onehot_keep_original(self, encoding_df: pl.DataFrame) -> None:
        """Test one-hot encoding keeping original column."""
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green"], drop_original=False
        )
        result, _ = plan.process(encoding_df)

        assert "color" in result.columns
        assert "color_red" in result.columns
        assert "color_green" in result.columns

    def test_enc_onehot_unknown_all_zero(self, encoding_df: pl.DataFrame) -> None:
        """Test one-hot encoding with unknown values set to all zeros."""
        # Only include some categories - "blue" will be unknown
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green"], unknown_value="all_zero"
        )
        result, _ = plan.process(encoding_df)

        # Row 2 is "blue" which is unknown
        assert result["color_red"][2] == 0
        assert result["color_green"][2] == 0

    def test_enc_onehot_unknown_ignore(self, encoding_df: pl.DataFrame) -> None:
        """Test one-hot encoding with unknown values returning null."""
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green"], unknown_value="ignore"
        )
        result, _ = plan.process(encoding_df)

        # Row 2 is "blue" which is unknown - should get null
        assert result["color_red"][2] is None
        assert result["color_green"][2] is None

    def test_enc_onehot_drop_first(self, encoding_df: pl.DataFrame) -> None:
        """Test one-hot encoding with drop='first' to avoid multicollinearity."""
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green", "blue"], drop="first"
        )
        result, _ = plan.process(encoding_df)

        # "red" (first category) should be dropped
        assert "color_red" not in result.columns
        assert "color_green" in result.columns
        assert "color_blue" in result.columns

        # Row 0 is "red" - should have 0 for green and blue
        assert result["color_green"][0] == 0
        assert result["color_blue"][0] == 0

        # Row 1 is "green"
        assert result["color_green"][1] == 1
        assert result["color_blue"][1] == 0

    def test_enc_onehot_drop_last(self, encoding_df: pl.DataFrame) -> None:
        """Test one-hot encoding with drop='last'."""
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green", "blue"], drop="last"
        )
        result, _ = plan.process(encoding_df)

        # "blue" (last category) should be dropped
        assert "color_red" in result.columns
        assert "color_green" in result.columns
        assert "color_blue" not in result.columns

        # Row 2 is "blue" - should have 0 for red and green
        assert result["color_red"][2] == 0
        assert result["color_green"][2] == 0

    def test_enc_onehot_drop_specific_value(self, encoding_df: pl.DataFrame) -> None:
        """Test one-hot encoding dropping a specific category value."""
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green", "blue"], drop="green"
        )
        result, _ = plan.process(encoding_df)

        # "green" should be dropped
        assert "color_red" in result.columns
        assert "color_green" not in result.columns
        assert "color_blue" in result.columns

        # Row 1 is "green" - should have 0 for red and blue
        assert result["color_red"][1] == 0
        assert result["color_blue"][1] == 0

    def test_enc_onehot_drop_with_derived_categories(self) -> None:
        """Test one-hot encoding with drop and categories derived from data."""
        df = pl.DataFrame({"color": ["red", "green", "blue"]})
        plan = TransformPlan().enc_onehot("color", drop="first")
        result, _ = plan.process(df)

        # Categories are derived alphabetically: blue, green, red
        # "blue" (first alphabetically) should be dropped
        assert "color_blue" not in result.columns
        assert "color_green" in result.columns
        assert "color_red" in result.columns

    def test_enc_onehot_drop_literal_takes_precedence(self) -> None:
        """Test that literal values take precedence over 'first'/'last' keywords."""
        # Category list where "first" is NOT the first element
        df = pl.DataFrame({"pos": ["last", "middle", "first"]})
        plan = TransformPlan().enc_onehot(
            "pos", categories=["last", "middle", "first"], drop="first"
        )
        result, _ = plan.process(df)

        # "first" should be interpreted as the literal value, not positional
        # So "first" (the value) should be dropped, NOT "last" (the first position)
        assert "pos_last" in result.columns
        assert "pos_middle" in result.columns
        assert "pos_first" not in result.columns

    def test_enc_onehot_drop_keyword_when_not_in_categories(self) -> None:
        """Test that 'first'/'last' work as keywords when not in categories."""
        df = pl.DataFrame({"color": ["red", "green", "blue"]})
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green", "blue"], drop="first"
        )
        result, _ = plan.process(df)

        # "first" is not in categories, so it's interpreted as keyword
        # Should drop "red" (first in list)
        assert "color_red" not in result.columns
        assert "color_green" in result.columns
        assert "color_blue" in result.columns


class TestEncOrdinal:
    """Tests for enc_ordinal operation."""

    def test_enc_ordinal_with_categories(self, encoding_df: pl.DataFrame) -> None:
        """Test ordinal encoding with explicit ordering."""
        plan = TransformPlan().enc_ordinal(
            "size", categories=["small", "medium", "large"]
        )
        result, _ = plan.process(encoding_df)

        # small=0, medium=1, large=2
        expected = [0, 1, 2, 1, 0]
        assert result["size"].to_list() == expected

    def test_enc_ordinal_derive_categories(self, encoding_df: pl.DataFrame) -> None:
        """Test ordinal encoding deriving categories alphabetically."""
        plan = TransformPlan().enc_ordinal("size")
        result, _ = plan.process(encoding_df)

        # Alphabetically: large=0, medium=1, small=2
        # Original: ["small", "medium", "large", "medium", "small"]
        expected = [2, 1, 0, 1, 2]
        assert result["size"].to_list() == expected

    def test_enc_ordinal_new_column(self, encoding_df: pl.DataFrame) -> None:
        """Test ordinal encoding to new column."""
        plan = TransformPlan().enc_ordinal(
            "size",
            categories=["small", "medium", "large"],
            new_column="size_encoded",
        )
        result, _ = plan.process(encoding_df)

        assert "size_encoded" in result.columns
        # Original dropped by default when new_column differs
        assert "size" not in result.columns

        expected = [0, 1, 2, 1, 0]
        assert result["size_encoded"].to_list() == expected

    def test_enc_ordinal_keep_original(self, encoding_df: pl.DataFrame) -> None:
        """Test ordinal encoding keeping original column."""
        plan = TransformPlan().enc_ordinal(
            "size",
            categories=["small", "medium", "large"],
            new_column="size_encoded",
            drop_original=False,
        )
        result, _ = plan.process(encoding_df)

        assert "size" in result.columns
        assert "size_encoded" in result.columns

    def test_enc_ordinal_unknown_value(self) -> None:
        """Test ordinal encoding with unknown values."""
        df = pl.DataFrame({"size": ["small", "medium", "xl"]})
        plan = TransformPlan().enc_ordinal(
            "size", categories=["small", "medium", "large"], unknown_value=-1
        )
        result, _ = plan.process(df)

        # "xl" is unknown
        assert result["size"].to_list() == [0, 1, -1]

    def test_enc_ordinal_custom_unknown_value(self) -> None:
        """Test ordinal encoding with custom unknown value."""
        df = pl.DataFrame({"size": ["small", "unknown"]})
        plan = TransformPlan().enc_ordinal(
            "size", categories=["small", "medium"], unknown_value=99
        )
        result, _ = plan.process(df)

        assert result["size"].to_list() == [0, 99]


class TestEncLabel:
    """Tests for enc_label operation."""

    def test_enc_label_with_categories(self, encoding_df: pl.DataFrame) -> None:
        """Test label encoding with explicit categories."""
        plan = TransformPlan().enc_label(
            "department", categories=["HR", "Engineering", "Sales"]
        )
        result, _ = plan.process(encoding_df)

        # HR=0, Engineering=1, Sales=2
        expected = [0, 1, 2, 0, 1]
        assert result["department"].to_list() == expected

    def test_enc_label_derive_categories(self, encoding_df: pl.DataFrame) -> None:
        """Test label encoding deriving categories alphabetically."""
        plan = TransformPlan().enc_label("department")
        result, _ = plan.process(encoding_df)

        # Alphabetically: Engineering=0, HR=1, Sales=2
        # Original: ["HR", "Engineering", "Sales", "HR", "Engineering"]
        expected = [1, 0, 2, 1, 0]
        assert result["department"].to_list() == expected

    def test_enc_label_new_column(self, encoding_df: pl.DataFrame) -> None:
        """Test label encoding to new column."""
        plan = TransformPlan().enc_label(
            "department",
            categories=["HR", "Engineering", "Sales"],
            new_column="dept_id",
        )
        result, _ = plan.process(encoding_df)

        assert "dept_id" in result.columns
        assert "department" not in result.columns

    def test_enc_label_unknown_value(self) -> None:
        """Test label encoding with unknown values."""
        df = pl.DataFrame({"dept": ["HR", "Marketing"]})
        plan = TransformPlan().enc_label(
            "dept", categories=["HR", "Engineering"], unknown_value=-1
        )
        result, _ = plan.process(df)

        assert result["dept"].to_list() == [0, -1]


class TestEncodingValidation:
    """Tests for encoding validation errors."""

    def test_enc_onehot_missing_column(self, encoding_df: pl.DataFrame) -> None:
        """Test validation error for missing column."""
        plan = TransformPlan().enc_onehot("nonexistent")
        result = plan.validate(encoding_df)

        assert not result.is_valid
        assert any("nonexistent" in str(e) for e in result.errors)

    def test_enc_onehot_duplicate_categories(self, encoding_df: pl.DataFrame) -> None:
        """Test validation error for duplicate categories."""
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "red", "blue"]
        )
        result = plan.validate(encoding_df)

        assert not result.is_valid
        assert any("Duplicate" in str(e) for e in result.errors)

    def test_enc_onehot_column_collision(self) -> None:
        """Test validation error for column name collision."""
        df = pl.DataFrame({"color": ["red"], "color_red": [1]})
        plan = TransformPlan().enc_onehot("color", categories=["red"])
        result = plan.validate(df)

        assert not result.is_valid
        assert any("already exists" in str(e) for e in result.errors)

    def test_enc_onehot_drop_invalid_value(self, encoding_df: pl.DataFrame) -> None:
        """Test validation error for drop value not in categories."""
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green", "blue"], drop="purple"
        )
        result = plan.validate(encoding_df)

        assert not result.is_valid
        assert any("not in categories" in str(e) for e in result.errors)

    def test_enc_onehot_drop_avoids_collision(self) -> None:
        """Test that drop='first' avoids column collision when first column exists."""
        # color_blue already exists, but we're dropping blue (first alphabetically)
        df = pl.DataFrame({"color": ["red", "green", "blue"], "color_blue": [1, 2, 3]})
        plan = TransformPlan().enc_onehot(
            "color", categories=["blue", "green", "red"], drop="first"
        )
        result = plan.validate(df)

        # Should be valid because we're dropping color_blue
        assert result.is_valid

    def test_enc_ordinal_missing_column(self, encoding_df: pl.DataFrame) -> None:
        """Test validation error for missing column."""
        plan = TransformPlan().enc_ordinal("nonexistent")
        result = plan.validate(encoding_df)

        assert not result.is_valid
        assert any("nonexistent" in str(e) for e in result.errors)

    def test_enc_ordinal_duplicate_categories(self, encoding_df: pl.DataFrame) -> None:
        """Test validation error for duplicate categories."""
        plan = TransformPlan().enc_ordinal(
            "size", categories=["small", "small", "large"]
        )
        result = plan.validate(encoding_df)

        assert not result.is_valid
        assert any("Duplicate" in str(e) for e in result.errors)

    def test_enc_label_missing_column(self, encoding_df: pl.DataFrame) -> None:
        """Test validation error for missing column."""
        plan = TransformPlan().enc_label("nonexistent")
        result = plan.validate(encoding_df)

        assert not result.is_valid

    def test_enc_label_duplicate_categories(self, encoding_df: pl.DataFrame) -> None:
        """Test validation error for duplicate categories."""
        plan = TransformPlan().enc_label(
            "department", categories=["HR", "HR"]
        )
        result = plan.validate(encoding_df)

        assert not result.is_valid
        assert any("Duplicate" in str(e) for e in result.errors)


class TestEncodingEdgeCases:
    """Tests for edge cases in encoding operations."""

    def test_enc_onehot_with_nulls(self) -> None:
        """Test one-hot encoding with null values."""
        df = pl.DataFrame({"color": ["red", None, "blue"]})
        plan = TransformPlan().enc_onehot("color", categories=["red", "blue"])
        result, _ = plan.process(df)

        # Null should be treated as unknown (all zeros with default setting)
        assert result["color_red"][1] == 0
        assert result["color_blue"][1] == 0

    def test_enc_onehot_empty_dataframe(self, empty_df: pl.DataFrame) -> None:
        """Test one-hot encoding with empty DataFrame."""
        df = pl.DataFrame({"color": pl.Series([], dtype=pl.Utf8)})
        plan = TransformPlan().enc_onehot("color", categories=["red", "blue"])
        result, _ = plan.process(df)

        assert "color_red" in result.columns
        assert "color_blue" in result.columns
        assert len(result) == 0

    def test_enc_onehot_single_category(self) -> None:
        """Test one-hot encoding with single category."""
        df = pl.DataFrame({"status": ["active", "active", "active"]})
        plan = TransformPlan().enc_onehot("status", categories=["active"])
        result, _ = plan.process(df)

        assert "status_active" in result.columns
        assert result["status_active"].to_list() == [1, 1, 1]

    def test_enc_ordinal_with_nulls(self) -> None:
        """Test ordinal encoding with null values."""
        df = pl.DataFrame({"size": ["small", None, "large"]})
        plan = TransformPlan().enc_ordinal(
            "size", categories=["small", "medium", "large"], unknown_value=-1
        )
        result, _ = plan.process(df)

        # Null is treated as unknown
        assert result["size"].to_list() == [0, -1, 2]

    def test_enc_ordinal_empty_categories(self) -> None:
        """Test ordinal encoding with empty categories list."""
        df = pl.DataFrame({"size": ["small", "medium"]})
        plan = TransformPlan().enc_ordinal("size", categories=[], unknown_value=-1)
        result, _ = plan.process(df)

        # All values should be unknown
        assert result["size"].to_list() == [-1, -1]


class TestEncodingChaining:
    """Tests for chaining encoding operations."""

    def test_multiple_encodings(self, encoding_df: pl.DataFrame) -> None:
        """Test chaining multiple encoding operations."""
        plan = (
            TransformPlan()
            .enc_onehot("color", categories=["red", "green", "blue"])
            .enc_ordinal("size", categories=["small", "medium", "large"])
        )
        result, _ = plan.process(encoding_df)

        # Check both encodings applied
        assert "color_red" in result.columns
        # Polars uses Int32 for small integer literals
        assert result["size"].dtype in (pl.Int32, pl.Int64)

    def test_encoding_with_other_ops(self, encoding_df: pl.DataFrame) -> None:
        """Test encoding combined with other operations."""
        plan = (
            TransformPlan()
            .enc_ordinal(
                "size",
                categories=["small", "medium", "large"],
                new_column="size_encoded",
                drop_original=False,
            )
            .col_drop("department")
        )
        result, _ = plan.process(encoding_df)

        assert "size_encoded" in result.columns
        assert "size" in result.columns
        assert "department" not in result.columns


class TestEncodingProtocol:
    """Tests for encoding operations in the protocol/audit trail."""

    def test_enc_onehot_in_protocol(self, encoding_df: pl.DataFrame) -> None:
        """Test that one-hot encoding is recorded in protocol."""
        plan = TransformPlan().enc_onehot(
            "color", categories=["red", "green", "blue"]
        )
        _, protocol = plan.process(encoding_df)

        protocol_dict = protocol.to_dict()
        assert len(protocol_dict["steps"]) == 1
        step = protocol_dict["steps"][0]
        assert step["operation"] == "enc_onehot"
        assert step["params"]["column"] == "color"
        assert step["params"]["categories"] == ["red", "green", "blue"]

    def test_enc_ordinal_in_protocol(self, encoding_df: pl.DataFrame) -> None:
        """Test that ordinal encoding is recorded in protocol."""
        plan = TransformPlan().enc_ordinal(
            "size", categories=["small", "medium", "large"]
        )
        _, protocol = plan.process(encoding_df)

        protocol_dict = protocol.to_dict()
        step = protocol_dict["steps"][0]
        assert step["operation"] == "enc_ordinal"
        assert step["params"]["categories"] == ["small", "medium", "large"]
