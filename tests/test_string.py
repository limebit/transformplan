"""Tests for string operations (ops/string.py)."""

import polars as pl

from transformplan import TransformPlan


class TestStrReplace:
    """Tests for str_replace operation."""

    def test_str_replace_literal(self, string_df: pl.DataFrame) -> None:
        """Test replacing a literal string."""
        plan = TransformPlan().str_replace("text", "World", "Universe")
        result, _ = plan.process(string_df)
        assert "Universe" in result["text"][0]
        assert "World" not in result["text"][0]

    def test_str_replace_all_occurrences(self) -> None:
        """Test that all occurrences are replaced."""
        df = pl.DataFrame({"text": ["foo foo foo"]})
        plan = TransformPlan().str_replace("text", "foo", "bar")
        result, _ = plan.process(df)
        assert result["text"][0] == "bar bar bar"

    def test_str_replace_regex(self, string_df: pl.DataFrame) -> None:
        """Test replacing with regex pattern."""
        plan = TransformPlan().str_replace("text", r"\d+", "NUM", literal=False)
        result, _ = plan.process(string_df)
        # ABC123xyz -> ABCNUMxyz
        assert "NUM" in result["text"][2]
        assert "123" not in result["text"][2]

    def test_str_replace_nonexistent_column_raises(
        self, string_df: pl.DataFrame
    ) -> None:
        """Test that replacing in nonexistent column fails."""
        plan = TransformPlan().str_replace("nonexistent", "a", "b")
        result = plan.validate(string_df)
        assert not result.is_valid

    def test_str_replace_non_string_column_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that replacing in non-string column fails."""
        plan = TransformPlan().str_replace("age", "a", "b")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "expected string" in str(result.errors[0])


class TestStrSlice:
    """Tests for str_slice operation."""

    def test_str_slice_from_start(self, string_df: pl.DataFrame) -> None:
        """Test slicing from start."""
        plan = TransformPlan().str_slice("code", 0, 3)
        result, _ = plan.process(string_df)
        assert result["code"][0] == "PRD"
        assert result["code"][2] == "TST"

    def test_str_slice_from_offset(self, string_df: pl.DataFrame) -> None:
        """Test slicing with offset."""
        plan = TransformPlan().str_slice("code", 4)
        result, _ = plan.process(string_df)
        assert result["code"][0] == "001"

    def test_str_slice_negative_offset(self) -> None:
        """Test slicing with negative offset."""
        df = pl.DataFrame({"text": ["hello world"]})
        plan = TransformPlan().str_slice("text", -5)
        result, _ = plan.process(df)
        assert result["text"][0] == "world"


class TestStrTruncate:
    """Tests for str_truncate operation."""

    def test_str_truncate_basic(self, string_df: pl.DataFrame) -> None:
        """Test basic truncation."""
        plan = TransformPlan().str_truncate("text", 10)
        result, _ = plan.process(string_df)
        assert len(result["text"][0]) <= 10
        # Strings longer than 10 should end with ...
        if len(string_df["text"][0]) > 10:
            assert result["text"][0].endswith("...")

    def test_str_truncate_custom_suffix(self) -> None:
        """Test truncation with custom suffix."""
        df = pl.DataFrame({"text": ["This is a very long string"]})
        plan = TransformPlan().str_truncate("text", 15, suffix="~")
        result, _ = plan.process(df)
        assert result["text"][0].endswith("~")
        assert len(result["text"][0]) <= 15

    def test_str_truncate_no_change_short_string(self) -> None:
        """Test that short strings are unchanged."""
        df = pl.DataFrame({"text": ["short"]})
        plan = TransformPlan().str_truncate("text", 20)
        result, _ = plan.process(df)
        assert result["text"][0] == "short"


class TestStrLower:
    """Tests for str_lower operation."""

    def test_str_lower_basic(self, string_df: pl.DataFrame) -> None:
        """Test converting to lowercase."""
        plan = TransformPlan().str_lower("first_name")
        result, _ = plan.process(string_df)
        assert result["first_name"].to_list() == [
            "john",
            "jane",
            "bob",
            "alice",
            "charlie",
        ]

    def test_str_lower_mixed_case(self) -> None:
        """Test with mixed case input."""
        df = pl.DataFrame({"text": ["HELLO", "World", "MiXeD"]})
        plan = TransformPlan().str_lower("text")
        result, _ = plan.process(df)
        assert result["text"].to_list() == ["hello", "world", "mixed"]


class TestStrUpper:
    """Tests for str_upper operation."""

    def test_str_upper_basic(self, string_df: pl.DataFrame) -> None:
        """Test converting to uppercase."""
        plan = TransformPlan().str_upper("first_name")
        result, _ = plan.process(string_df)
        assert result["first_name"].to_list() == [
            "JOHN",
            "JANE",
            "BOB",
            "ALICE",
            "CHARLIE",
        ]


class TestStrStrip:
    """Tests for str_strip operation."""

    def test_str_strip_whitespace(self, string_df: pl.DataFrame) -> None:
        """Test stripping whitespace."""
        plan = TransformPlan().str_strip("text")
        result, _ = plan.process(string_df)
        assert result["text"][0] == "Hello World"  # Was "  Hello World  "

    def test_str_strip_custom_chars(self) -> None:
        """Test stripping custom characters."""
        df = pl.DataFrame({"text": ["###hello###", "##world##"]})
        plan = TransformPlan().str_strip("text", chars="#")
        result, _ = plan.process(df)
        assert result["text"][0] == "hello"
        assert result["text"][1] == "world"


class TestStrPad:
    """Tests for str_pad operation."""

    def test_str_pad_left(self) -> None:
        """Test left padding."""
        df = pl.DataFrame({"num": ["1", "22", "333"]})
        plan = TransformPlan().str_pad("num", length=5, fill_char="0", side="left")
        result, _ = plan.process(df)
        assert result["num"].to_list() == ["00001", "00022", "00333"]

    def test_str_pad_right(self) -> None:
        """Test right padding."""
        df = pl.DataFrame({"text": ["a", "bb", "ccc"]})
        plan = TransformPlan().str_pad("text", length=5, fill_char=".", side="right")
        result, _ = plan.process(df)
        assert result["text"].to_list() == ["a....", "bb...", "ccc.."]

    def test_str_pad_no_change_already_long(self) -> None:
        """Test that already-long strings are unchanged."""
        df = pl.DataFrame({"text": ["hello world"]})
        plan = TransformPlan().str_pad("text", length=5, fill_char="X")
        result, _ = plan.process(df)
        assert result["text"][0] == "hello world"


class TestStrSplit:
    """Tests for str_split operation."""

    def test_str_split_into_columns(self, string_df: pl.DataFrame) -> None:
        """Test splitting into multiple columns."""
        plan = TransformPlan().str_split("code", "-", new_columns=["prefix", "number"])
        result, _ = plan.process(string_df)
        assert "prefix" in result.columns
        assert "number" in result.columns
        assert result["prefix"][0] == "PRD"
        assert result["number"][0] == "001"
        assert "code" not in result.columns  # Original dropped by default

    def test_str_split_keep_original(self, string_df: pl.DataFrame) -> None:
        """Test splitting while keeping original column."""
        plan = TransformPlan().str_split(
            "code", "-", new_columns=["prefix", "number"], keep_original=True
        )
        result, _ = plan.process(string_df)
        assert "code" in result.columns
        assert "prefix" in result.columns

    def test_str_split_explode(self) -> None:
        """Test splitting and exploding into rows."""
        df = pl.DataFrame({"tags": ["a,b,c", "d,e"]})
        plan = TransformPlan().str_split("tags", ",")
        result, _ = plan.process(df)
        # Should explode into 5 rows
        assert len(result) == 5


class TestStrConcat:
    """Tests for str_concat operation."""

    def test_str_concat_basic(self, string_df: pl.DataFrame) -> None:
        """Test concatenating columns."""
        plan = TransformPlan().str_concat(
            ["first_name", "last_name"], "full_name", separator=" "
        )
        result, _ = plan.process(string_df)
        assert "full_name" in result.columns
        assert result["full_name"][0] == "John Doe"

    def test_str_concat_no_separator(self, string_df: pl.DataFrame) -> None:
        """Test concatenating without separator."""
        plan = TransformPlan().str_concat(["first_name", "last_name"], "full_name")
        result, _ = plan.process(string_df)
        assert result["full_name"][0] == "JohnDoe"

    def test_str_concat_multiple_columns(self) -> None:
        """Test concatenating multiple columns."""
        df = pl.DataFrame(
            {
                "a": ["A", "D"],
                "b": ["B", "E"],
                "c": ["C", "F"],
            }
        )
        plan = TransformPlan().str_concat(["a", "b", "c"], "combined", separator="-")
        result, _ = plan.process(df)
        assert result["combined"].to_list() == ["A-B-C", "D-E-F"]


class TestStrExtract:
    """Tests for str_extract operation."""

    def test_str_extract_basic(self) -> None:
        """Test extracting with regex."""
        df = pl.DataFrame({"text": ["order-123", "order-456", "order-789"]})
        plan = TransformPlan().str_extract("text", r"order-(\d+)", group_index=1)
        result, _ = plan.process(df)
        assert result["text"].to_list() == ["123", "456", "789"]

    def test_str_extract_new_column(self) -> None:
        """Test extracting to new column."""
        df = pl.DataFrame({"email": ["user@example.com", "admin@test.org"]})
        plan = TransformPlan().str_extract(
            "email", r"@(\w+)\.", group_index=1, new_column="domain"
        )
        result, _ = plan.process(df)
        assert "domain" in result.columns
        assert "email" in result.columns  # Original preserved
        assert result["domain"].to_list() == ["example", "test"]

    def test_str_extract_no_match(self) -> None:
        """Test extraction when pattern doesn't match."""
        df = pl.DataFrame({"text": ["no-numbers-here"]})
        plan = TransformPlan().str_extract("text", r"(\d+)", group_index=1)
        result, _ = plan.process(df)
        assert result["text"][0] is None


class TestStringChaining:
    """Tests for chaining multiple string operations."""

    def test_multiple_string_operations(self, string_df: pl.DataFrame) -> None:
        """Test chaining multiple string operations."""
        plan = (
            TransformPlan()
            .str_strip("text")
            .str_lower("text")
            .str_replace("text", " ", "_")
        )
        result, _ = plan.process(string_df)
        assert result["text"][0] == "hello_world"

    def test_concat_then_transform(self, string_df: pl.DataFrame) -> None:
        """Test concatenating then transforming."""
        plan = (
            TransformPlan()
            .str_concat(["first_name", "last_name"], "full_name", separator=" ")
            .str_upper("full_name")
        )
        result, _ = plan.process(string_df)
        assert result["full_name"][0] == "JOHN DOE"


class TestStringEdgeCases:
    """Tests for edge cases in string operations."""

    def test_empty_string(self) -> None:
        """Test operations on empty string."""
        df = pl.DataFrame({"text": ["", "hello", ""]})
        plan = TransformPlan().str_upper("text")
        result, _ = plan.process(df)
        assert result["text"][0] == ""

    def test_unicode_string(self) -> None:
        """Test operations on unicode strings."""
        df = pl.DataFrame({"text": ["héllo", "wörld", "日本語"]})
        plan = TransformPlan().str_upper("text")
        result, _ = plan.process(df)
        assert result["text"][0] == "HÉLLO"
        assert result["text"][1] == "WÖRLD"
