"""Tests for filter expressions (filters.py)."""

import polars as pl
import pytest

from transformplan import Col
from transformplan.filters import (
    And,
    Between,
    Eq,
    Filter,
    Ge,
    Gt,
    IsIn,
    IsNotNull,
    IsNull,
    Le,
    Lt,
    Ne,
    Not,
    Or,
    StrContains,
    StrEndsWith,
    StrStartsWith,
    _sql_escape_like,
    _sql_format_value,
    _sql_quote_identifier,
)


class TestCol:
    """Tests for Col class."""

    def test_col_eq(self) -> None:
        """Test Col equality operator."""
        result = Col("name") == "Alice"
        assert isinstance(result, Eq)
        assert result.column == "name"
        assert result.value == "Alice"

    def test_col_ne(self) -> None:
        """Test Col inequality operator."""
        result = Col("name") != "Bob"
        assert isinstance(result, Ne)
        assert result.column == "name"
        assert result.value == "Bob"

    def test_col_gt(self) -> None:
        """Test Col greater than operator."""
        result = Col("age") > 18
        assert isinstance(result, Gt)
        assert result.column == "age"
        assert result.value == 18

    def test_col_ge(self) -> None:
        """Test Col greater than or equal operator."""
        result = Col("age") >= 18
        assert isinstance(result, Ge)
        assert result.column == "age"
        assert result.value == 18

    def test_col_lt(self) -> None:
        """Test Col less than operator."""
        result = Col("age") < 65
        assert isinstance(result, Lt)
        assert result.column == "age"
        assert result.value == 65

    def test_col_le(self) -> None:
        """Test Col less than or equal operator."""
        result = Col("age") <= 65
        assert isinstance(result, Le)
        assert result.column == "age"
        assert result.value == 65

    def test_col_is_in(self) -> None:
        """Test Col is_in method."""
        result = Col("status").is_in(["active", "pending"])
        assert isinstance(result, IsIn)
        assert result.column == "status"
        assert list(result.values) == ["active", "pending"]

    def test_col_is_null(self) -> None:
        """Test Col is_null method."""
        result = Col("name").is_null()
        assert isinstance(result, IsNull)
        assert result.column == "name"

    def test_col_is_not_null(self) -> None:
        """Test Col is_not_null method."""
        result = Col("name").is_not_null()
        assert isinstance(result, IsNotNull)
        assert result.column == "name"

    def test_col_str_contains(self) -> None:
        """Test Col str_contains method."""
        result = Col("email").str_contains("@example.com")
        assert isinstance(result, StrContains)
        assert result.column == "email"
        assert result.pattern == "@example.com"
        assert result.literal is True

    def test_col_str_contains_regex(self) -> None:
        """Test Col str_contains with regex."""
        result = Col("text").str_contains(r"\d+", literal=False)
        assert isinstance(result, StrContains)
        assert result.literal is False

    def test_col_str_starts_with(self) -> None:
        """Test Col str_starts_with method."""
        result = Col("code").str_starts_with("PRD-")
        assert isinstance(result, StrStartsWith)
        assert result.column == "code"
        assert result.prefix == "PRD-"

    def test_col_str_ends_with(self) -> None:
        """Test Col str_ends_with method."""
        result = Col("filename").str_ends_with(".csv")
        assert isinstance(result, StrEndsWith)
        assert result.column == "filename"
        assert result.suffix == ".csv"

    def test_col_between(self) -> None:
        """Test Col between method."""
        result = Col("age").between(18, 65)
        assert isinstance(result, Between)
        assert result.column == "age"
        assert result.lower == 18
        assert result.upper == 65


class TestComparisonFilters:
    """Tests for comparison filter classes."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "score": [80, 90, 85],
            }
        )

    def test_eq_filter(self, sample_df: pl.DataFrame) -> None:
        """Test Eq filter execution."""
        f = Eq("name", "Alice")
        result = sample_df.filter(f.to_expr())
        assert len(result) == 1
        assert result["name"][0] == "Alice"

    def test_ne_filter(self, sample_df: pl.DataFrame) -> None:
        """Test Ne filter execution."""
        f = Ne("name", "Alice")
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2
        assert "Alice" not in result["name"].to_list()

    def test_gt_filter(self, sample_df: pl.DataFrame) -> None:
        """Test Gt filter execution."""
        f = Gt("age", 25)
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2
        assert all(age > 25 for age in result["age"].to_list())

    def test_ge_filter(self, sample_df: pl.DataFrame) -> None:
        """Test Ge filter execution."""
        f = Ge("age", 30)
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2
        assert all(age >= 30 for age in result["age"].to_list())

    def test_lt_filter(self, sample_df: pl.DataFrame) -> None:
        """Test Lt filter execution."""
        f = Lt("age", 35)
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2
        assert all(age < 35 for age in result["age"].to_list())

    def test_le_filter(self, sample_df: pl.DataFrame) -> None:
        """Test Le filter execution."""
        f = Le("age", 30)
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2
        assert all(age <= 30 for age in result["age"].to_list())

    def test_is_in_filter(self, sample_df: pl.DataFrame) -> None:
        """Test IsIn filter execution."""
        f = IsIn("name", ["Alice", "Bob"])
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2
        assert set(result["name"].to_list()) == {"Alice", "Bob"}

    def test_between_filter(self, sample_df: pl.DataFrame) -> None:
        """Test Between filter execution."""
        f = Between("age", 25, 30)
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2
        assert all(25 <= age <= 30 for age in result["age"].to_list())


class TestNullFilters:
    """Tests for null check filters."""

    @pytest.fixture
    def df_with_nulls(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "name": ["Alice", None, "Charlie"],
                "age": [25, 30, None],
            }
        )

    def test_is_null_filter(self, df_with_nulls: pl.DataFrame) -> None:
        """Test IsNull filter execution."""
        f = IsNull("name")
        result = df_with_nulls.filter(f.to_expr())
        assert len(result) == 1
        assert result["name"][0] is None

    def test_is_not_null_filter(self, df_with_nulls: pl.DataFrame) -> None:
        """Test IsNotNull filter execution."""
        f = IsNotNull("name")
        result = df_with_nulls.filter(f.to_expr())
        assert len(result) == 2
        assert all(name is not None for name in result["name"].to_list())


class TestStringFilters:
    """Tests for string filters."""

    @pytest.fixture
    def string_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "email": ["alice@example.com", "bob@test.com", "charlie@example.org"],
                "code": ["PRD-001", "TST-002", "PRD-003"],
            }
        )

    def test_str_contains_filter(self, string_df: pl.DataFrame) -> None:
        """Test StrContains filter execution."""
        f = StrContains("email", "@example")
        result = string_df.filter(f.to_expr())
        assert len(result) == 2

    def test_str_contains_regex(self, string_df: pl.DataFrame) -> None:
        """Test StrContains filter with regex."""
        f = StrContains("code", r"\d{3}", literal=False)
        result = string_df.filter(f.to_expr())
        assert len(result) == 3

    def test_str_starts_with_filter(self, string_df: pl.DataFrame) -> None:
        """Test StrStartsWith filter execution."""
        f = StrStartsWith("code", "PRD")
        result = string_df.filter(f.to_expr())
        assert len(result) == 2

    def test_str_ends_with_filter(self, string_df: pl.DataFrame) -> None:
        """Test StrEndsWith filter execution."""
        f = StrEndsWith("email", ".com")
        result = string_df.filter(f.to_expr())
        assert len(result) == 2


class TestLogicalOperators:
    """Tests for logical operators (And, Or, Not)."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
                "active": [True, True, False, True],
            }
        )

    def test_and_filter(self, sample_df: pl.DataFrame) -> None:
        """Test And filter execution."""
        f = And(Ge("age", 30), Eq("active", True))
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2
        assert set(result["name"].to_list()) == {"Bob", "David"}

    def test_or_filter(self, sample_df: pl.DataFrame) -> None:
        """Test Or filter execution."""
        f = Or(Eq("name", "Alice"), Eq("name", "Bob"))
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2
        assert set(result["name"].to_list()) == {"Alice", "Bob"}

    def test_not_filter(self, sample_df: pl.DataFrame) -> None:
        """Test Not filter execution."""
        f = Not(Eq("active", True))
        result = sample_df.filter(f.to_expr())
        assert len(result) == 1
        assert result["name"][0] == "Charlie"

    def test_and_operator(self, sample_df: pl.DataFrame) -> None:
        """Test & operator for combining filters."""
        f = (Col("age") >= 30) & (Col("active") == True)  # noqa: E712
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2

    def test_or_operator(self, sample_df: pl.DataFrame) -> None:
        """Test | operator for combining filters."""
        f = (Col("name") == "Alice") | (Col("name") == "Bob")
        result = sample_df.filter(f.to_expr())
        assert len(result) == 2

    def test_not_operator(self, sample_df: pl.DataFrame) -> None:
        """Test ~ operator for inverting filters."""
        f = ~(Col("active") == True)  # noqa: E712
        result = sample_df.filter(f.to_expr())
        assert len(result) == 1

    def test_complex_combined_filter(self, sample_df: pl.DataFrame) -> None:
        """Test complex combination of logical operators."""
        f = ((Col("age") >= 25) & (Col("age") <= 35)) | (Col("name") == "David")
        result = sample_df.filter(f.to_expr())
        assert len(result) == 4


class TestFilterSerialization:
    """Tests for filter serialization and deserialization."""

    def test_eq_to_dict(self) -> None:
        """Test Eq serialization."""
        f = Eq("name", "Alice")
        d = f.to_dict()
        assert d == {"type": "eq", "column": "name", "value": "Alice"}

    def test_ne_to_dict(self) -> None:
        """Test Ne serialization."""
        f = Ne("status", "deleted")
        d = f.to_dict()
        assert d == {"type": "ne", "column": "status", "value": "deleted"}

    def test_gt_to_dict(self) -> None:
        """Test Gt serialization."""
        f = Gt("age", 18)
        d = f.to_dict()
        assert d == {"type": "gt", "column": "age", "value": 18}

    def test_is_in_to_dict(self) -> None:
        """Test IsIn serialization."""
        f = IsIn("status", ["a", "b", "c"])
        d = f.to_dict()
        assert d == {"type": "is_in", "column": "status", "values": ["a", "b", "c"]}

    def test_between_to_dict(self) -> None:
        """Test Between serialization."""
        f = Between("age", 18, 65)
        d = f.to_dict()
        assert d == {"type": "between", "column": "age", "lower": 18, "upper": 65}

    def test_is_null_to_dict(self) -> None:
        """Test IsNull serialization."""
        f = IsNull("name")
        d = f.to_dict()
        assert d == {"type": "is_null", "column": "name"}

    def test_str_contains_to_dict(self) -> None:
        """Test StrContains serialization."""
        f = StrContains("text", "pattern", literal=True)
        d = f.to_dict()
        assert d == {
            "type": "str_contains",
            "column": "text",
            "pattern": "pattern",
            "literal": True,
        }

    def test_str_starts_with_to_dict(self) -> None:
        """Test StrStartsWith serialization."""
        f = StrStartsWith("code", "PRD")
        d = f.to_dict()
        assert d == {"type": "str_starts_with", "column": "code", "prefix": "PRD"}

    def test_str_ends_with_to_dict(self) -> None:
        """Test StrEndsWith serialization."""
        f = StrEndsWith("file", ".csv")
        d = f.to_dict()
        assert d == {"type": "str_ends_with", "column": "file", "suffix": ".csv"}

    def test_and_to_dict(self) -> None:
        """Test And serialization."""
        f = And(Eq("a", 1), Gt("b", 2))
        d = f.to_dict()
        assert d["type"] == "and"
        assert d["left"] == {"type": "eq", "column": "a", "value": 1}
        assert d["right"] == {"type": "gt", "column": "b", "value": 2}

    def test_or_to_dict(self) -> None:
        """Test Or serialization."""
        f = Or(Eq("a", 1), Eq("a", 2))
        d = f.to_dict()
        assert d["type"] == "or"

    def test_not_to_dict(self) -> None:
        """Test Not serialization."""
        f = Not(Eq("active", True))
        d = f.to_dict()
        assert d["type"] == "not"
        assert d["operand"] == {"type": "eq", "column": "active", "value": True}


class TestFilterDeserialization:
    """Tests for Filter.from_dict deserialization."""

    def test_eq_from_dict(self) -> None:
        """Test Eq deserialization."""
        d = {"type": "eq", "column": "name", "value": "Alice"}
        f = Filter.from_dict(d)
        assert isinstance(f, Eq)
        assert f.column == "name"
        assert f.value == "Alice"

    def test_ne_from_dict(self) -> None:
        """Test Ne deserialization."""
        d = {"type": "ne", "column": "status", "value": "deleted"}
        f = Filter.from_dict(d)
        assert isinstance(f, Ne)

    def test_gt_from_dict(self) -> None:
        """Test Gt deserialization."""
        d = {"type": "gt", "column": "age", "value": 18}
        f = Filter.from_dict(d)
        assert isinstance(f, Gt)

    def test_ge_from_dict(self) -> None:
        """Test Ge deserialization."""
        d = {"type": "ge", "column": "age", "value": 18}
        f = Filter.from_dict(d)
        assert isinstance(f, Ge)

    def test_lt_from_dict(self) -> None:
        """Test Lt deserialization."""
        d = {"type": "lt", "column": "age", "value": 65}
        f = Filter.from_dict(d)
        assert isinstance(f, Lt)

    def test_le_from_dict(self) -> None:
        """Test Le deserialization."""
        d = {"type": "le", "column": "age", "value": 65}
        f = Filter.from_dict(d)
        assert isinstance(f, Le)

    def test_is_in_from_dict(self) -> None:
        """Test IsIn deserialization."""
        d = {"type": "is_in", "column": "status", "values": ["a", "b"]}
        f = Filter.from_dict(d)
        assert isinstance(f, IsIn)
        assert list(f.values) == ["a", "b"]

    def test_between_from_dict(self) -> None:
        """Test Between deserialization."""
        d = {"type": "between", "column": "age", "lower": 18, "upper": 65}
        f = Filter.from_dict(d)
        assert isinstance(f, Between)
        assert f.lower == 18
        assert f.upper == 65

    def test_is_null_from_dict(self) -> None:
        """Test IsNull deserialization."""
        d = {"type": "is_null", "column": "name"}
        f = Filter.from_dict(d)
        assert isinstance(f, IsNull)

    def test_is_not_null_from_dict(self) -> None:
        """Test IsNotNull deserialization."""
        d = {"type": "is_not_null", "column": "name"}
        f = Filter.from_dict(d)
        assert isinstance(f, IsNotNull)

    def test_str_contains_from_dict(self) -> None:
        """Test StrContains deserialization."""
        d = {
            "type": "str_contains",
            "column": "text",
            "pattern": "foo",
            "literal": True,
        }
        f = Filter.from_dict(d)
        assert isinstance(f, StrContains)
        assert f.pattern == "foo"
        assert f.literal is True

    def test_and_from_dict(self) -> None:
        """Test And deserialization."""
        d = {
            "type": "and",
            "left": {"type": "eq", "column": "a", "value": 1},
            "right": {"type": "gt", "column": "b", "value": 2},
        }
        f = Filter.from_dict(d)
        assert isinstance(f, And)
        assert isinstance(f.left, Eq)
        assert isinstance(f.right, Gt)

    def test_or_from_dict(self) -> None:
        """Test Or deserialization."""
        d = {
            "type": "or",
            "left": {"type": "eq", "column": "a", "value": 1},
            "right": {"type": "eq", "column": "a", "value": 2},
        }
        f = Filter.from_dict(d)
        assert isinstance(f, Or)

    def test_not_from_dict(self) -> None:
        """Test Not deserialization."""
        d = {
            "type": "not",
            "operand": {"type": "eq", "column": "active", "value": True},
        }
        f = Filter.from_dict(d)
        assert isinstance(f, Not)
        assert isinstance(f.operand, Eq)

    def test_unknown_type_raises(self) -> None:
        """Test that unknown type raises ValueError."""
        d = {"type": "unknown", "column": "x"}
        with pytest.raises(ValueError, match="Unknown filter type"):
            Filter.from_dict(d)

    def test_missing_type_raises(self) -> None:
        """Test that missing type raises ValueError."""
        d = {"column": "x", "value": 1}
        with pytest.raises(ValueError, match="Missing 'type'"):
            Filter.from_dict(d)


class TestFilterRoundtrip:
    """Tests for filter serialization roundtrip."""

    def test_eq_roundtrip(self) -> None:
        """Test Eq roundtrip."""
        original = Eq("name", "Alice")
        restored = Filter.from_dict(original.to_dict())
        assert isinstance(restored, Eq)
        assert restored.column == original.column
        assert restored.value == original.value

    def test_complex_filter_roundtrip(self) -> None:
        """Test complex filter roundtrip."""
        original = And(
            Or(Eq("a", 1), Eq("a", 2)),
            Not(IsNull("b")),
        )
        d = original.to_dict()
        restored = Filter.from_dict(d)
        assert isinstance(restored, And)
        assert isinstance(restored.left, Or)
        assert isinstance(restored.right, Not)

    def test_filter_roundtrip_produces_same_results(self) -> None:
        """Test that roundtrip filter produces same results."""
        df = pl.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
            }
        )

        original = (Col("age") >= 25) & (Col("age") <= 30)
        restored = Filter.from_dict(original.to_dict())

        result1 = df.filter(original.to_expr())
        result2 = df.filter(restored.to_expr())

        assert result1.equals(result2)


class TestFilterSerializationMissing:
    """Tests for missing filter serialization coverage."""

    def test_lt_to_dict(self) -> None:
        """Test Lt.to_dict() serialization."""
        f = Lt("age", 65)
        d = f.to_dict()
        assert d == {"type": "lt", "column": "age", "value": 65}

    def test_is_not_null_to_dict(self) -> None:
        """Test IsNotNull.to_dict() serialization."""
        f = IsNotNull("name")
        d = f.to_dict()
        assert d == {"type": "is_not_null", "column": "name"}

    def test_str_starts_with_roundtrip(self) -> None:
        """Test StrStartsWith round-trip via Filter.from_dict()."""
        original = StrStartsWith("code", "PRD-")
        d = original.to_dict()
        restored = Filter.from_dict(d)
        assert isinstance(restored, StrStartsWith)
        assert restored.column == "code"
        assert restored.prefix == "PRD-"

    def test_str_ends_with_roundtrip(self) -> None:
        """Test StrEndsWith round-trip via Filter.from_dict()."""
        original = StrEndsWith("file", ".csv")
        d = original.to_dict()
        restored = Filter.from_dict(d)
        assert isinstance(restored, StrEndsWith)
        assert restored.column == "file"
        assert restored.suffix == ".csv"


class TestSqlHelpers:
    """Tests for SQL helper functions."""

    def test_quote_identifier_simple(self) -> None:
        assert _sql_quote_identifier("age") == '"age"'

    def test_quote_identifier_with_double_quote(self) -> None:
        assert _sql_quote_identifier('col"name') == '"col""name"'

    def test_format_value_none(self) -> None:
        assert _sql_format_value(None) == "NULL"

    def test_format_value_bool_true(self) -> None:
        assert _sql_format_value(True) == "TRUE"

    def test_format_value_bool_false(self) -> None:
        assert _sql_format_value(False) == "FALSE"

    def test_format_value_int(self) -> None:
        assert _sql_format_value(42) == "42"

    def test_format_value_float(self) -> None:
        assert _sql_format_value(2.5) == "2.5"

    def test_format_value_string(self) -> None:
        assert _sql_format_value("hello") == "'hello'"

    def test_format_value_string_with_quote(self) -> None:
        assert _sql_format_value("O'Brien") == "'O''Brien'"

    def test_escape_like_metacharacters(self) -> None:
        assert _sql_escape_like("100%_done") == "100\\%\\_done"

    def test_escape_like_backslash(self) -> None:
        assert _sql_escape_like("a\\b") == "a\\\\b"

    def test_escape_like_single_quote(self) -> None:
        assert _sql_escape_like("it's") == "it''s"


class TestFilterToSql:
    """Tests for to_sql() on all filter types."""

    def test_eq_string(self) -> None:
        assert Eq("name", "Alice").to_sql() == "\"name\" = 'Alice'"

    def test_eq_int(self) -> None:
        assert Eq("age", 30).to_sql() == '"age" = 30'

    def test_eq_none(self) -> None:
        assert Eq("name", None).to_sql() == '"name" IS NULL'

    def test_eq_bool(self) -> None:
        assert Eq("active", True).to_sql() == '"active" = TRUE'

    def test_ne_string(self) -> None:
        assert Ne("status", "deleted").to_sql() == "\"status\" != 'deleted'"

    def test_ne_none(self) -> None:
        assert Ne("name", None).to_sql() == '"name" IS NOT NULL'

    def test_gt(self) -> None:
        assert Gt("age", 18).to_sql() == '"age" > 18'

    def test_ge(self) -> None:
        assert Ge("age", 18).to_sql() == '"age" >= 18'

    def test_lt(self) -> None:
        assert Lt("age", 65).to_sql() == '"age" < 65'

    def test_le(self) -> None:
        assert Le("score", 99.5).to_sql() == '"score" <= 99.5'

    def test_is_in(self) -> None:
        assert IsIn("status", ["a", "b"]).to_sql() == "\"status\" IN ('a', 'b')"

    def test_is_in_empty(self) -> None:
        assert IsIn("status", []).to_sql() == "FALSE"

    def test_is_in_ints(self) -> None:
        assert IsIn("id", [1, 2, 3]).to_sql() == '"id" IN (1, 2, 3)'

    def test_between_int(self) -> None:
        assert Between("age", 18, 65).to_sql() == '"age" BETWEEN 18 AND 65'

    def test_between_string(self) -> None:
        assert Between("name", "A", "M").to_sql() == "\"name\" BETWEEN 'A' AND 'M'"

    def test_is_null(self) -> None:
        assert IsNull("name").to_sql() == '"name" IS NULL'

    def test_is_not_null(self) -> None:
        assert IsNotNull("name").to_sql() == '"name" IS NOT NULL'

    def test_str_contains_literal(self) -> None:
        result = StrContains("email", "@example").to_sql()
        assert result == "\"email\" LIKE '%@example%' ESCAPE '\\'"

    def test_str_contains_regex(self) -> None:
        result = StrContains("text", r"\d+", literal=False).to_sql()
        assert result == "regexp_matches(\"text\", '\\d+')"

    def test_str_contains_like_escaping(self) -> None:
        result = StrContains("col", "100%_done").to_sql()
        assert result == "\"col\" LIKE '%100\\%\\_done%' ESCAPE '\\'"

    def test_str_starts_with(self) -> None:
        result = StrStartsWith("code", "PRD-").to_sql()
        assert result == "\"code\" LIKE 'PRD-%' ESCAPE '\\'"

    def test_str_ends_with(self) -> None:
        result = StrEndsWith("file", ".csv").to_sql()
        assert result == "\"file\" LIKE '%.csv' ESCAPE '\\'"

    def test_and(self) -> None:
        f = And(Ge("age", 18), Eq("active", True))
        assert f.to_sql() == '("age" >= 18) AND ("active" = TRUE)'

    def test_or(self) -> None:
        f = Or(Eq("a", 1), Eq("a", 2))
        assert f.to_sql() == '("a" = 1) OR ("a" = 2)'

    def test_not(self) -> None:
        f = Not(Eq("active", True))
        assert f.to_sql() == 'NOT ("active" = TRUE)'

    def test_complex_nested(self) -> None:
        f = (Col("age") >= 18) & (Col("status") == "active")
        assert f.to_sql() == '("age" >= 18) AND ("status" = \'active\')'

    def test_or_and_not_nested(self) -> None:
        f = Or(And(Gt("a", 0), Lt("a", 10)), Not(IsNull("b")))
        expected = '(("a" > 0) AND ("a" < 10)) OR (NOT ("b" IS NULL))'
        assert f.to_sql() == expected

    def test_string_with_single_quote(self) -> None:
        assert Eq("name", "O'Brien").to_sql() == "\"name\" = 'O''Brien'"

    def test_column_with_double_quote(self) -> None:
        assert Eq('col"x', 1).to_sql() == '"col""x" = 1'

    def test_str_contains_regex_with_quote(self) -> None:
        result = StrContains("col", "it's.*", literal=False).to_sql()
        assert result == "regexp_matches(\"col\", 'it''s.*')"

    def test_str_starts_with_quote_in_prefix(self) -> None:
        result = StrStartsWith("col", "it's").to_sql()
        assert result == "\"col\" LIKE 'it''s%' ESCAPE '\\'"
