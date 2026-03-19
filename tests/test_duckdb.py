"""Tests for DuckDB backend.

Mirrors the Polars test suite structure: same fixture data, same expected outputs,
adapted for DuckDB relations as input/output.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

pytestmark = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")

from transformplan import Col, TransformPlan  # noqa: E402
from transformplan.backends.duckdb import DuckDBBackend  # noqa: E402

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


@pytest.fixture
def backend(con: duckdb.DuckDBPyConnection) -> DuckDBBackend:
    return DuckDBBackend(con)


@pytest.fixture
def basic_rel(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    return con.sql(
        "SELECT * FROM (VALUES "
        "(1, 'Alice', 25, 50000.0, TRUE), "
        "(2, 'Bob', 30, 60000.0, TRUE), "
        "(3, 'Charlie', 35, 70000.0, FALSE), "
        "(4, 'David', 40, 80000.0, TRUE), "
        "(5, 'Eve', 45, 90000.0, FALSE)"
        ") AS t(id, name, age, salary, active)"
    )


@pytest.fixture
def numeric_rel(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    return con.sql(
        "SELECT * FROM (VALUES "
        "(1, 10, 100), (2, 20, 200), (3, 30, 300), (4, 40, 400), (5, 50, 500)"
        ") AS t(a, b, c)"
    )


@pytest.fixture
def string_rel(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    return con.sql(
        "SELECT * FROM (VALUES "
        "('  Hello World  ', 'PRD-001', 'John', 'Doe'), "
        "('foo bar baz', 'PRD-002', 'Jane', 'Smith'), "
        "('ABC123xyz', 'TST-001', 'Bob', 'Brown'), "
        "('test@email.com', 'DEV-001', 'Alice', 'Johnson'), "
        "('', 'PRD-003', 'Charlie', 'Williams')"
        ") AS t(text, code, first_name, last_name)"
    )


@pytest.fixture
def datetime_rel(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    return con.sql(
        "SELECT * FROM (VALUES "
        "(DATE '2024-01-15', TIMESTAMP '2024-01-15 10:30:00', '2024-01-15', DATE '1990-05-10'), "
        "(DATE '2024-03-20', TIMESTAMP '2024-03-20 14:45:00', '2024-03-20', DATE '1985-08-22'), "
        "(DATE '2024-06-10', TIMESTAMP '2024-06-10 08:15:00', '2024-06-10', DATE '2000-01-01'), "
        "(DATE '2024-09-05', TIMESTAMP '2024-09-05 16:00:00', '2024-09-05', DATE '1975-12-15'), "
        "(DATE '2024-12-25', TIMESTAMP '2024-12-25 12:00:00', '2024-12-25', DATE '1995-03-30')"
        ") AS t(date_col, datetime_col, date_str, birth_date)"
    )


@pytest.fixture
def map_rel(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    return con.sql(
        "SELECT * FROM (VALUES "
        "('A', 85, 'X', 1, 'One'), "
        "('B', 72, 'Y', 2, 'Two'), "
        "('A', 91, 'X', 1, 'One'), "
        "('C', 68, 'Z', 3, 'Three'), "
        "('B', 79, 'Y', 2, 'Two')"
        ") AS t(status, score, category, lookup_key, lookup_value)"
    )


@pytest.fixture
def null_rel(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    return con.sql(
        "SELECT * FROM (VALUES "
        "(1, 'Alice', 25, 50000.0), "
        "(2, NULL, 30, NULL), "
        "(3, 'Charlie', NULL, 70000.0), "
        "(4, 'David', 40, NULL), "
        "(5, NULL, 45, 90000.0)"
        ") AS t(id, name, age, salary)"
    )


def _plan(backend: DuckDBBackend) -> TransformPlan:
    """Create a TransformPlan with DuckDB backend."""
    return TransformPlan(backend=backend)


def _col_values(
    rel: duckdb.DuckDBPyRelation, col: str
) -> list[Any]:
    """Fetch values of a single column."""
    idx = list(rel.columns).index(col)
    return [row[idx] for row in rel.fetchall()]


# =============================================================================
# Meta methods
# =============================================================================


class TestMetaMethods:
    def test_get_shape(self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation) -> None:
        assert backend.get_shape(basic_rel) == (5, 5)

    def test_get_columns(self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation) -> None:
        assert backend.get_columns(basic_rel) == ["id", "name", "age", "salary", "active"]

    def test_get_schema(self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation) -> None:
        schema = backend.get_schema(basic_rel)
        assert "id" in schema
        assert isinstance(schema["id"], str)

    def test_compute_hash_deterministic(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        h1 = backend.compute_hash(basic_rel)
        h2 = backend.compute_hash(basic_rel)
        assert h1 == h2
        assert len(h1) == 16


# =============================================================================
# Column operations
# =============================================================================


class TestColDrop:
    def test_col_drop(self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = _plan(backend).col_drop("age").process(basic_rel)
        assert "age" not in result.columns
        assert len(result.columns) == 4

    def test_col_drop_preserves_other_columns(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).col_drop("age").process(basic_rel)
        assert "id" in result.columns
        assert "name" in result.columns


class TestColRename:
    def test_col_rename(self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = _plan(backend).col_rename("name", "full_name").process(basic_rel)
        assert "full_name" in result.columns
        assert "name" not in result.columns

    def test_col_rename_preserves_data(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).col_rename("name", "full_name").process(basic_rel)
        vals = _col_values(result, "full_name")
        assert vals == ["Alice", "Bob", "Charlie", "David", "Eve"]


class TestColCast:
    def test_col_cast_int_to_float(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).col_cast("age", float).process(basic_rel)
        vals = _col_values(result, "age")
        assert all(isinstance(v, float) for v in vals)

    def test_col_cast_int_to_string(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).col_cast("id", str).process(basic_rel)
        vals = _col_values(result, "id")
        assert vals[0] == "1"


class TestColReorder:
    def test_col_reorder(self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .col_reorder(["salary", "name", "id", "age", "active"])
            .process(basic_rel)
        )
        assert list(result.columns) == ["salary", "name", "id", "age", "active"]


class TestColSelect:
    def test_col_select(self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = _plan(backend).col_select(["id", "name"]).process(basic_rel)
        assert list(result.columns) == ["id", "name"]
        assert backend.get_shape(result) == (5, 2)


class TestColDuplicate:
    def test_col_duplicate(self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = _plan(backend).col_duplicate("name", "name_copy").process(basic_rel)
        assert "name_copy" in result.columns
        assert _col_values(result, "name") == _col_values(result, "name_copy")


class TestColFillNull:
    def test_col_fill_null_with_value(
        self, backend: DuckDBBackend, null_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).col_fill_null("name", value="Unknown").process(null_rel)
        vals = _col_values(result, "name")
        assert None not in vals
        assert vals[1] == "Unknown"

    def test_col_fill_null_zero_strategy(
        self, backend: DuckDBBackend, null_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).col_fill_null("age", strategy="zero").process(null_rel)
        )
        vals = _col_values(result, "age")
        assert None not in vals


class TestColDropNull:
    def test_col_drop_null(
        self, backend: DuckDBBackend, null_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).col_drop_null(columns=["name"]).process(null_rel)
        assert backend.get_shape(result)[0] == 3  # rows 2 and 5 had null names


class TestColDropZero:
    def test_col_drop_zero(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql(
            "SELECT * FROM (VALUES (1, 10), (2, 0), (3, 30)) AS t(id, val)"
        )
        result, _ = _plan(backend).col_drop_zero("val").process(rel)
        assert backend.get_shape(result)[0] == 2


class TestColAdd:
    def test_col_add_with_value(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).col_add("status", value="active").process(basic_rel)
        assert "status" in result.columns
        vals = _col_values(result, "status")
        assert all(v == "active" for v in vals)

    def test_col_add_from_column(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).col_add("name_copy", expr="name").process(basic_rel)
        assert _col_values(result, "name_copy") == _col_values(result, "name")


class TestColAddUuid:
    def test_col_add_uuid(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).col_add_uuid("uuid", length=8).process(basic_rel)
        assert "uuid" in result.columns
        vals = _col_values(result, "uuid")
        assert all(len(v) == 8 for v in vals)
        assert len(set(vals)) == 5  # all unique


class TestColHash:
    def test_col_hash(self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .col_hash(columns=["id", "name"], new_column="hash", salt="test")
            .process(basic_rel)
        )
        assert "hash" in result.columns
        vals = _col_values(result, "hash")
        assert all(len(v) == 16 for v in vals)


class TestColCoalesce:
    def test_col_coalesce(
        self, backend: DuckDBBackend, null_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .col_coalesce(columns=["name", "id"], new_column="first_non_null")
            .process(null_rel)
        )
        assert "first_non_null" in result.columns


# =============================================================================
# Math operations
# =============================================================================


class TestMathScalar:
    def test_math_add(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = _plan(backend).math_add("a", 10).process(numeric_rel)
        assert _col_values(result, "a") == [11, 12, 13, 14, 15]

    def test_math_subtract(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = _plan(backend).math_subtract("a", 1).process(numeric_rel)
        assert _col_values(result, "a") == [0, 1, 2, 3, 4]

    def test_math_multiply(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = _plan(backend).math_multiply("a", 2).process(numeric_rel)
        assert _col_values(result, "a") == [2, 4, 6, 8, 10]

    def test_math_divide(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = _plan(backend).math_divide("b", 10).process(numeric_rel)
        assert _col_values(result, "b") == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_math_abs(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql("SELECT * FROM (VALUES (-1,), (2,), (-3,)) AS t(a)")
        result, _ = _plan(backend).math_abs("a").process(rel)
        assert _col_values(result, "a") == [1, 2, 3]

    def test_math_round(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql("SELECT * FROM (VALUES (1.234,), (5.678,)) AS t(a)")
        result, _ = _plan(backend).math_round("a", 1).process(rel)
        vals = _col_values(result, "a")
        assert [float(v) for v in vals] == [1.2, 5.7]

    def test_math_clamp(
        self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).math_clamp("a", lower=2, upper=4).process(numeric_rel)
        assert _col_values(result, "a") == [2, 2, 3, 4, 4]

    def test_math_set_min(
        self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).math_set_min("a", 3).process(numeric_rel)
        assert _col_values(result, "a") == [3, 3, 3, 4, 5]

    def test_math_set_max(
        self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).math_set_max("a", 3).process(numeric_rel)
        assert _col_values(result, "a") == [1, 2, 3, 3, 3]


class TestMathColumnWise:
    def test_math_add_columns(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .math_add_columns("a", "b", "sum")
            .process(numeric_rel)
        )
        assert _col_values(result, "sum") == [11, 22, 33, 44, 55]

    def test_math_subtract_columns(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .math_subtract_columns("b", "a", "diff")
            .process(numeric_rel)
        )
        assert _col_values(result, "diff") == [9, 18, 27, 36, 45]

    def test_math_multiply_columns(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .math_multiply_columns("a", "b", "prod")
            .process(numeric_rel)
        )
        assert _col_values(result, "prod") == [10, 40, 90, 160, 250]

    def test_math_divide_columns(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .math_divide_columns("b", "a", "ratio")
            .process(numeric_rel)
        )
        assert _col_values(result, "ratio") == [10.0, 10.0, 10.0, 10.0, 10.0]

    def test_math_percent_of(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .math_percent_of("a", "c", "pct", multiply_by=100.0)
            .process(numeric_rel)
        )
        assert _col_values(result, "pct") == [1.0, 1.0, 1.0, 1.0, 1.0]


class TestMathWindow:
    def test_math_cumsum(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = _plan(backend).math_cumsum("a", "cumulative").process(numeric_rel)
        assert _col_values(result, "cumulative") == [1, 3, 6, 10, 15]

    def test_math_rank(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .math_rank("a", "rank", method="ordinal", descending=False)
            .process(numeric_rel)
        )
        assert _col_values(result, "rank") == [1, 2, 3, 4, 5]


class TestMathScaling:
    def test_math_standardize(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend).math_standardize("a", new_column="z_score").process(numeric_rel)
        )
        vals = _col_values(result, "z_score")
        assert len(vals) == 5
        # Mean should be ~0
        assert abs(sum(vals) / len(vals)) < 0.01

    def test_math_minmax(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .math_minmax("a", feature_range=(0.0, 1.0), new_column="scaled")
            .process(numeric_rel)
        )
        vals = _col_values(result, "scaled")
        assert vals[0] == pytest.approx(0.0)
        assert vals[-1] == pytest.approx(1.0)

    def test_math_robust_scale(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .math_robust_scale("a", new_column="robust")
            .process(numeric_rel)
        )
        assert "robust" in result.columns


class TestMathTransform:
    def test_math_log(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend).math_log("a", offset=0, new_column="log_a").process(numeric_rel)
        )
        vals = _col_values(result, "log_a")
        assert vals[0] == pytest.approx(0.0, abs=0.01)  # ln(1) = 0

    def test_math_sqrt(self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection) -> None:
        rel = con.sql("SELECT * FROM (VALUES (4,), (9,), (16,)) AS t(a)")
        result, _ = _plan(backend).math_sqrt("a", new_column="sqrt_a").process(rel)
        assert _col_values(result, "sqrt_a") == [2.0, 3.0, 4.0]

    def test_math_power(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend).math_power("a", 2, new_column="sq").process(numeric_rel)
        )
        assert _col_values(result, "sq") == [1.0, 4.0, 9.0, 16.0, 25.0]

    def test_math_winsorize(self, backend: DuckDBBackend, numeric_rel: duckdb.DuckDBPyRelation) -> None:
        result, _ = (
            _plan(backend)
            .math_winsorize("a", lower_value=2, upper_value=4, new_column="w")
            .process(numeric_rel)
        )
        vals = _col_values(result, "w")
        assert min(vals) >= 2
        assert max(vals) <= 4


# =============================================================================
# Row operations
# =============================================================================


class TestRowsFilter:
    def test_rows_filter_ge(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).rows_filter(Col("age") >= 35).process(basic_rel)
        )
        assert backend.get_shape(result)[0] == 3

    def test_rows_filter_eq(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).rows_filter(Col("name") == "Alice").process(basic_rel)
        )
        assert backend.get_shape(result)[0] == 1

    def test_rows_filter_combined(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .rows_filter((Col("age") >= 30) & (Col("age") <= 40))
            .process(basic_rel)
        )
        assert backend.get_shape(result)[0] == 3


class TestRowsDrop:
    def test_rows_drop(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).rows_drop(Col("age") < 30).process(basic_rel)
        )
        assert backend.get_shape(result)[0] == 4  # only age=25 dropped


class TestRowsDropNulls:
    def test_rows_drop_nulls(
        self, backend: DuckDBBackend, null_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).rows_drop_nulls(columns=["name", "age"]).process(null_rel)
        )
        # Rows with null name (2,5) or null age (3) are dropped
        assert backend.get_shape(result)[0] == 2


class TestRowsFlag:
    def test_rows_flag(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .rows_flag(Col("age") >= 35, "senior", true_value="yes", false_value="no")
            .process(basic_rel)
        )
        assert "senior" in result.columns
        vals = _col_values(result, "senior")
        assert vals == ["no", "no", "yes", "yes", "yes"]


class TestRowsUnique:
    def test_rows_unique_first(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql(
            "SELECT * FROM (VALUES (1, 'A'), (1, 'A'), (2, 'B')) AS t(id, name)"
        )
        result, _ = (
            _plan(backend).rows_unique(columns=["id", "name"], keep="first").process(rel)
        )
        assert backend.get_shape(result)[0] == 2

    def test_rows_unique_none(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql(
            "SELECT * FROM (VALUES (1, 'A'), (1, 'A'), (2, 'B')) AS t(id, name)"
        )
        result, _ = (
            _plan(backend).rows_unique(columns=["id", "name"], keep="none").process(rel)
        )
        assert backend.get_shape(result)[0] == 1  # only (2, 'B') kept


class TestRowsDeduplicate:
    def test_rows_deduplicate(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql(
            "SELECT * FROM (VALUES "
            "(1, 'A', 10), (1, 'A', 20), (2, 'B', 30)"
            ") AS t(id, name, val)"
        )
        result, _ = (
            _plan(backend)
            .rows_deduplicate(columns=["id"], sort_by="val", keep="first", descending=False)
            .process(rel)
        )
        assert backend.get_shape(result)[0] == 2
        vals = _col_values(result, "val")
        assert 10 in vals  # first by val ascending for id=1


class TestRowsSort:
    def test_rows_sort_ascending(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).rows_sort(by=["age"], descending=False).process(basic_rel)
        )
        vals = _col_values(result, "age")
        assert vals == sorted(vals)

    def test_rows_sort_descending(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).rows_sort(by=["age"], descending=True).process(basic_rel)
        )
        vals = _col_values(result, "age")
        assert vals == sorted(vals, reverse=True)


class TestRowsHeadTail:
    def test_rows_head(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).rows_head(3).process(basic_rel)
        assert backend.get_shape(result)[0] == 3

    def test_rows_tail(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).rows_tail(2).process(basic_rel)
        assert backend.get_shape(result)[0] == 2


class TestRowsSample:
    def test_rows_sample_n(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).rows_sample(n=3).process(basic_rel)
        assert backend.get_shape(result)[0] == 3


class TestRowsExplode:
    def test_rows_explode(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql(
            "SELECT * FROM (VALUES "
            "(1, ['a', 'b', 'c']), (2, ['d', 'e'])"
            ") AS t(id, tags)"
        )
        result, _ = _plan(backend).rows_explode("tags").process(rel)
        assert backend.get_shape(result)[0] == 5


class TestRowsMelt:
    def test_rows_melt(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql(
            "SELECT * FROM (VALUES (1, 'A', 10, 20), (2, 'B', 30, 40)) "
            "AS t(id, name, q1, q2)"
        )
        result, _ = (
            _plan(backend)
            .rows_melt(
                id_columns=["id", "name"],
                value_columns=["q1", "q2"],
                variable_name="quarter",
                value_name="value",
            )
            .process(rel)
        )
        assert backend.get_shape(result)[0] == 4
        assert "quarter" in result.columns
        assert "value" in result.columns


class TestRowsPivot:
    def test_rows_pivot(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql(
            "SELECT * FROM (VALUES "
            "(1, 'Q1', 100), (1, 'Q2', 110), "
            "(2, 'Q1', 150), (2, 'Q2', 160)"
            ") AS t(id, quarter, value)"
        )
        result, _ = (
            _plan(backend)
            .rows_pivot(
                index=["id"],
                columns="quarter",
                values="value",
                aggregate_function="sum",
            )
            .process(rel)
        )
        assert backend.get_shape(result)[0] == 2


# =============================================================================
# String operations
# =============================================================================


class TestStrReplace:
    def test_str_replace_literal(
        self, backend: DuckDBBackend, string_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .str_replace("code", "PRD", "PROD", literal=True)
            .process(string_rel)
        )
        vals = _col_values(result, "code")
        assert vals[0] == "PROD-001"


class TestStrSlice:
    def test_str_slice(
        self, backend: DuckDBBackend, string_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).str_slice("code", offset=0, length=3).process(string_rel)
        )
        vals = _col_values(result, "code")
        assert vals[0] == "PRD"


class TestStrTruncate:
    def test_str_truncate(
        self, backend: DuckDBBackend, string_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).str_truncate("text", max_length=10, suffix="...").process(string_rel)
        )
        vals = _col_values(result, "text")
        # "  Hello World  " (15 chars) should be truncated
        assert len(vals[0]) <= 10


class TestStrCase:
    def test_str_lower(
        self, backend: DuckDBBackend, string_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).str_lower("code").process(string_rel)
        vals = _col_values(result, "code")
        assert vals[0] == "prd-001"

    def test_str_upper(
        self, backend: DuckDBBackend, string_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).str_upper("first_name").process(string_rel)
        vals = _col_values(result, "first_name")
        assert vals[0] == "JOHN"


class TestStrStrip:
    def test_str_strip(
        self, backend: DuckDBBackend, string_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).str_strip("text").process(string_rel)
        vals = _col_values(result, "text")
        assert vals[0] == "Hello World"


class TestStrPad:
    def test_str_pad_left(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql("SELECT * FROM (VALUES ('1',), ('22',), ('333',)) AS t(a)")
        result, _ = (
            _plan(backend).str_pad("a", length=5, fill_char="0", side="left").process(rel)
        )
        vals = _col_values(result, "a")
        assert vals[0] == "00001"

    def test_str_pad_right(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql("SELECT * FROM (VALUES ('1',), ('22',)) AS t(a)")
        result, _ = (
            _plan(backend).str_pad("a", length=5, fill_char=".", side="right").process(rel)
        )
        vals = _col_values(result, "a")
        assert vals[0] == "1...."


class TestStrConcat:
    def test_str_concat(
        self, backend: DuckDBBackend, string_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .str_concat(["first_name", "last_name"], "full_name", separator=" ")
            .process(string_rel)
        )
        vals = _col_values(result, "full_name")
        assert vals[0] == "John Doe"


class TestStrExtract:
    def test_str_extract(
        self, backend: DuckDBBackend, string_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .str_extract("code", r"(\w+)-(\d+)", group_index=1, new_column="prefix")
            .process(string_rel)
        )
        vals = _col_values(result, "prefix")
        assert vals[0] == "PRD"


class TestStrSplit:
    def test_str_split_with_columns(
        self, backend: DuckDBBackend, con: duckdb.DuckDBPyConnection
    ) -> None:
        rel = con.sql(
            "SELECT * FROM (VALUES ('a-b-c',), ('d-e-f',)) AS t(text)"
        )
        result, _ = (
            _plan(backend)
            .str_split("text", separator="-", new_columns=["p1", "p2", "p3"])
            .process(rel)
        )
        assert "p1" in result.columns
        assert "p2" in result.columns
        assert "p3" in result.columns
        vals = _col_values(result, "p1")
        assert vals[0] == "a"


# =============================================================================
# Datetime operations
# =============================================================================


class TestDtExtract:
    def test_dt_year(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).dt_year("date_col", "year").process(datetime_rel)
        )
        vals = _col_values(result, "year")
        assert all(v == 2024 for v in vals)

    def test_dt_month(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).dt_month("date_col", "month").process(datetime_rel)
        )
        vals = _col_values(result, "month")
        assert vals[0] == 1
        assert vals[1] == 3

    def test_dt_day(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).dt_day("date_col", "day").process(datetime_rel)
        )
        vals = _col_values(result, "day")
        assert vals[0] == 15

    def test_dt_quarter(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).dt_quarter("date_col", "q").process(datetime_rel)
        )
        vals = _col_values(result, "q")
        assert vals[0] == 1
        assert vals[-1] == 4

    def test_dt_week(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).dt_week("date_col", "week").process(datetime_rel)
        )
        vals = _col_values(result, "week")
        assert all(isinstance(v, int) for v in vals)


class TestDtFormat:
    def test_dt_year_month(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .dt_year_month("date_col", "ym", fmt="%Y-%m")
            .process(datetime_rel)
        )
        vals = _col_values(result, "ym")
        assert vals[0] == "2024-01"

    def test_dt_quarter_year(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).dt_quarter_year("date_col", "qy").process(datetime_rel)
        )
        vals = _col_values(result, "qy")
        assert vals[0] == "Q1-2024"
        assert vals[-1] == "Q4-2024"

    def test_dt_calendar_week(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).dt_calendar_week("date_col", "cw").process(datetime_rel)
        )
        vals = _col_values(result, "cw")
        assert vals[0].startswith("2024-W")

    def test_dt_format(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .dt_format("date_col", "%Y/%m/%d", "formatted")
            .process(datetime_rel)
        )
        vals = _col_values(result, "formatted")
        assert vals[0] == "2024/01/15"


class TestDtParse:
    def test_dt_parse(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .dt_parse("date_str", "%Y-%m-%d", "parsed")
            .process(datetime_rel)
        )
        vals = _col_values(result, "parsed")
        assert vals[0] == date(2024, 1, 15)


class TestDtArithmetic:
    def test_dt_diff_days(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .dt_diff_days("date_col", "birth_date", "age_days")
            .process(datetime_rel)
        )
        vals = _col_values(result, "age_days")
        assert all(v > 0 for v in vals)

    def test_dt_age_years(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .dt_age_years("birth_date", reference_column="date_col", new_column="age")
            .process(datetime_rel)
        )
        vals = _col_values(result, "age")
        assert all(v > 0 for v in vals)


class TestDtTruncate:
    def test_dt_truncate_month(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .dt_truncate("date_col", "1mo", "month_start")
            .process(datetime_rel)
        )
        vals = _col_values(result, "month_start")
        # All dates should be 1st of month
        assert all(v.day == 1 for v in vals)


class TestDtIsBetween:
    def test_dt_is_between(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .dt_is_between(
                "date_col",
                start="2024-01-01",
                end="2024-06-30",
                new_column="in_h1",
                closed="both",
            )
            .process(datetime_rel)
        )
        vals = _col_values(result, "in_h1")
        assert vals[0] is True   # Jan 15
        assert vals[2] is True   # Jun 10
        assert vals[3] is False  # Sep 5


# =============================================================================
# Map operations
# =============================================================================


class TestMapValues:
    def test_map_values(
        self, backend: DuckDBBackend, map_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .map_values(
                "status",
                mapping={"A": "Active", "B": "Beta", "C": "Closed"},
                default="Unknown",
            )
            .process(map_rel)
        )
        vals = _col_values(result, "status")
        assert vals[0] == "Active"
        assert vals[1] == "Beta"
        assert vals[3] == "Closed"


class TestMapCase:
    def test_map_case(
        self, backend: DuckDBBackend, map_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .map_case(
                "status",
                cases=[("A", "High"), ("B", "Medium"), ("C", "Low")],
                default="Unknown",
                new_column="priority",
            )
            .process(map_rel)
        )
        vals = _col_values(result, "priority")
        assert vals[0] == "High"
        assert vals[3] == "Low"


class TestMapFromColumn:
    def test_map_from_column(
        self, backend: DuckDBBackend, map_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .map_from_column(
                "lookup_key",
                lookup_column="lookup_key",
                value_column="lookup_value",
                new_column="resolved",
                default="N/A",
            )
            .process(map_rel)
        )
        vals = _col_values(result, "resolved")
        assert vals[0] == "One"
        assert vals[3] == "Three"


class TestMapDiscretize:
    def test_map_discretize(
        self, backend: DuckDBBackend, map_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .map_discretize(
                "score",
                bins=[70, 80, 90],
                labels=["D", "C", "B", "A"],
                new_column="grade",
                right=True,
            )
            .process(map_rel)
        )
        vals = _col_values(result, "grade")
        assert "A" in vals  # score=91
        assert "D" in vals  # score=68


class TestMapOnehot:
    def test_map_onehot(
        self, backend: DuckDBBackend, map_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .map_onehot("category", prefix="cat", drop_original=True)
            .process(map_rel)
        )
        assert "category" not in result.columns
        assert "cat_X" in result.columns
        assert "cat_Y" in result.columns
        assert "cat_Z" in result.columns


class TestMapOrdinal:
    def test_map_ordinal(
        self, backend: DuckDBBackend, map_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .map_ordinal(
                "status",
                categories=["A", "B", "C"],
                new_column="status_ord",
            )
            .process(map_rel)
        )
        vals = _col_values(result, "status_ord")
        assert vals[0] == 0  # A
        assert vals[1] == 1  # B
        assert vals[3] == 2  # C


class TestMapLabel:
    def test_map_label(
        self, backend: DuckDBBackend, map_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .map_label(
                "status",
                categories=["A", "B", "C"],
                new_column="label",
            )
            .process(map_rel)
        )
        vals = _col_values(result, "label")
        assert vals[0] == 0


class TestMapBoolToInt:
    def test_map_bool_to_int(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = _plan(backend).map_bool_to_int("active").process(basic_rel)
        vals = _col_values(result, "active")
        assert vals[0] == 1  # True -> 1
        assert vals[2] == 0  # False -> 0


class TestMapNullToValue:
    def test_map_null_to_value(
        self, backend: DuckDBBackend, null_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).map_null_to_value("name", "Unknown").process(null_rel)
        )
        vals = _col_values(result, "name")
        assert None not in vals
        assert vals[1] == "Unknown"


class TestMapValueToNull:
    def test_map_value_to_null(
        self, backend: DuckDBBackend, map_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend).map_value_to_null("status", "C").process(map_rel)
        )
        vals = _col_values(result, "status")
        assert vals[3] is None  # C -> null


# =============================================================================
# Integration / Pipeline tests
# =============================================================================


class TestPipeline:
    def test_multi_step_pipeline(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        result, _ = (
            _plan(backend)
            .rows_filter(Col("age") >= 30)
            .col_drop("active")
            .math_multiply("salary", 1.1)
            .col_rename("salary", "adjusted_salary")
            .process(basic_rel)
        )
        assert backend.get_shape(result)[0] == 4
        assert "active" not in result.columns
        assert "adjusted_salary" in result.columns

    def test_protocol_tracking(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        _, protocol = (
            _plan(backend)
            .col_drop("active")
            .math_add("age", 1)
            .process(basic_rel)
        )
        assert protocol.input_hash is not None
        assert protocol._input_shape == (5, 5)
        assert len(protocol._steps) == 2

    def test_from_dict_duckdb(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        plan = _plan(backend).col_drop("active").math_add("age", 1)
        d = plan.to_dict()
        assert d["backend"] == "duckdb"

        restored = TransformPlan.from_dict(d)
        result, _ = restored.process(basic_rel)
        assert "active" not in result.columns


# =============================================================================
# Validation tests for DuckDB backend
# =============================================================================


class TestDuckDBValidation:
    """Tests for schema validation with DuckDB backend."""

    def test_validate_valid_plan(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """validate() works for a valid DuckDB plan."""
        plan = _plan(backend).math_add("age", 1)
        result = plan.validate(basic_rel)
        assert result.is_valid

    def test_validate_missing_column(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """validate() catches missing column for DuckDB."""
        plan = _plan(backend).math_add("nonexistent", 1)
        result = plan.validate(basic_rel)
        assert not result.is_valid
        assert "does not exist" in str(result.errors[0])

    def test_validate_wrong_type(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """validate() catches type mismatch for DuckDB."""
        plan = _plan(backend).math_add("name", 10)
        result = plan.validate(basic_rel)
        assert not result.is_valid
        assert "expected numeric" in str(result.errors[0])

    def test_validate_string_on_numeric(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """validate() catches string op on numeric column for DuckDB."""
        plan = _plan(backend).str_lower("age")
        result = plan.validate(basic_rel)
        assert not result.is_valid
        assert "expected string" in str(result.errors[0])

    def test_validate_multi_step(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """validate() tracks schema changes across steps for DuckDB."""
        plan = _plan(backend).col_drop("name").col_drop("name")
        result = plan.validate(basic_rel)
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_dry_run(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """dry_run() works for DuckDB."""
        plan = _plan(backend).col_drop("active").math_add("age", 1)
        preview = plan.dry_run(basic_rel)
        assert preview.is_valid
        assert len(preview.steps) == 2
        assert "active" in preview.steps[0].columns_removed

    def test_dry_run_invalid(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """dry_run() reports errors for DuckDB."""
        plan = _plan(backend).col_drop("nonexistent")
        preview = plan.dry_run(basic_rel)
        assert not preview.is_valid

    def test_process_with_validation(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """process(validate=True) works for DuckDB (was previously skipped)."""
        plan = _plan(backend).math_add("age", 1)
        result, _protocol = plan.process(basic_rel, validate=True)
        rows = result.fetchall()
        assert len(rows) == 5

    def test_process_validation_catches_error(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """process(validate=True) raises for invalid DuckDB plan."""
        from transformplan.validation import SchemaValidationError

        plan = _plan(backend).math_add("nonexistent", 1)
        with pytest.raises(SchemaValidationError):
            plan.process(basic_rel, validate=True)

    def test_validate_datetime_op(
        self, backend: DuckDBBackend, datetime_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """validate() works for datetime ops with DuckDB."""
        plan = _plan(backend).dt_year("date_col", "year")
        result = plan.validate(datetime_rel)
        assert result.is_valid

    def test_validate_col_rename_chain(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """validate() correctly tracks renames for DuckDB."""
        plan = _plan(backend).col_rename("age", "years").math_add("years", 1)
        result = plan.validate(basic_rel)
        assert result.is_valid


class TestCrossBackendSerialization:
    """Tests for cross-backend plan serialization."""

    def test_polars_plan_to_duckdb(
        self, backend: DuckDBBackend, basic_rel: duckdb.DuckDBPyRelation
    ) -> None:
        """Plan built with Polars can be serialized and run on DuckDB."""
        # Build plan with default (Polars) backend
        polars_plan = TransformPlan().col_drop("active").math_add("age", 1)
        d = polars_plan.to_dict()

        # Restore with DuckDB backend
        d.pop("backend", None)  # remove backend key if present
        d["backend"] = "duckdb"
        restored = TransformPlan.from_dict(d)
        result, _ = restored.process(basic_rel)
        assert "active" not in result.columns

    def test_duckdb_plan_to_polars(
        self, backend: DuckDBBackend
    ) -> None:
        """Plan built with DuckDB can be serialized and run on Polars."""
        import polars as pl

        # Build plan with DuckDB backend
        duckdb_plan = _plan(backend).col_drop("active").math_add("age", 1)
        d = duckdb_plan.to_dict()

        # Restore with default (Polars) backend
        d.pop("backend", None)  # force polars
        restored = TransformPlan.from_dict(d)

        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "salary": [50000.0, 60000.0, 70000.0],
            "active": [True, True, False],
        })
        result, _ = restored.process(df)
        assert "active" not in result.columns
        assert result["age"].to_list() == [26, 31, 36]
