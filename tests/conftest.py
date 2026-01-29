"""Shared pytest fixtures for TransformPlan tests."""

from datetime import date, datetime

import polars as pl
import pytest


@pytest.fixture
def basic_df() -> pl.DataFrame:
    """Basic DataFrame with common column types."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "active": [True, True, False, True, False],
        }
    )


@pytest.fixture
def df_with_nulls() -> pl.DataFrame:
    """DataFrame with null values in various columns."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", None, "Charlie", "David", None],
            "age": [25, 30, None, 40, 45],
            "salary": [50000.0, None, 70000.0, None, 90000.0],
            "active": [True, True, None, True, False],
        }
    )


@pytest.fixture
def numeric_df() -> pl.DataFrame:
    """DataFrame for math operations."""
    return pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [100, 200, 300, 400, 500],
        }
    )


@pytest.fixture
def string_df() -> pl.DataFrame:
    """DataFrame for string operations."""
    return pl.DataFrame(
        {
            "text": [
                "  Hello World  ",
                "foo bar baz",
                "ABC123xyz",
                "test@email.com",
                "",
            ],
            "code": ["PRD-001", "PRD-002", "TST-001", "DEV-001", "PRD-003"],
            "first_name": ["John", "Jane", "Bob", "Alice", "Charlie"],
            "last_name": ["Doe", "Smith", "Brown", "Johnson", "Williams"],
        }
    )


@pytest.fixture
def datetime_df() -> pl.DataFrame:
    """DataFrame for datetime operations."""
    return pl.DataFrame(
        {
            "date_col": [
                date(2024, 1, 15),
                date(2024, 3, 20),
                date(2024, 6, 10),
                date(2024, 9, 5),
                date(2024, 12, 25),
            ],
            "datetime_col": [
                datetime(2024, 1, 15, 10, 30, 0),
                datetime(2024, 3, 20, 14, 45, 0),
                datetime(2024, 6, 10, 8, 15, 0),
                datetime(2024, 9, 5, 16, 0, 0),
                datetime(2024, 12, 25, 12, 0, 0),
            ],
            "date_str": [
                "2024-01-15",
                "2024-03-20",
                "2024-06-10",
                "2024-09-05",
                "2024-12-25",
            ],
            "birth_date": [
                date(1990, 5, 10),
                date(1985, 8, 22),
                date(2000, 1, 1),
                date(1975, 12, 15),
                date(1995, 3, 30),
            ],
        }
    )


@pytest.fixture
def list_df() -> pl.DataFrame:
    """DataFrame with list column for explode tests."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "tags": [["a", "b", "c"], ["d", "e"], ["f"]],
            "name": ["Item1", "Item2", "Item3"],
        }
    )


@pytest.fixture
def wide_df() -> pl.DataFrame:
    """Wide DataFrame for pivot/melt tests."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "q1": [100, 150, 200],
            "q2": [110, 160, 210],
            "q3": [120, 170, 220],
            "q4": [130, 180, 230],
        }
    )


@pytest.fixture
def long_df() -> pl.DataFrame:
    """Long DataFrame for pivot tests."""
    return pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "quarter": ["Q1", "Q2", "Q1", "Q2", "Q1", "Q2"],
            "value": [100, 110, 150, 160, 200, 210],
        }
    )


@pytest.fixture
def empty_df() -> pl.DataFrame:
    """Empty DataFrame edge case."""
    return pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "name": pl.Series([], dtype=pl.Utf8),
            "value": pl.Series([], dtype=pl.Float64),
        }
    )


@pytest.fixture
def single_row_df() -> pl.DataFrame:
    """Single row DataFrame edge case."""
    return pl.DataFrame(
        {
            "id": [1],
            "name": ["Alice"],
            "value": [100.0],
        }
    )


@pytest.fixture
def duplicates_df() -> pl.DataFrame:
    """DataFrame with duplicate rows for deduplication tests."""
    return pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "name": ["Alice", "Alice", "Bob", "Bob", "Charlie"],
            "timestamp": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 2, 10, 0),
                datetime(2024, 1, 2, 9, 0),
                datetime(2024, 1, 3, 10, 0),
            ],
            "value": [100, 150, 200, 180, 300],
        }
    )


@pytest.fixture
def map_df() -> pl.DataFrame:
    """DataFrame for map operations."""
    return pl.DataFrame(
        {
            "status": ["A", "B", "A", "C", "B"],
            "score": [85, 72, 91, 68, 79],
            "category": ["X", "Y", "X", "Z", "Y"],
            "lookup_key": [1, 2, 1, 3, 2],
            "lookup_value": ["One", "Two", "One", "Three", "Two"],
        }
    )
