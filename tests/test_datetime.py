"""Tests for datetime operations (ops/datetime.py)."""

from datetime import date

import polars as pl

from transformplan import TransformPlan


class TestDtYear:
    """Tests for dt_year operation."""

    def test_dt_year_new_column(self, datetime_df: pl.DataFrame) -> None:
        """Test extracting year to new column."""
        plan = TransformPlan().dt_year("date_col", "year")
        result, _ = plan.process(datetime_df)
        assert "year" in result.columns
        assert result["year"].to_list() == [2024, 2024, 2024, 2024, 2024]

    def test_dt_year_in_place(self, datetime_df: pl.DataFrame) -> None:
        """Test extracting year in place."""
        plan = TransformPlan().dt_year("date_col")
        result, _ = plan.process(datetime_df)
        assert result["date_col"].to_list() == [2024, 2024, 2024, 2024, 2024]

    def test_dt_year_nonexistent_column_raises(self, datetime_df: pl.DataFrame) -> None:
        """Test that nonexistent column fails validation."""
        plan = TransformPlan().dt_year("nonexistent")
        result = plan.validate(datetime_df)
        assert not result.is_valid

    def test_dt_year_non_datetime_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that non-datetime column fails validation."""
        plan = TransformPlan().dt_year("name")
        result = plan.validate(basic_df)
        assert not result.is_valid


class TestDtMonth:
    """Tests for dt_month operation."""

    def test_dt_month_new_column(self, datetime_df: pl.DataFrame) -> None:
        """Test extracting month to new column."""
        plan = TransformPlan().dt_month("date_col", "month")
        result, _ = plan.process(datetime_df)
        assert "month" in result.columns
        assert result["month"].to_list() == [1, 3, 6, 9, 12]


class TestDtDay:
    """Tests for dt_day operation."""

    def test_dt_day_new_column(self, datetime_df: pl.DataFrame) -> None:
        """Test extracting day to new column."""
        plan = TransformPlan().dt_day("date_col", "day")
        result, _ = plan.process(datetime_df)
        assert "day" in result.columns
        assert result["day"].to_list() == [15, 20, 10, 5, 25]


class TestDtWeek:
    """Tests for dt_week operation."""

    def test_dt_week_new_column(self, datetime_df: pl.DataFrame) -> None:
        """Test extracting ISO week to new column."""
        plan = TransformPlan().dt_week("date_col", "week")
        result, _ = plan.process(datetime_df)
        assert "week" in result.columns
        # Week numbers for the dates
        assert all(1 <= w <= 53 for w in result["week"].to_list())


class TestDtQuarter:
    """Tests for dt_quarter operation."""

    def test_dt_quarter_new_column(self, datetime_df: pl.DataFrame) -> None:
        """Test extracting quarter to new column."""
        plan = TransformPlan().dt_quarter("date_col", "quarter")
        result, _ = plan.process(datetime_df)
        assert "quarter" in result.columns
        # Jan=Q1, Mar=Q1, Jun=Q2, Sep=Q3, Dec=Q4
        assert result["quarter"].to_list() == [1, 1, 2, 3, 4]


class TestDtYearMonth:
    """Tests for dt_year_month operation."""

    def test_dt_year_month_default_format(self, datetime_df: pl.DataFrame) -> None:
        """Test year-month formatting with default format."""
        plan = TransformPlan().dt_year_month("date_col", "ym")
        result, _ = plan.process(datetime_df)
        assert "ym" in result.columns
        assert result["ym"].to_list() == ["2024-01", "2024-03", "2024-06", "2024-09", "2024-12"]

    def test_dt_year_month_custom_format(self, datetime_df: pl.DataFrame) -> None:
        """Test year-month formatting with custom format."""
        plan = TransformPlan().dt_year_month("date_col", "ym", fmt="%Y/%m")
        result, _ = plan.process(datetime_df)
        assert result["ym"].to_list() == ["2024/01", "2024/03", "2024/06", "2024/09", "2024/12"]


class TestDtQuarterYear:
    """Tests for dt_quarter_year operation."""

    def test_dt_quarter_year(self, datetime_df: pl.DataFrame) -> None:
        """Test quarter-year formatting."""
        plan = TransformPlan().dt_quarter_year("date_col", "qy")
        result, _ = plan.process(datetime_df)
        assert "qy" in result.columns
        assert result["qy"].to_list() == ["Q1-2024", "Q1-2024", "Q2-2024", "Q3-2024", "Q4-2024"]


class TestDtCalendarWeek:
    """Tests for dt_calendar_week operation."""

    def test_dt_calendar_week(self, datetime_df: pl.DataFrame) -> None:
        """Test calendar week formatting."""
        plan = TransformPlan().dt_calendar_week("date_col", "cw")
        result, _ = plan.process(datetime_df)
        assert "cw" in result.columns
        # Should be in format YYYY-WXX
        assert all(cw.startswith("2024-W") for cw in result["cw"].to_list())


class TestDtParse:
    """Tests for dt_parse operation."""

    def test_dt_parse_default_format(self, datetime_df: pl.DataFrame) -> None:
        """Test parsing with default format."""
        plan = TransformPlan().dt_parse("date_str", new_column="parsed")
        result, _ = plan.process(datetime_df)
        assert "parsed" in result.columns
        assert result["parsed"].dtype == pl.Date
        assert result["parsed"].to_list() == datetime_df["date_col"].to_list()

    def test_dt_parse_custom_format(self) -> None:
        """Test parsing with custom format."""
        df = pl.DataFrame({"date_str": ["15/01/2024", "20/03/2024"]})
        plan = TransformPlan().dt_parse("date_str", fmt="%d/%m/%Y", new_column="parsed")
        result, _ = plan.process(df)
        assert result["parsed"].to_list() == [date(2024, 1, 15), date(2024, 3, 20)]

    def test_dt_parse_in_place(self, datetime_df: pl.DataFrame) -> None:
        """Test parsing in place."""
        plan = TransformPlan().dt_parse("date_str")
        result, _ = plan.process(datetime_df)
        assert result["date_str"].dtype == pl.Date


class TestDtFormat:
    """Tests for dt_format operation."""

    def test_dt_format_basic(self, datetime_df: pl.DataFrame) -> None:
        """Test formatting date as string."""
        plan = TransformPlan().dt_format("date_col", "%d/%m/%Y", new_column="formatted")
        result, _ = plan.process(datetime_df)
        assert "formatted" in result.columns
        assert result["formatted"].dtype == pl.Utf8
        assert result["formatted"][0] == "15/01/2024"

    def test_dt_format_in_place(self, datetime_df: pl.DataFrame) -> None:
        """Test formatting in place."""
        plan = TransformPlan().dt_format("date_col", "%Y%m%d")
        result, _ = plan.process(datetime_df)
        assert result["date_col"].dtype == pl.Utf8
        assert result["date_col"][0] == "20240115"


class TestDtDiffDays:
    """Tests for dt_diff_days operation."""

    def test_dt_diff_days_basic(self, datetime_df: pl.DataFrame) -> None:
        """Test calculating difference in days."""
        plan = TransformPlan().dt_diff_days("date_col", "birth_date", "days_since_birth")
        result, _ = plan.process(datetime_df)
        assert "days_since_birth" in result.columns
        # All differences should be positive (dates are after birth dates)
        assert all(d > 0 for d in result["days_since_birth"].to_list())

    def test_dt_diff_days_order_matters(self) -> None:
        """Test that order of columns matters."""
        df = pl.DataFrame(
            {
                "start": [date(2024, 1, 1)],
                "end": [date(2024, 1, 10)],
            }
        )
        plan = TransformPlan().dt_diff_days("end", "start", "diff")
        result, _ = plan.process(df)
        assert result["diff"][0] == 9


class TestDtAgeYears:
    """Tests for dt_age_years operation."""

    def test_dt_age_years_from_today(self, datetime_df: pl.DataFrame) -> None:
        """Test calculating age from today."""
        plan = TransformPlan().dt_age_years("birth_date", new_column="age")
        result, _ = plan.process(datetime_df)
        assert "age" in result.columns
        # All ages should be positive
        assert all(age >= 0 for age in result["age"].to_list())

    def test_dt_age_years_from_reference(self, datetime_df: pl.DataFrame) -> None:
        """Test calculating age from reference date."""
        plan = TransformPlan().dt_age_years(
            "birth_date", reference_column="date_col", new_column="age_at_date"
        )
        result, _ = plan.process(datetime_df)
        assert "age_at_date" in result.columns


class TestDtTruncate:
    """Tests for dt_truncate operation."""

    def test_dt_truncate_to_month(self, datetime_df: pl.DataFrame) -> None:
        """Test truncating to month."""
        plan = TransformPlan().dt_truncate("date_col", "1mo", new_column="month_start")
        result, _ = plan.process(datetime_df)
        assert "month_start" in result.columns
        # All dates should be first of month
        assert all(d.day == 1 for d in result["month_start"].to_list())

    def test_dt_truncate_to_year(self, datetime_df: pl.DataFrame) -> None:
        """Test truncating to year."""
        plan = TransformPlan().dt_truncate("date_col", "1y", new_column="year_start")
        result, _ = plan.process(datetime_df)
        # All dates should be Jan 1
        for d in result["year_start"].to_list():
            assert d.month == 1
            assert d.day == 1


class TestDtIsBetween:
    """Tests for dt_is_between operation."""

    def test_dt_is_between_basic(self, datetime_df: pl.DataFrame) -> None:
        """Test checking if date is between range."""
        plan = TransformPlan().dt_is_between(
            "date_col", "2024-03-01", "2024-09-30", "in_range"
        )
        result, _ = plan.process(datetime_df)
        assert "in_range" in result.columns
        assert result["in_range"].dtype == pl.Boolean
        # Mar, Jun, Sep are in range; Jan, Dec are out
        assert result["in_range"].to_list() == [False, True, True, True, False]

    def test_dt_is_between_closed_both(self, datetime_df: pl.DataFrame) -> None:
        """Test with both endpoints included."""
        plan = TransformPlan().dt_is_between(
            "date_col", "2024-01-15", "2024-12-25", "in_range", closed="both"
        )
        result, _ = plan.process(datetime_df)
        # Both endpoints should be included
        assert all(result["in_range"].to_list())


class TestDatetimeChaining:
    """Tests for chaining datetime operations."""

    def test_parse_then_extract(self, datetime_df: pl.DataFrame) -> None:
        """Test parsing then extracting components."""
        # dt_parse converts string to date in place or to new column
        # Then we can extract year/month from the parsed column
        plan = (
            TransformPlan()
            .dt_parse("date_str")  # Parses in place, converts date_str to Date type
            .dt_year("date_str", "year")
            .dt_month("date_str", "month")
        )
        result, _ = plan.process(datetime_df)
        assert "year" in result.columns
        assert "month" in result.columns

    def test_multiple_extractions(self, datetime_df: pl.DataFrame) -> None:
        """Test extracting multiple components from same column."""
        plan = (
            TransformPlan()
            .dt_year("date_col", "year")
            .dt_month("date_col", "month")
            .dt_day("date_col", "day")
            .dt_quarter("date_col", "quarter")
        )
        result, _ = plan.process(datetime_df)
        assert "year" in result.columns
        assert "month" in result.columns
        assert "day" in result.columns
        assert "quarter" in result.columns


class TestDatetimeEdgeCases:
    """Tests for edge cases in datetime operations."""

    def test_leap_year(self) -> None:
        """Test operations with leap year date."""
        df = pl.DataFrame({"date": [date(2024, 2, 29)]})
        plan = TransformPlan().dt_day("date", "day")
        result, _ = plan.process(df)
        assert result["day"][0] == 29

    def test_year_boundaries(self) -> None:
        """Test operations at year boundaries."""
        df = pl.DataFrame(
            {
                "date": [date(2023, 12, 31), date(2024, 1, 1)],
            }
        )
        plan = TransformPlan().dt_year("date", "year")
        result, _ = plan.process(df)
        assert result["year"].to_list() == [2023, 2024]

    def test_diff_across_years(self) -> None:
        """Test date difference across year boundary."""
        df = pl.DataFrame(
            {
                "start": [date(2023, 12, 25)],
                "end": [date(2024, 1, 5)],
            }
        )
        plan = TransformPlan().dt_diff_days("end", "start", "diff")
        result, _ = plan.process(df)
        assert result["diff"][0] == 11
