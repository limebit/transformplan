# Datetime Operations

Date and time extraction and manipulation operations.

## Overview

Datetime operations allow you to extract components from date/datetime columns, parse date strings, and perform date arithmetic.

```python
from transformplan import TransformPlan

plan = (
    TransformPlan()
    .dt_parse("date_string", fmt="%Y-%m-%d")
    .dt_year("order_date", new_column="order_year")
    .dt_diff_days("end_date", "start_date", new_column="duration")
)
```

## Class Reference

::: transformplan.ops.datetime.DatetimeOps
    options:
      show_root_heading: true
      members:
        - dt_year
        - dt_month
        - dt_day
        - dt_week
        - dt_quarter
        - dt_year_month
        - dt_quarter_year
        - dt_calendar_week
        - dt_parse
        - dt_format
        - dt_diff_days
        - dt_age_years
        - dt_truncate
        - dt_is_between

## Examples

### Extracting Date Components

```python
# Extract year
plan = TransformPlan().dt_year("date", new_column="year")

# Extract month
plan = TransformPlan().dt_month("date", new_column="month")

# Extract day
plan = TransformPlan().dt_day("date", new_column="day")

# Extract week number
plan = TransformPlan().dt_week("date", new_column="week")

# Extract quarter
plan = TransformPlan().dt_quarter("date", new_column="quarter")
```

### Formatted Date Strings

```python
# Year-month string (e.g., "2024-01")
plan = TransformPlan().dt_year_month("date", new_column="year_month")

# Quarter-year string (e.g., "Q1-2024")
plan = TransformPlan().dt_quarter_year("date", new_column="quarter_year")

# Calendar week string (e.g., "2024-W05")
plan = TransformPlan().dt_calendar_week("date", new_column="calendar_week")
```

### Parsing and Formatting

```python
# Parse string to date
plan = TransformPlan().dt_parse(
    column="date_string",
    fmt="%Y-%m-%d",
    new_column="date"
)

# Format date to string
plan = TransformPlan().dt_format(
    column="date",
    fmt="%B %d, %Y",
    new_column="formatted_date"
)
```

### Date Arithmetic

```python
# Calculate difference in days
plan = TransformPlan().dt_diff_days(
    column_a="end_date",
    column_b="start_date",
    new_column="duration_days"
)

# Calculate age in years
plan = TransformPlan().dt_age_years(
    birth_column="birth_date",
    new_column="age"
)

# Age relative to reference column
plan = TransformPlan().dt_age_years(
    birth_column="birth_date",
    reference_column="event_date",
    new_column="age_at_event"
)
```

### Truncation

```python
# Truncate to month start
plan = TransformPlan().dt_truncate("timestamp", every="1mo")

# Truncate to day
plan = TransformPlan().dt_truncate("timestamp", every="1d")

# Truncate to year
plan = TransformPlan().dt_truncate("timestamp", every="1y")
```

### Range Checks

```python
# Check if date is within range
plan = TransformPlan().dt_is_between(
    column="order_date",
    start="2024-01-01",
    end="2024-12-31",
    new_column="is_2024_order",
    closed="both"  # or "left", "right", "none"
)
```
