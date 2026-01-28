# Row Operations

Operations for filtering, sorting, and transforming rows.

## Overview

Row operations modify which rows are included in the DataFrame and how they are ordered. Use the `Col` class to build filter expressions.

```python
from transformplan import TransformPlan, Col

plan = (
    TransformPlan()
    .rows_filter(Col("status") == "active")
    .rows_sort("created_at", descending=True)
    .rows_unique(columns=["email"])
)
```

## Class Reference

::: transformplan.ops.rows.RowOps
    options:
      show_root_heading: true
      members:
        - rows_filter
        - rows_drop
        - rows_drop_nulls
        - rows_unique
        - rows_deduplicate
        - rows_sort
        - rows_flag
        - rows_head
        - rows_tail
        - rows_sample
        - rows_explode
        - rows_melt
        - rows_pivot

## Examples

### Filtering Rows

```python
from transformplan import Col

# Keep rows matching condition
plan = TransformPlan().rows_filter(Col("age") >= 18)

# Drop rows matching condition
plan = TransformPlan().rows_drop(Col("status") == "deleted")

# Complex filters
plan = TransformPlan().rows_filter(
    (Col("score") >= 50) & (Col("active") == True)
)
```

### Flagging Rows

Add a boolean column based on a condition without removing rows:

```python
plan = TransformPlan().rows_flag(
    filter=Col("score") >= 90,
    new_column="is_excellent",
    true_value=True,
    false_value=False
)
```

### Sorting

```python
# Sort by single column
plan = TransformPlan().rows_sort("name")

# Sort descending
plan = TransformPlan().rows_sort("score", descending=True)

# Sort by multiple columns
plan = TransformPlan().rows_sort(
    by=["category", "price"],
    descending=[False, True]
)
```

### Removing Duplicates

```python
# Keep first occurrence of each unique value
plan = TransformPlan().rows_unique(columns=["email"])

# Keep last occurrence
plan = TransformPlan().rows_unique(columns=["user_id"], keep="last")

# Deduplicate with specific sort order
plan = TransformPlan().rows_deduplicate(
    columns=["user_id"],
    sort_by="updated_at",
    keep="last",
    descending=True
)
```

### Handling Nulls

```python
# Drop rows with nulls in any column
plan = TransformPlan().rows_drop_nulls()

# Drop rows with nulls in specific columns
plan = TransformPlan().rows_drop_nulls(columns=["required_field"])
```

### Limiting Rows

```python
# Keep first n rows
plan = TransformPlan().rows_head(10)

# Keep last n rows
plan = TransformPlan().rows_tail(10)

# Random sample
plan = TransformPlan().rows_sample(n=100, seed=42)
plan = TransformPlan().rows_sample(fraction=0.1, seed=42)
```

### Reshaping

```python
# Explode list column into multiple rows
plan = TransformPlan().rows_explode("tags")

# Unpivot from wide to long format
plan = TransformPlan().rows_melt(
    id_columns=["id", "name"],
    value_columns=["q1", "q2", "q3", "q4"],
    variable_name="quarter",
    value_name="sales"
)

# Pivot from long to wide format
plan = TransformPlan().rows_pivot(
    index=["id"],
    columns="quarter",
    values="sales",
    aggregate_function="sum"
)
```
