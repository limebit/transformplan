# Column Operations

Operations for adding, dropping, renaming, and transforming columns.

## Overview

Column operations modify the structure of a DataFrame by adding, removing, or transforming columns. All operations return the TransformPlan instance for method chaining.

```python
from transformplan import TransformPlan

plan = (
    TransformPlan()
    .col_rename("old_name", "new_name")
    .col_drop("temp_column")
    .col_cast("price", pl.Float64)
    .col_add("status", value="active")
)
```

## Class Reference

::: transformplan.ops.column.ColumnOps
    options:
      show_root_heading: true
      members:
        - col_drop
        - col_rename
        - col_cast
        - col_reorder
        - col_select
        - col_duplicate
        - col_fill_null
        - col_drop_null
        - col_drop_zero
        - col_add
        - col_add_uuid
        - col_hash
        - col_coalesce
        - col_expr

## Examples

### Basic Column Operations

```python
# Drop a column
plan = TransformPlan().col_drop("temp")

# Rename a column
plan = TransformPlan().col_rename("old", "new")

# Cast to a different type
plan = TransformPlan().col_cast("price", pl.Float64)
```

### Column Selection

```python
# Keep only specific columns (in order)
plan = TransformPlan().col_select(["id", "name", "value"])

# Reorder columns (drops unlisted columns)
plan = TransformPlan().col_reorder(["value", "name", "id"])
```

### Adding Columns

```python
# Add column with constant value
plan = TransformPlan().col_add("status", value="pending")

# Copy from existing column
plan = TransformPlan().col_add("price_backup", expr="price")

# Add unique identifiers
plan = TransformPlan().col_add_uuid("row_id", length=16)
```

### Handling Null Values

```python
# Fill nulls with a value
plan = TransformPlan().col_fill_null("score", value=0)

# Fill with strategy
plan = TransformPlan().col_fill_null("value", strategy="forward")

# Drop rows with nulls
plan = TransformPlan().col_drop_null(columns=["required_field"])
```

### Advanced Operations

```python
# Create hash from multiple columns
plan = TransformPlan().col_hash(
    columns=["first_name", "last_name", "email"],
    new_column="user_hash",
    salt="my_salt"
)

# Take first non-null from multiple columns
plan = TransformPlan().col_coalesce(
    columns=["primary_email", "secondary_email", "backup_email"],
    new_column="contact_email"
)

# Duplicate a column
plan = TransformPlan().col_duplicate("original", "copy")

# Add column from SQL expression (works on both backends)
plan = TransformPlan().col_expr(
    new_column="category",
    expr="CASE WHEN age > 30 THEN 'senior' ELSE 'junior' END",
)
```
