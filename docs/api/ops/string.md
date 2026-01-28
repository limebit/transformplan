# String Operations

Text manipulation operations on string columns.

## Overview

String operations allow you to transform text data in DataFrame columns. Operations include case conversion, trimming, splitting, concatenation, and pattern matching.

```python
from transformplan import TransformPlan

plan = (
    TransformPlan()
    .str_lower("email")
    .str_strip("name")
    .str_replace("phone", "-", "")
)
```

## Class Reference

::: transformplan.ops.string.StrOps
    options:
      show_root_heading: true
      members:
        - str_replace
        - str_slice
        - str_truncate
        - str_lower
        - str_upper
        - str_strip
        - str_pad
        - str_split
        - str_concat
        - str_extract

## Examples

### Case Conversion

```python
# Convert to lowercase
plan = TransformPlan().str_lower("email")

# Convert to uppercase
plan = TransformPlan().str_upper("code")
```

### Trimming and Padding

```python
# Strip whitespace
plan = TransformPlan().str_strip("name")

# Strip specific characters
plan = TransformPlan().str_strip("code", chars="-_")

# Pad to fixed length
plan = TransformPlan().str_pad("id", length=10, fill_char="0", side="left")
```

### Replacement

```python
# Replace literal string
plan = TransformPlan().str_replace("phone", "-", "")

# Replace with regex
plan = TransformPlan().str_replace(
    column="text",
    pattern=r"\s+",
    replacement=" ",
    literal=False
)
```

### Substring Operations

```python
# Extract substring by position
plan = TransformPlan().str_slice("code", offset=0, length=3)

# Truncate with suffix
plan = TransformPlan().str_truncate("description", max_length=100, suffix="...")
```

### Splitting

```python
# Split into rows (explode)
plan = TransformPlan().str_split("tags", separator=",")

# Split into columns
plan = TransformPlan().str_split(
    column="full_name",
    separator=" ",
    new_columns=["first_name", "last_name"],
    keep_original=False
)
```

### Concatenation

```python
# Concatenate columns
plan = TransformPlan().str_concat(
    columns=["first_name", "last_name"],
    new_column="full_name",
    separator=" "
)
```

### Pattern Extraction

```python
# Extract with regex capture group
plan = TransformPlan().str_extract(
    column="email",
    pattern=r"@(.+)$",
    group_index=1,
    new_column="domain"
)
```
