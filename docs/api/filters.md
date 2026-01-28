# Filters

Serializable filter expressions for row filtering operations.

## Overview

The filter system provides a way to build complex filter conditions that can be serialized to JSON and deserialized back. This enables reproducible pipelines that can be saved and shared.

```python
from transformplan import Col, Filter

# Build a filter
filter_expr = (Col("age") >= 18) & (Col("status") == "active")

# Use in pipeline
plan = TransformPlan().rows_filter(filter_expr)

# Serialize
filter_dict = filter_expr.to_dict()

# Deserialize
restored = Filter.from_dict(filter_dict)
```

## Col Class

::: transformplan.filters.Col
    options:
      show_root_heading: true
      members_order: source

## Filter Base Class

::: transformplan.filters.Filter
    options:
      show_root_heading: true
      members:
        - to_expr
        - to_dict
        - from_dict

## Comparison Filters

### Eq (Equal)

::: transformplan.filters.Eq
    options:
      show_root_heading: true

### Ne (Not Equal)

::: transformplan.filters.Ne
    options:
      show_root_heading: true

### Gt (Greater Than)

::: transformplan.filters.Gt
    options:
      show_root_heading: true

### Ge (Greater Than or Equal)

::: transformplan.filters.Ge
    options:
      show_root_heading: true

### Lt (Less Than)

::: transformplan.filters.Lt
    options:
      show_root_heading: true

### Le (Less Than or Equal)

::: transformplan.filters.Le
    options:
      show_root_heading: true

### IsIn

::: transformplan.filters.IsIn
    options:
      show_root_heading: true

### Between

::: transformplan.filters.Between
    options:
      show_root_heading: true

## Null Filters

### IsNull

::: transformplan.filters.IsNull
    options:
      show_root_heading: true

### IsNotNull

::: transformplan.filters.IsNotNull
    options:
      show_root_heading: true

## String Filters

### StrContains

::: transformplan.filters.StrContains
    options:
      show_root_heading: true

### StrStartsWith

::: transformplan.filters.StrStartsWith
    options:
      show_root_heading: true

### StrEndsWith

::: transformplan.filters.StrEndsWith
    options:
      show_root_heading: true

## Logical Combinators

### And

::: transformplan.filters.And
    options:
      show_root_heading: true

### Or

::: transformplan.filters.Or
    options:
      show_root_heading: true

### Not

::: transformplan.filters.Not
    options:
      show_root_heading: true

## Examples

### Simple Comparisons

```python
from transformplan import Col

# Numeric comparisons
Col("age") >= 18
Col("price") < 100
Col("quantity") == 0

# String equality
Col("status") == "active"
Col("country") != "US"
```

### String Matching

```python
# Contains substring
Col("email").str_contains("@company.com")

# Starts/ends with
Col("code").str_starts_with("PRD-")
Col("filename").str_ends_with(".csv")
```

### Membership Tests

```python
# Check if value is in list
Col("status").is_in(["active", "pending"])

# Range check
Col("age").between(18, 65)
```

### Null Checks

```python
# Filter nulls
Col("email").is_not_null()
Col("optional_field").is_null()
```

### Combining Conditions

```python
# AND: both conditions must be true
(Col("age") >= 18) & (Col("status") == "active")

# OR: at least one condition must be true
(Col("role") == "admin") | (Col("role") == "moderator")

# NOT: invert condition
~(Col("deleted") == True)

# Complex combinations
(
    (Col("age") >= 18) &
    (Col("country").is_in(["US", "CA"])) &
    ~(Col("status") == "banned")
)
```
