# Map Operations

Value mapping, discretization, and transformation operations.

## Overview

Map operations transform column values using dictionaries, bins, or other columns. They're useful for categorization, value replacement, and data normalization.

```python
from transformplan import TransformPlan

plan = (
    TransformPlan()
    .map_values("status", {"A": "Active", "I": "Inactive"})
    .map_discretize("age", bins=[18, 35, 55], labels=["Young", "Adult", "Senior"])
)
```

## Class Reference

::: transformplan.ops.map.MapOps
    options:
      show_root_heading: true
      members:
        - map_values
        - map_discretize
        - map_case
        - map_bool_to_int
        - map_null_to_value
        - map_value_to_null
        - map_from_column

## Examples

### Dictionary Mapping

```python
# Map values using a dictionary
plan = TransformPlan().map_values(
    column="country_code",
    mapping={"US": "United States", "CA": "Canada", "MX": "Mexico"}
)

# With default for unmapped values
plan = TransformPlan().map_values(
    column="status",
    mapping={"A": "Active", "I": "Inactive"},
    default="Unknown",
    keep_unmapped=False
)
```

### Discretization (Binning)

```python
# Discretize numeric values into categories
plan = TransformPlan().map_discretize(
    column="age",
    bins=[0, 18, 35, 55, 100],
    labels=["Child", "Young Adult", "Adult", "Senior"],
    new_column="age_group"
)

# Auto-generated labels
plan = TransformPlan().map_discretize(
    column="score",
    bins=[0, 50, 75, 100],
    new_column="score_band"
)
```

### Case-When Logic

```python
# Apply case-when transformations
plan = TransformPlan().map_case(
    column="score",
    cases=[
        (90, "A"),
        (80, "B"),
        (70, "C"),
        (60, "D"),
    ],
    default="F",
    new_column="grade"
)
```

### Null Handling

```python
# Replace null with a value
plan = TransformPlan().map_null_to_value("status", "Unknown")

# Replace a value with null
plan = TransformPlan().map_value_to_null("status", "N/A")
```

### Type Conversion

```python
# Convert boolean to integer
plan = TransformPlan().map_bool_to_int("is_active")
# True -> 1, False -> 0
```

### Column-based Lookup

```python
# Map using values from other columns (vlookup-style)
plan = TransformPlan().map_from_column(
    column="category_id",
    lookup_column="category_id",
    value_column="category_name",
    new_column="category_label",
    default="Unknown"
)
```

## Use Cases

### Categorizing Continuous Data

```python
# Income brackets
plan = TransformPlan().map_discretize(
    column="income",
    bins=[0, 30000, 60000, 100000, 200000],
    labels=["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
    new_column="income_bracket"
)
```

### Standardizing Codes

```python
# Standardize department codes
plan = TransformPlan().map_values(
    column="dept",
    mapping={
        "ENG": "Engineering",
        "MKT": "Marketing",
        "SAL": "Sales",
        "HR": "Human Resources"
    }
)
```

### Data Cleaning

```python
# Replace sentinel values with null
plan = TransformPlan().map_value_to_null("score", -999)

# Replace null with default
plan = TransformPlan().map_null_to_value("category", "Uncategorized")
```
