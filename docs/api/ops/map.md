# Map Operations

Value mapping, discretization, encoding, and transformation operations.

## Overview

Map operations transform column values using dictionaries, bins, or encoding schemes. They're useful for categorization, value replacement, data normalization, and ML feature preparation.

```python
from transformplan import TransformPlan

plan = (
    TransformPlan()
    .map_values("status", {"A": "Active", "I": "Inactive"})
    .map_discretize("age", bins=[18, 35, 55], labels=["Young", "Adult", "Senior"])
    .map_onehot("color", categories=["red", "green", "blue"], drop="first")
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
        - map_onehot
        - map_ordinal
        - map_label

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

### One-Hot Encoding

```python
# Basic one-hot encoding
plan = TransformPlan().map_onehot(
    column="color",
    categories=["red", "green", "blue"]
)
# Creates columns: color_red, color_green, color_blue

# Drop first category to avoid multicollinearity (for regression models)
plan = TransformPlan().map_onehot(
    column="color",
    categories=["red", "green", "blue"],
    drop="first"
)
# Creates columns: color_green, color_blue (drops color_red)
```

### Ordinal Encoding

```python
# Ordinal encoding with meaningful order
plan = TransformPlan().map_ordinal(
    column="size",
    categories=["small", "medium", "large"]
)
# Maps: small -> 0, medium -> 1, large -> 2
```

### Label Encoding

```python
# Label encoding (alphabetically sorted by default)
plan = TransformPlan().map_label(column="department")
# Maps alphabetically: Engineering -> 0, HR -> 1, Sales -> 2
```

### ML Feature Preparation

```python
# One-hot encode categorical features, dropping first to avoid multicollinearity
plan = (
    TransformPlan()
    .map_onehot("color", categories=["red", "green", "blue"], drop="first")
    .map_ordinal("quality", categories=["low", "medium", "high"])
)
```
