# Encoding Operations

Categorical encoding operations for machine learning preparation.

## Overview

Encoding operations transform categorical columns into numeric representations suitable for machine learning models. They support one-hot encoding, ordinal encoding, and label encoding.

```python
from transformplan import TransformPlan

plan = (
    TransformPlan()
    .enc_onehot("color", categories=["red", "green", "blue"], drop="first")
    .enc_ordinal("size", categories=["small", "medium", "large"])
)
```

## Class Reference

::: transformplan.ops.encoding.EncodingOps
    options:
      show_root_heading: true
      members:
        - enc_onehot
        - enc_ordinal
        - enc_label

## Examples

### One-Hot Encoding

Creates binary indicator columns (0/1) for each category.

```python
# Basic one-hot encoding
plan = TransformPlan().enc_onehot(
    column="color",
    categories=["red", "green", "blue"]
)
# Creates columns: color_red, color_green, color_blue

# Drop first category to avoid multicollinearity (for regression models)
plan = TransformPlan().enc_onehot(
    column="color",
    categories=["red", "green", "blue"],
    drop="first"
)
# Creates columns: color_green, color_blue (drops color_red)

# Drop last category
plan = TransformPlan().enc_onehot(
    column="color",
    categories=["red", "green", "blue"],
    drop="last"
)
# Creates columns: color_red, color_green (drops color_blue)

# Drop specific category
plan = TransformPlan().enc_onehot(
    column="color",
    categories=["red", "green", "blue"],
    drop="green"
)
# Creates columns: color_red, color_blue (drops color_green)

# Custom prefix for new columns
plan = TransformPlan().enc_onehot(
    column="color",
    categories=["red", "green", "blue"],
    prefix="c"
)
# Creates columns: c_red, c_green, c_blue

# Keep original column
plan = TransformPlan().enc_onehot(
    column="color",
    categories=["red", "green", "blue"],
    drop_original=False
)
# Keeps color column alongside color_red, color_green, color_blue
```

### Ordinal Encoding

Maps categories to integers based on explicit ordering (first=0, second=1, etc.).

```python
# Ordinal encoding with meaningful order
plan = TransformPlan().enc_ordinal(
    column="size",
    categories=["small", "medium", "large"]
)
# Maps: small -> 0, medium -> 1, large -> 2

# Output to new column
plan = TransformPlan().enc_ordinal(
    column="size",
    categories=["small", "medium", "large"],
    new_column="size_encoded"
)

# Custom unknown value
plan = TransformPlan().enc_ordinal(
    column="size",
    categories=["small", "medium", "large"],
    unknown_value=-1  # Default
)
# Values not in categories get -1
```

### Label Encoding

Simple integer encoding, alphabetically sorted by default. Similar to ordinal encoding but without semantic ordering.

```python
# Label encoding (alphabetically sorted)
plan = TransformPlan().enc_label(column="department")
# Maps alphabetically: Engineering -> 0, HR -> 1, Sales -> 2

# With explicit categories
plan = TransformPlan().enc_label(
    column="department",
    categories=["HR", "Engineering", "Sales"]
)
# Maps: HR -> 0, Engineering -> 1, Sales -> 2
```

## Use Cases

### Preparing Data for Machine Learning

```python
# One-hot encode categorical features, dropping first to avoid multicollinearity
plan = (
    TransformPlan()
    .enc_onehot("color", categories=["red", "green", "blue"], drop="first")
    .enc_onehot("size", categories=["S", "M", "L", "XL"], drop="first")
    .enc_ordinal("quality", categories=["low", "medium", "high"])
)
```

### Handling Unknown Categories

```python
# Unknown values get all zeros (one-hot)
plan = TransformPlan().enc_onehot(
    column="color",
    categories=["red", "green", "blue"],
    unknown_value="all_zero"  # Default
)

# Unknown values get -1 (ordinal/label)
plan = TransformPlan().enc_ordinal(
    column="size",
    categories=["small", "medium", "large"],
    unknown_value=-1
)
```

### Deriving Categories from Data

When categories are not specified, they are derived from the data (sorted alphabetically):

```python
# Categories derived from data
plan = TransformPlan().enc_onehot("color")
# Uses sorted unique values from the column

# Note: For reproducibility, explicitly specify categories
plan = TransformPlan().enc_onehot(
    column="color",
    categories=["blue", "green", "red"]  # Explicit is better
)
```

## Multicollinearity Note

When using one-hot encoding for linear models (regression, logistic regression), you should drop one category to avoid the [dummy variable trap](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)). Use the `drop` parameter:

```python
# For regression models, drop one category
plan = TransformPlan().enc_onehot("color", drop="first")

# Tree-based models (random forest, XGBoost) don't require this
plan = TransformPlan().enc_onehot("color")  # Keep all
```
