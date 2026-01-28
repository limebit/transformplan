# Math Operations

Arithmetic and numeric operations on DataFrame columns.

## Overview

Math operations perform arithmetic on numeric columns. They support both scalar operations (column with a constant) and column-wise operations (column with column).

```python
from transformplan import TransformPlan

plan = (
    TransformPlan()
    .math_multiply("price", 1.1)  # 10% increase
    .math_round("price", decimals=2)
    .math_add_columns("subtotal", "tax", "total")
)
```

## Class Reference

::: transformplan.ops.math.MathOps
    options:
      show_root_heading: true
      members:
        - math_add
        - math_subtract
        - math_multiply
        - math_divide
        - math_clamp
        - math_set_min
        - math_set_max
        - math_abs
        - math_round
        - math_add_columns
        - math_subtract_columns
        - math_multiply_columns
        - math_divide_columns
        - math_percent_of
        - math_cumsum
        - math_rank

## Examples

### Scalar Operations

```python
# Add to every value
plan = TransformPlan().math_add("price", 10)

# Subtract from every value
plan = TransformPlan().math_subtract("score", 5)

# Multiply every value
plan = TransformPlan().math_multiply("quantity", 1.5)

# Divide every value
plan = TransformPlan().math_divide("total", 100)
```

### Column-wise Operations

```python
# Add two columns into a new column
plan = TransformPlan().math_add_columns("base", "bonus", "total")

# Subtract columns
plan = TransformPlan().math_subtract_columns("revenue", "cost", "profit")

# Multiply columns
plan = TransformPlan().math_multiply_columns("price", "quantity", "total")

# Divide columns
plan = TransformPlan().math_divide_columns("score", "max_score", "percentage")
```

### Value Clamping

```python
# Clamp to range
plan = TransformPlan().math_clamp("score", lower=0, upper=100)

# Set minimum value
plan = TransformPlan().math_set_min("quantity", min_value=0)

# Set maximum value
plan = TransformPlan().math_set_max("discount", max_value=50)
```

### Transformations

```python
# Absolute value
plan = TransformPlan().math_abs("difference")

# Round to decimal places
plan = TransformPlan().math_round("price", decimals=2)
```

### Percentage Calculation

```python
# Calculate percentage
plan = TransformPlan().math_percent_of(
    column="part",
    total_column="whole",
    new_column="percentage",
    multiply_by=100  # default
)
```

### Cumulative and Ranking Operations

```python
# Cumulative sum
plan = TransformPlan().math_cumsum(
    column="sales",
    new_column="running_total",
    group_by="region"
)

# Rank values
plan = TransformPlan().math_rank(
    column="score",
    new_column="rank",
    method="dense",
    descending=True,
    group_by="category"
)
```
