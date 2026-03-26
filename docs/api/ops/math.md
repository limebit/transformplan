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
        - math_diff_from_agg
        - math_diff_lag
        - math_standardize
        - math_minmax
        - math_robust_scale
        - math_log
        - math_sqrt
        - math_power
        - math_winsorize

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

### Window Difference

```python
# Difference from group minimum (e.g., time since first measurement)
plan = TransformPlan().math_diff_from_agg(
    column="timestamp",
    agg="min",
    new_column="time_since_first",
    group_by="patient_id",
)

# Difference from group mean (deviation from average)
plan = TransformPlan().math_diff_from_agg(
    column="score",
    agg="mean",
    new_column="deviation",
    group_by="category",
)

# Global aggregate (no group_by)
plan = TransformPlan().math_diff_from_agg(
    column="value",
    agg="max",
    new_column="diff_from_max",
)

# Row-to-row difference (lag)
plan = TransformPlan().math_diff_lag(
    column="timestamp",
    order_by="timestamp",
    new_column="time_between",
    group_by="patient_id",
)

# Numeric change ordered by date
plan = TransformPlan().math_diff_lag(
    column="price",
    order_by="date",
    new_column="daily_change",
)

# Lag of 2 rows
plan = TransformPlan().math_diff_lag(
    column="value",
    order_by="seq",
    new_column="diff_2",
    lag=2,
)
```

### Scaling Operations

```python
# Z-score standardization (explicit params for reproducibility)
plan = TransformPlan().math_standardize("income", mean=50000, std=25000)

# Derive from data
plan = TransformPlan().math_standardize("income")

# Min-max normalization to [0, 1]
plan = TransformPlan().math_minmax("age", min_val=0, max_val=100)

# Custom range
plan = TransformPlan().math_minmax("score", min_val=0, max_val=100, feature_range=(0, 10))

# Robust scaling (resistant to outliers)
plan = TransformPlan().math_robust_scale("salary", median=60000, iqr=30000)
```

### Transform Operations

```python
# Natural log
plan = TransformPlan().math_log("price")

# Log base 10
plan = TransformPlan().math_log("price", base=10)

# Log with offset for zeros
plan = TransformPlan().math_log("count", offset=1)  # log(x + 1)

# Square root
plan = TransformPlan().math_sqrt("variance")

# Power transform
plan = TransformPlan().math_power("value", exponent=2)  # square
plan = TransformPlan().math_power("value", exponent=0.5)  # sqrt
```

### Outlier Handling

```python
# Winsorize by percentiles
plan = TransformPlan().math_winsorize("salary", lower=0.05, upper=0.95)

# Winsorize by explicit values
plan = TransformPlan().math_winsorize("salary", lower_value=20000, upper_value=200000)
```
