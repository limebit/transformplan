# Validation

Schema validation and dry-run preview for TransformPlan pipelines.

## Overview

TransformPlan validates operations against DataFrame schemas before execution. This catches errors like:

- Referencing non-existent columns
- Applying string operations to numeric columns
- Creating columns that already exist

```python
from transformplan import TransformPlan

plan = TransformPlan().col_drop("nonexistent")
result = plan.validate(df)

if not result.is_valid:
    for error in result.errors:
        print(error)
    # Step 1 (col_drop): Column 'nonexistent' does not exist
```

## ValidationResult

::: transformplan.validation.ValidationResult
    options:
      show_root_heading: true
      members:
        - is_valid
        - errors
        - add_error
        - raise_if_invalid

## ValidationError

::: transformplan.validation.ValidationError
    options:
      show_root_heading: true

## SchemaValidationError

::: transformplan.validation.SchemaValidationError
    options:
      show_root_heading: true

## DryRunResult

::: transformplan.validation.DryRunResult
    options:
      show_root_heading: true
      members:
        - is_valid
        - errors
        - steps
        - input_schema
        - output_schema
        - input_columns
        - output_columns
        - summary
        - print

## DryRunStep

::: transformplan.validation.DryRunStep
    options:
      show_root_heading: true

## Example: Validation

```python
from transformplan import TransformPlan, Col

df = pl.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [25, 30],
    "salary": [50000, 60000]
})

plan = (
    TransformPlan()
    .col_drop("age")
    .rows_filter(Col("age") > 18)  # Error: age was dropped!
)

result = plan.validate(df)
print(result)
# ValidationResult(valid=False, errors=1)

for error in result.errors:
    print(error)
# Step 2 (rows_filter): Column 'age' does not exist
```

## Example: Dry Run

```python
plan = (
    TransformPlan()
    .col_drop("temp")
    .col_add("bonus", value=1000)
    .math_multiply("salary", 1.1)
)

preview = plan.dry_run(df)
preview.print()
```

Output:

```
======================================================================
DRY RUN PREVIEW
======================================================================
Validation: PASSED
----------------------------------------------------------------------
Input: 3 columns
----------------------------------------------------------------------

#    Operation            Columns        Changes
----------------------------------------------------------------------
1    col_drop             2              -['temp']
     -> column='temp'
2    col_add              3              +['bonus']
     -> new_column='bonus', value=1000
3    math_multiply        3              ~['salary']
     -> column='salary', value=1.1
======================================================================
Output: 3 columns
```

## Type Checking

Validation includes type checking for operations that require specific types:

| Operation Type | Required Column Type |
|---------------|---------------------|
| `math_*` | Numeric (Int, Float) |
| `str_*` | String (Utf8) |
| `dt_*` | Datetime (Date, Datetime, Time) |
