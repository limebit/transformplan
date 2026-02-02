<div align="center">
  <img src="https://raw.githubusercontent.com/limebit/transformplan-static/refs/heads/main/logos/logo_wordmark_white.png" alt="TransformPlan Logo" width="450">
</div>

# TransformPlan

A Python library for safe, reproducible data transformations with built-in auditing and validation.

TransformPlan tracks transformation history, validates operations against DataFrame schemas, and generates audit trails for data processing workflows.

## Features

- **Declarative transformations**: Build transformation pipelines using method chaining
- **Schema validation**: Validate operations before execution with dry-run capability
- **Audit trails**: Generate complete audit protocols with deterministic DataFrame hashing
- **Multi-backend support**: Works with both Polars (primary) and Pandas DataFrames
- **Serializable pipelines**: Save and load transformation plans as JSON

## Quick Example

```python
from transformplan import TransformPlan, Col

# Build readable pipelines with 70+ chainable operations
plan = (
    TransformPlan()
    # Standardize column names
    .col_rename(column="PatientID", new_name="patient_id")
    .col_rename(column="DOB", new_name="date_of_birth")
    .str_strip(column="patient_id")

    # Calculate derived values
    .dt_age_years(column="date_of_birth", new_column="age")
    .math_clamp(column="age", min_value=0, max_value=120)

    # Categorize patients age
    .map_discretize(column="age", bins=[18, 40, 65], labels=["young", "adult", "senior"], new_column="age_group")

    # Filter and clean
    .rows_filter(Col("age") >= 18)
    .rows_drop_nulls(columns=["patient_id", "age"])
    .col_drop(column="date_of_birth")
)

# Execute with schema validation — catch errors before they hit production
df_result, protocol = plan.process(df, validate=True)

# Serialize pipelines to JSON — version control your transformations
plan.to_json("patient_transform.json")

# Reload and reapply — reproducible results across environments
plan = TransformPlan.from_json("patient_transform.json")
df_result, protocol = plan.process(new_data)
```

### Full Audit Trail — Every Step Tracked and Hashed

```python
protocol.print(show_params=False)
```

```
======================================================================
TRANSFORM PROTOCOL
======================================================================
Input:  1000 rows × 5 cols  [a4f8b2c1]
Output: 847 rows × 5 cols   [e7d3f9a2]
Total time: 0.0247s
----------------------------------------------------------------------

#    Operation            Rows         Cols         Time       Hash
----------------------------------------------------------------------
0    input                1000         5            -          a4f8b2c1
1    col_rename           1000         5            0.0012s    b2e4a7f3
2    col_rename           1000         5            0.0008s    c9d1e5b8
3    str_strip            1000         5            0.0013s    c9d1e5b8        ○
4    dt_age_years         1000         6 (+1)       0.0041s    d4f2c8a1
5    math_clamp           1000         6            0.0015s    e1b7d3f9
6    map_discretize       1000         7 (+1)       0.0028s    f8a4c2e6
7    rows_filter          858 (-142)   7            0.0037s    a2e9f4b7
8    rows_drop_nulls      847 (-11)    7            0.0019s    b5c1d8e3
9    col_drop             847          6 (-1)       0.0006s    e7d3f9a2
======================================================================
○ = no effect (steps 3 did not change data)
```

## Available Operations

| Category   | Description               | Examples                                                                   |
| ---------- | ------------------------- | -------------------------------------------------------------------------- |
| **col\_**  | Column operations         | `col_rename`, `col_drop`, `col_cast`, `col_add`, `col_select`              |
| **math\_** | Arithmetic operations     | `math_add`, `math_multiply`, `math_clamp`, `math_round`, `math_abs`        |
| **rows\_** | Row filtering & reshaping | `rows_filter`, `rows_drop_nulls`, `rows_sort`, `rows_unique`, `rows_pivot` |
| **str\_**  | String operations         | `str_lower`, `str_upper`, `str_strip`, `str_replace`, `str_split`          |
| **dt\_**   | Datetime operations       | `dt_year`, `dt_month`, `dt_parse`, `dt_age_years`, `dt_diff_days`          |
| **map\_**  | Value mapping & encoding  | `map_values`, `map_discretize`, `map_onehot`, `map_ordinal`                |

## Getting Started

- [Installation](getting-started/installation.md) - How to install TransformPlan
- [Quickstart](getting-started/quickstart.md) - Your first transformation pipeline

## API Reference

- [TransformPlan](api/plan.md) - Main class for building pipelines
- [Filters](api/filters.md) - Filter expressions for row operations
- [Protocol](api/protocol.md) - Audit trail generation
- [Chunked Processing](api/chunking.md) - Process large files that exceed RAM
- [Validation](api/validation.md) - Schema validation utilities
