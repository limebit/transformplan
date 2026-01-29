<div align="center">
  <img src="site/assets/images/logo_wordmark_black.png" alt="TransformPlan">
</div>

<h1 align="center">TransformPlan: Auditable Data Transformation Pipelines</h1>

<div align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python 3.10+">
  <img src="./coverage.svg" alt="Coverage">
</div>

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

| Category | Description | Examples |
|----------|-------------|----------|
| **col_** | Column operations | `col_rename`, `col_drop`, `col_cast`, `col_add`, `col_select` |
| **math_** | Arithmetic operations | `math_add`, `math_multiply`, `math_clamp`, `math_round`, `math_abs` |
| **rows_** | Row filtering & reshaping | `rows_filter`, `rows_drop_nulls`, `rows_sort`, `rows_unique`, `rows_pivot` |
| **str_** | String operations | `str_lower`, `str_upper`, `str_strip`, `str_replace`, `str_split` |
| **dt_** | Datetime operations | `dt_year`, `dt_month`, `dt_parse`, `dt_age_years`, `dt_diff_days` |
| **map_** | Value mapping | `map_values`, `map_discretize`, `map_case`, `map_from_column` |

## Installation

```bash
pip install transformplan
```

Or with uv:

```bash
uv add transformplan
```

## Development Setup

```bash
make install-dev   # Install with dev dependencies and pre-commit hooks
make test          # Run the test suite
make lint          # Run ruff linting and pyright type checking
make format        # Fix import sorting and format code
```

## License

MIT License - see [LICENSE](LICENSE) for details.
