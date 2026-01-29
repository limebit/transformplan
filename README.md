# TransformPlan: Data Transformation Pipeline

<table>
<tr>
<td width="30%" valign="top">
<img src="docs/assets/images/logo_blue.png" alt="TransformPlan Logo" width="100%">
</td>
<td width="70%" valign="top">

# Features

- **Declarative transformations**: Build transformation pipelines using method chaining
- **Schema validation**: Validate operations before execution with dry-run capability
- **Audit trails**: Generate complete audit protocols with deterministic DataFrame hashing
- **Multi-backend support**: Works with both Polars (primary) and Pandas DataFrames
- **Serializable pipelines**: Save and load transformation plans as JSON

</td>
</tr>
</table>

## Quick Example

```python
from transformplan import TransformPlan, Col

# Define a data transformation plan
plan = (
    TransformPlan()
    # Standardize column names
    .col_rename(column="PatientID", new_name="patient_id")
    .col_rename(column="DOB", new_name="date_of_birth")

    # Calculate derived values
    .dt_age_years(column="date_of_birth", new_column="age")
    .math_clamp(column="age", min_value=0, max_value=120)

    # Categorize patients
    .map_discretize(column="age", bins=[18, 40, 65], labels=["young", "adult", "senior"], new_column="age_group")

    # Filter and clean
    .rows_filter(Col("age") >= 18)
    .rows_drop_nulls(columns=["patient_id", "age"])
    .col_drop(column="date_of_birth")
)

# Run it on your data with automated pre-validation
df_result, protocol = plan.process(df,validate=True)

# Save pipeline to JSON
plan.to_json("patient_transform.json")

# Load and reuse
plan = TransformPlan.from_json("patient_transform.json")
df_result, protocol = plan.process(new_data)
```

### Generated Audit Protocol

```
======================================================================
TRANSFORM PROTOCOL
======================================================================
Input:  1000 rows × 5 cols  [a4f8b2c1]
Output: 847 rows × 5 cols   [e7d3f9a2]
Total time: 0.0234s
----------------------------------------------------------------------

#    Operation            Rows         Cols         Time       Hash
----------------------------------------------------------------------
0    input                1000         5            -          a4f8b2c1
1    col_rename           1000         5            0.0012s    b2e4a7f3
2    col_rename           1000         5            0.0008s    c9d1e5b8
3    dt_age_years         1000         6 (+1)       0.0041s    d4f2c8a1
4    math_clamp           1000         6            0.0015s    e1b7d3f9
5    map_discretize       1000         7 (+1)       0.0028s    f8a4c2e6
6    rows_filter          858 (-142)   7            0.0037s    a2e9f4b7
7    rows_drop_nulls      847 (-11)    7            0.0019s    b5c1d8e3
8    col_drop             847          6 (-1)       0.0006s    e7d3f9a2
======================================================================
```

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
