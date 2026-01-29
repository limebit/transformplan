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

df_result, protocol = plan.process(df)
```

### Generated Audit Protocol

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRANSFORMATION PROTOCOL                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input:  4a7b2c1d  (1000 rows × 8 cols)                                      │
│ Output: 8f3e9a2b  (847 rows × 8 cols)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Operations:                                                                 │
│   1. col_rename: PatientID → patient_id                                     │
│   2. col_rename: DOB → date_of_birth                                        │
│   3. dt_age_years: date_of_birth → age                                      │
│   4. math_clamp: age [0, 120]                                               │
│   5. map_discretize: age → age_group                                        │
│   6. rows_filter: Col("age") >= 18  (-142 rows)                             │
│   7. rows_drop_nulls: ["patient_id", "age"]  (-11 rows)                     │
│   8. col_drop: date_of_birth                                                │
└─────────────────────────────────────────────────────────────────────────────┘
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
