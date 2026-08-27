<div align="center">
  <img src="https://raw.githubusercontent.com/limebit/transformplan-static/refs/heads/main/logos/logo_wordmark_black.png" alt="TransformPlan" width="600">
</div>

<h1 align="center">TransformPlan: Auditable Data Transformation Pipelines</h1>

<div align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue" alt="Python 3.10+">
  <img src="https://raw.githubusercontent.com/limebit/transformplan-static/6cb5e0d4d33699ee663f9d40ff9b6724279fd394/icons/coverage-badge.svg" alt="Coverage">
</div>

## Features

- **Declarative transformations**: Build transformation pipelines using method chaining
- **Schema validation**: Validate operations before execution with dry-run capability
- **Audit trails**: Generate complete audit protocols with deterministic DataFrame hashing
- **Multi-backend support**: Polars (default) and DuckDB backends with a pluggable Backend ABC
- **Serializable pipelines**: Save and load transformation plans as JSON
- **Structured plans**: Group long pipelines into named sections and compose them from reusable blocks
- **Documented decisions**: Record *why* a step exists, not just what it did

## Quick Example

```python
from transformplan import TransformPlan, Col

# Build readable pipelines with 89 chainable operations
plan = (
    TransformPlan()
    .section("Standardize")
    .col_rename(column="PatientID", new_name="patient_id")
    .col_rename(column="DOB", new_name="date_of_birth")
    .str_strip(column="patient_id")
    .col_drop(["tmp_a", "tmp_b"]).because("scratch columns from the export")

    .section("Derive")
    .dt_age_years(birth_column="date_of_birth", new_column="age")
    .col_cast("raw_weight", "Float64")
    .map_discretize(column="age", bins=[18, 40, 65], labels=["minor", "young", "adult", "senior"], new_column="age_group")

    .section("Filter")
    .rows_filter(Col("age") >= 18).because("cohort is adults only")
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
Input:  1000 rows x 5 cols  [a8bfc98263e8aa4c]
Output: 841 rows x 4 cols  [6f62d5e6d677cb81]
Total time: 0.0011s
----------------------------------------------------------------------

SECTIONS

  Standardize                  1000 →     1000 rows   (5 steps, -2 cols)
  Derive                       1000 →     1000 rows   (3 steps, +2 cols)
  Filter                       1000 →      841 rows   (3 steps, -1 cols)
----------------------------------------------------------------------

#    Operation            Rows         Cols         Time       Hash
----------------------------------------------------------------------
0    input                1000         5            -          a8bfc98263e8aa4c
[Standardize]
1    col_rename           1000         5            0.0000s    4a3d3ba5644cf847
2    col_rename           1000         5            0.0000s    e535f4ddd1e78927
3    str_strip            1000         5            0.0001s    14cf5b02000d13a4
4    col_drop             1000         4 (-1)       0.0001s    ccaddd0bc65974d7
     ↳ scratch columns from the export
5    col_drop             1000         3 (-1)       0.0000s    08a006cf6351723c
     ↳ scratch columns from the export
[Derive]
6    dt_age_years         1000         4 (+1)       0.0001s    2e3c2e80b77f7e10
7    col_cast             1000         4            0.0001s    e7e31d60ee332aa6
8    map_discretize       1000         5 (+1)       0.0002s    f3c8c0998d8ee820
[Filter]
9    rows_filter          851 (-149)   5            0.0002s    97f54c043b457d0c
     ↳ cohort is adults only
10   rows_drop_nulls      841 (-10)    5            0.0002s    7bb8de08021dfe20
11   col_drop             841          4 (-1)       0.0001s    6f62d5e6d677cb81
======================================================================
```

Sections give a plan with a hundred steps a readable balance instead of one flat
list, and `because()` records the domain knowledge that otherwise lives only in a
code comment — the question an audit actually asks.

### Composing Long Pipelines

Repeated blocks can live in their own function or their own plan, without breaking
the chain:

```python
# A function that takes and returns a plan
def decimal_hour(plan, source, target):
    return (
        plan.dt_format(source, "%H", target)
        .col_cast(target, "Float64")
    )

# A reusable block, defined as a plan of its own
CLEAN_IDS = TransformPlan().str_strip("patient_id").str_upper("patient_id")

plan = (
    TransformPlan()
    .pipe(decimal_hour, "admitted_at", "admit_hour")
    .pipe(decimal_hour, "discharged_at", "discharge_hour")
    .extend(CLEAN_IDS)          # or: plan_a + plan_b
)
```

`pipe()` registers no step of its own, so the protocol is unchanged. `extend()`
deep-copies the block, so the same block can be reused across plans without them
sharing state — and unlike `pipe()`, its steps are serialized with the plan.

Operations that work in place also take a sequence of columns, so repetition
collapses into one call:

```python
plan = (
    TransformPlan()
    .col_drop(["fiscal_year", "import_batch", "row_checksum"])
    .str_slice(["primary_code", "secondary_code"], 0, 3)
    .col_cast(["weight", "height"], "Float64")
)
```

Each column still becomes its own protocol step.

### DuckDB Backend

Run the same pipelines on DuckDB for SQL-based execution and native large-file handling:

```python
import duckdb
from transformplan import TransformPlan, Col
from transformplan.backends.duckdb import DuckDBBackend

con = duckdb.connect()
rel = con.sql("SELECT * FROM 'patients.parquet'")

# Same plan — backend chosen at execution time
plan = (
    TransformPlan()
    .col_rename(column="PatientID", new_name="patient_id")
    .rows_filter(Col("age") >= 18)
    .math_round(column="score", decimals=2)
)

result, protocol = plan.process(rel, backend=DuckDBBackend(con))
```

## Available Operations

| Category   | Description               | Examples                                                                     |
| ---------- | ------------------------- | ---------------------------------------------------------------------------- |
| **col\_**  | Column operations         | `col_rename`, `col_drop`, `col_cast`, `col_add`, `col_select`                |
| **math\_** | Arithmetic & scaling      | `math_add`, `math_multiply`, `math_standardize`, `math_minmax`, `math_clamp` |
| **rows\_** | Row filtering & reshaping | `rows_filter`, `rows_drop_nulls`, `rows_sort`, `rows_unique`, `rows_pivot`   |
| **str\_**  | String operations         | `str_lower`, `str_upper`, `str_strip`, `str_replace`, `str_split`            |
| **dt\_**   | Datetime operations       | `dt_year`, `dt_month`, `dt_parse`, `dt_age_years`, `dt_diff_days`            |
| **map\_**  | Value mapping & encoding  | `map_values`, `map_discretize`, `map_onehot`, `map_ordinal`                  |

24 in-place operations accept either a single column or a sequence of columns.
Operations that name an output column (`new_column`) take one column at a time.

See the [roadmap](docs/roadmap.md) for planned additions.

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
make install-dev   # Install with dev dependencies
make test          # Run the test suite
make lint          # Run ruff linting and pyright type checking
make format        # Fix import sorting and format code
```

## License

MIT License - see [LICENSE](LICENSE) for details.
