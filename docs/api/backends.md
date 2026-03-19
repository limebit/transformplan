# Backends

TransformPlan uses a pluggable backend system. Each backend implements the `Backend` ABC, providing all 87 operations plus meta methods for hashing, schema inspection, and type classification.

## Overview

The backend determines how data is stored and transformed:

- **PolarsBackend** (default): Operates on Polars DataFrames using native Polars expressions
- **DuckDBBackend** (optional): Operates on DuckDB relations using SQL generation

All operations, validation, dry-run, and serialization work identically regardless of backend. Pipelines serialized with one backend can be loaded and executed with another.

```python
from transformplan import TransformPlan

# Default — uses PolarsBackend
plan = TransformPlan()

# Explicit Polars backend
from transformplan.backends.polars import PolarsBackend
plan = TransformPlan(backend=PolarsBackend())

# DuckDB backend
import duckdb
from transformplan.backends.duckdb import DuckDBBackend
con = duckdb.connect()
plan = TransformPlan(backend=DuckDBBackend(con))
```

## Backend ABC

The abstract base class that all backends must implement.

::: transformplan.backends.base.Backend
    options:
      show_root_heading: true
      members:
        - compute_hash
        - get_shape
        - get_schema
        - get_columns
        - is_numeric_type
        - is_string_type
        - is_datetime_type
        - is_boolean_type
        - float_type
        - int_type
        - string_type
        - bool_type

## PolarsBackend

The default backend, using Polars DataFrames.

::: transformplan.backends.polars.PolarsBackend
    options:
      show_root_heading: true
      members:
        - compute_hash
        - get_shape
        - get_schema
        - get_columns

```python
import polars as pl
from transformplan import TransformPlan

df = pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
plan = TransformPlan().rows_filter(Col("age") >= 18)
result, protocol = plan.process(df)
```

## DuckDBBackend

Optional backend using DuckDB relations and SQL generation. Requires `duckdb` to be installed.

!!! note "Optional Dependency"
    Install DuckDB separately: `pip install duckdb` or `uv add duckdb`

::: transformplan.backends.duckdb.DuckDBBackend
    options:
      show_root_heading: true
      members:
        - compute_hash
        - get_shape
        - get_schema
        - get_columns

```python
import duckdb
from transformplan import TransformPlan, Col
from transformplan.backends.duckdb import DuckDBBackend

con = duckdb.connect()
rel = con.sql("SELECT * FROM 'data.parquet'")

plan = (
    TransformPlan(backend=DuckDBBackend(con))
    .col_rename(column="ID", new_name="id")
    .rows_filter(Col("age") >= 18)
    .math_standardize(column="score", new_column="z_score")
)

result, protocol = plan.process(rel)
```

## Cross-Backend Serialization

Pipelines are backend-agnostic when serialized. You can build a pipeline with one backend and execute it with another:

```python
import polars as pl
import duckdb
from transformplan import TransformPlan, Col
from transformplan.backends.duckdb import DuckDBBackend

# Build and serialize with Polars (default)
plan = (
    TransformPlan()
    .col_rename(column="ID", new_name="id")
    .rows_filter(Col("age") >= 18)
)
plan.to_json("pipeline.json")

# Load and execute with DuckDB
con = duckdb.connect()
rel = con.sql("SELECT * FROM 'data.parquet'")
plan_duckdb = TransformPlan.from_json("pipeline.json", backend=DuckDBBackend(con))
result, protocol = plan_duckdb.process(rel)
```

## Type System

Each backend classifies its native types into categories used by the validation system:

| Method | PolarsBackend | DuckDBBackend |
|--------|--------------|---------------|
| `is_numeric_type()` | Polars Int/Float/Decimal dtypes | `INTEGER`, `BIGINT`, `DOUBLE`, `FLOAT`, etc. |
| `is_string_type()` | `pl.Utf8`, `pl.String` | `VARCHAR`, `TEXT`, etc. |
| `is_datetime_type()` | `pl.Date`, `pl.Datetime`, `pl.Time` | `DATE`, `TIMESTAMP`, `TIME`, etc. |
| `is_boolean_type()` | `pl.Boolean` | `BOOLEAN` |

Type factory methods (`float_type()`, `int_type()`, `string_type()`, `bool_type()`) return the appropriate native type for each backend, used by operations that create new columns.
