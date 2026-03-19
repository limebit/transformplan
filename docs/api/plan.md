# TransformPlan

The main class for building and executing transformation pipelines.

## Overview

`TransformPlan` uses a deferred execution model: operations are registered via method chaining, then executed together when you call `process()`, `validate()`, or `dry_run()`. An optional `backend` parameter selects the execution engine (defaults to `PolarsBackend`).

```python
from transformplan import TransformPlan, Col

plan = (
    TransformPlan()
    .col_drop("temp_column")
    .math_multiply("price", 1.1)
    .rows_filter(Col("active") == True)
)

# Execute
df_result, protocol = plan.process(df)
```

## Backend Selection

```python
# Default (Polars)
plan = TransformPlan()

# DuckDB
import duckdb
from transformplan.backends.duckdb import DuckDBBackend
con = duckdb.connect()
plan = TransformPlan(backend=DuckDBBackend(con))
```

See [Backends](backends.md) for details on each backend.

## Class Reference

::: transformplan.TransformPlan
    options:
      show_root_heading: true
      members:
        - process
        - process_chunked
        - validate
        - validate_chunked
        - dry_run
        - to_dict
        - from_dict
        - to_json
        - from_json
        - to_python

## Execution Methods

### process

Execute all registered operations and return transformed data with an audit protocol.

```python
df_result, protocol = plan.process(df)
```

### validate

Validate operations against the DataFrame schema without executing.

```python
result = plan.validate(df)
if not result.is_valid:
    for error in result.errors:
        print(error)
```

### dry_run

Preview what the pipeline will do without executing it.

```python
preview = plan.dry_run(df)
preview.print()
```

## Chunked Processing

For large Parquet files that exceed available RAM, use chunked processing methods.

### process_chunked

Process a large Parquet file in chunks, optionally keeping related rows together.

```python
result, protocol = plan.process_chunked(
    source="large_file.parquet",
    partition_key="patient_id",  # Keep patient rows together
    chunk_size=100_000,
)
protocol.print()
```

See [Chunked Processing](chunking.md) for details on operation compatibility.

### validate_chunked

Validate that a pipeline is compatible with chunked processing before executing.

```python
validation = plan.validate_chunked(
    schema={"id": pl.Int64, "name": pl.Utf8},
    partition_key="id"
)
if not validation.is_valid:
    print(validation.errors)
```

## Serialization

Pipelines can be saved and loaded as JSON:

```python
# Save
plan.to_json("pipeline.json")

# Load
loaded = TransformPlan.from_json("pipeline.json")
```

Plans are backend-agnostic when serialized — a pipeline saved from a Polars workflow can be loaded and executed with DuckDB, and vice versa.

Or generate executable Python code:

```python
print(plan.to_python())
```
