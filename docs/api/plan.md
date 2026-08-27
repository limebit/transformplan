# TransformPlan

The main class for building and executing transformation pipelines.

## Overview

`TransformPlan` uses a deferred execution model: operations are registered via method chaining, then executed together when you call `process()`, `validate()`, or `dry_run()`. The plan itself is backend-agnostic — the backend is chosen at execution time (defaults to `PolarsBackend`).

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

The backend is passed at execution time, not at construction:

```python
from transformplan.backends.duckdb import DuckDBBackend

plan = TransformPlan().col_drop("temp").math_add("age", 1)

# Default (Polars)
result, protocol = plan.process(polars_df)

# DuckDB
con = duckdb.connect()
result, protocol = plan.process(duckdb_rel, backend=DuckDBBackend(con))
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
        - pipe
        - extend
        - section
        - because
        - to_dict
        - from_dict
        - to_json
        - from_json
        - to_python

## Structuring Long Plans

Plans with dozens of steps become hard to read as a flat chain. Four methods
give them structure without changing what they do.

### pipe

Apply a function that takes and returns a plan, so repeated blocks can live in
their own function without breaking the chain.

```python
def decimal_hour(plan, source, target):
    return (
        plan.dt_format(source, "%H", target)
        .col_cast(target, "Float64")
    )

plan = (
    TransformPlan()
    .dt_diff_days("admitted", "discharged", "stay_days")
    .pipe(decimal_hour, "admitted_at", "admit_hour")
    .pipe(decimal_hour, "discharged_at", "discharge_hour")
)
```

`pipe` registers no step of its own — the function registers ordinary steps, so
the protocol is unchanged.

### extend

Append another plan's steps, so a reusable block can be defined once as its own
plan. Steps are deep-copied, so a block can be reused across plans without them
sharing state.

```python
CLEAN_NAMES = TransformPlan().str_strip("name").str_upper("name")

plan = TransformPlan().col_drop("temp").extend(CLEAN_NAMES)
combined = plan_a + plan_b  # same thing, as a new plan
```

Unlike `pipe`, the block's steps become part of the plan and are serialized with it.

### section

Group the following steps under a label. The protocol then reports a balance per
section instead of one flat list.

```python
plan = (
    TransformPlan()
    .section("Row filters")
    .rows_drop_nulls("shipped_at")
    .rows_drop_nulls("order_id")
    .section("Split product code")
    .col_duplicate("code", "code_group")
    .str_slice("code_group", 0, 3)
)
```

```
SECTIONS

  Row filters                   1500 →       49 rows   (2 steps)
  Split product code              49 →       49 rows   (2 steps, +1 cols)
```

Pass `None` to end a section. Plans without sections render exactly as before.

### because

Attach a reason to the step just registered. The protocol records that rows were
removed; the reason records why — the question an audit actually asks.

```python
plan = (
    TransformPlan()
    .rows_drop(Col("provider") != "MDK02")
    .because("cases where the reviewer awards the insurer nothing")
    .rows_drop_nulls("drg_code")
    .because("PEPP cases carry no DRG")
)
```

When the preceding call covered several columns, the reason is attached to every
step it produced.

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
