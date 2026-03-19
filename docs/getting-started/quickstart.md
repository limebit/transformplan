# Quickstart

This guide walks you through creating your first transformation pipeline with TransformPlan.

## Creating a Pipeline

A pipeline is a sequence of operations that transform a DataFrame. Operations are registered using method chaining and executed together when you call `process()`.

```python
import polars as pl
from transformplan import TransformPlan, Col

# Sample data
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "department": ["Engineering", "Sales", "Engineering", "Sales"],
    "salary": [75000, 65000, 80000, 70000],
    "years": [3, 5, 7, 2]
})

# Build a transformation plan
plan = (
    TransformPlan()
    .col_rename(column="name", new_name="employee")
    .math_multiply(column="salary", value=1.05)  # 5% raise
    .math_round(column="salary", decimals=0)
    .rows_filter(Col("years") >= 3)
)
```

## Validating Before Execution

Before processing data, validate that all operations are compatible with the DataFrame schema:

```python
result = plan.validate(df)
print(result)  # ValidationResult(valid=True)

if result.is_valid:
    print("Pipeline is valid!")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

## Dry Run Preview

See what a pipeline will do without actually executing it:

```python
preview = plan.dry_run(df)
preview.print()
```

This shows each step with columns added, removed, or modified.

## Processing Data

Execute the pipeline and get the transformed data along with an audit protocol:

```python
df_result, protocol = plan.process(df)

print(df_result)
# shape: (3, 4)
# +----------+-------------+--------+-------+
# | employee | department  | salary | years |
# +----------+-------------+--------+-------+
# | Alice    | Engineering | 78750  | 3     |
# | Bob      | Sales       | 68250  | 5     |
# | Charlie  | Engineering | 84000  | 7     |
# +----------+-------------+--------+-------+
```

## Using the DuckDB Backend

TransformPlan supports DuckDB as an alternative backend. All 86 operations, validation, and dry-run work identically — only the data type changes from Polars DataFrames to DuckDB relations.

```python
import duckdb
from transformplan import TransformPlan, Col
from transformplan.backends.duckdb import DuckDBBackend

con = duckdb.connect()
rel = con.sql("""
    SELECT 'Alice' AS name, 'Engineering' AS department, 75000 AS salary, 3 AS years
    UNION ALL SELECT 'Bob', 'Sales', 65000, 5
    UNION ALL SELECT 'Charlie', 'Engineering', 80000, 7
    UNION ALL SELECT 'Diana', 'Sales', 70000, 2
""")

plan = (
    TransformPlan(backend=DuckDBBackend(con))
    .col_rename(column="name", new_name="employee")
    .math_multiply(column="salary", value=1.05)
    .math_round(column="salary", decimals=0)
    .rows_filter(Col("years") >= 3)
)

# Validate and execute — same API as Polars
result = plan.validate(rel)
if result.is_valid:
    df_result, protocol = plan.process(rel)
```

## Viewing the Audit Protocol

The protocol captures complete transformation history:

```python
protocol.print()
```

Output shows:

- Input/output hashes for reproducibility verification
- Each operation with parameters
- Row and column changes at each step
- Execution time per operation

## Filtering Rows

Use the `Col` class to build filter expressions:

```python
from transformplan import Col

# Simple comparison
plan = TransformPlan().rows_filter(Col("age") >= 18)

# Multiple conditions
plan = TransformPlan().rows_filter(
    (Col("status") == "active") & (Col("score") >= 50)
)

# String operations
plan = TransformPlan().rows_filter(
    Col("email").str_contains("@company.com")
)
```

## Saving and Loading Pipelines

Pipelines can be serialized to JSON for storage or sharing:

```python
# Save to JSON
plan.to_json("pipeline.json")

# Load from JSON
loaded_plan = TransformPlan.from_json("pipeline.json")

# Or work with strings
json_str = plan.to_json()
plan_from_str = TransformPlan.from_json(json_str)
```

## Processing Large Files

For Parquet files that exceed available RAM, use `process_chunked()`:

```python
# Process a large file in chunks
result, protocol = plan.process_chunked(
    source="large_dataset.parquet",
    chunk_size=100_000,  # Rows per chunk
)

protocol.print()  # Shows per-chunk statistics
```

When using operations that need related rows together (like `rows_unique`), specify a partition key:

```python
plan = (
    TransformPlan()
    .col_rename(column="PatientID", new_name="patient_id")
    .rows_unique(columns=["patient_id"])  # Needs all patient rows together
)

result, protocol = plan.process_chunked(
    source="patients.parquet",
    partition_key="patient_id",  # Keep patient rows in same chunk
    chunk_size=50_000,
)
```

Validate compatibility before processing:

```python
validation = plan.validate_chunked(
    schema=df.schema,
    partition_key="patient_id"
)

if not validation.is_valid:
    print(validation.errors)
```

!!! note "Operation Restrictions"
    Some operations cannot be used with chunked processing:
    `rows_sort`, `rows_pivot`, `rows_sample`, `rows_head`, `rows_tail`.
    See [Chunked Processing](../api/chunking.md) for details.

!!! note "Polars Only"
    Chunked processing is designed for Polars DataFrames and Parquet files.
    DuckDB handles large datasets natively through its out-of-core execution engine — no chunking needed.

## Next Steps

- Explore the [API Reference](../api/index.md) for all available operations
- Learn about [Filters](../api/filters.md) for complex row filtering
- Understand [Protocols](../api/protocol.md) for audit trails
- Process large files with [Chunked Processing](../api/chunking.md)
