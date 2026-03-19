# Chunked Processing

Process large Parquet files that exceed available RAM by reading and transforming data in chunks.

!!! note "Polars Only"
    Chunked processing is designed for Polars DataFrames and Parquet files.
    DuckDB handles large datasets natively through its out-of-core execution engine — no chunking needed.

## Overview

When working with files larger than available memory, use `process_chunked()` instead of `process()`. This method:

- Reads Parquet files in configurable chunks using lazy evaluation
- Optionally keeps related rows together using a partition key
- Validates that operations are compatible with chunked processing
- Returns a `ChunkedProtocol` with per-chunk statistics

```python
from transformplan import TransformPlan, Col

plan = (
    TransformPlan()
    .col_rename(column="PatientID", new_name="patient_id")
    .rows_filter(Col("age") >= 18)
    .rows_unique(columns=["patient_id", "visit_date"])
)

# Process a large file in chunks
result, protocol = plan.process_chunked(
    source="patients_10gb.parquet",
    partition_key="patient_id",  # Keep patient rows together
    chunk_size=100_000,
)

protocol.print()
```

## Operation Compatibility

Not all operations can be used with chunked processing. Operations are classified into three categories:

### Chunkable Operations

These operations process each row independently and work with any chunk:

| Category | Operations |
|----------|------------|
| Column | `col_drop`, `col_rename`, `col_cast`, `col_reorder`, `col_select`, `col_duplicate`, `col_fill_null`, `col_drop_null`, `col_drop_zero`, `col_add`, `col_add_uuid`, `col_hash`, `col_coalesce` |
| Math | `math_add`, `math_subtract`, `math_multiply`, `math_divide`, `math_clamp`, `math_abs`, `math_round`, `math_set_min`, `math_set_max`, `math_add_columns`, `math_subtract_columns`, `math_multiply_columns`, `math_divide_columns`, `math_percent_of` |
| String | `str_replace`, `str_slice`, `str_truncate`, `str_lower`, `str_upper`, `str_strip`, `str_pad`, `str_split`, `str_concat`, `str_extract` |
| Datetime | `dt_year`, `dt_month`, `dt_day`, `dt_week`, `dt_quarter`, `dt_year_month`, `dt_quarter_year`, `dt_calendar_week`, `dt_parse`, `dt_format`, `dt_diff_days`, `dt_age_years`, `dt_is_between`, `dt_truncate` |
| Map | `map_values`, `map_discretize`, `map_bool_to_int`, `map_null_to_value`, `map_value_to_null`, `map_case`, `map_from_column` |
| Rows | `rows_filter`, `rows_drop`, `rows_flag`, `rows_explode`, `rows_drop_nulls`, `rows_melt` |

### Group-Dependent Operations

These operations need all rows for a group together. They work with chunked processing **only when** `partition_key` includes their grouping columns:

| Operation | Group Parameter | Requirement |
|-----------|-----------------|-------------|
| `rows_unique` | `columns` | `partition_key` must contain `columns` |
| `rows_deduplicate` | `columns` | `partition_key` must contain `columns` |
| `math_cumsum` | `group_by` | `partition_key` must contain `group_by` |
| `math_rank` | `group_by` | `partition_key` must contain `group_by` |

**Example**: To use `rows_unique(columns=["patient_id"])`, you must set `partition_key="patient_id"` (or a list containing it).

### Global Operations (Blocked)

These operations require the full dataset and **cannot** be used with chunked processing:

- `rows_sort` - Requires global ordering
- `rows_pivot` - Needs all values to determine output columns
- `rows_sample` - Random sampling requires full dataset
- `rows_head` - Requires global ordering
- `rows_tail` - Requires global ordering

Attempting to use these operations will raise a `ChunkingError`.

## Validation

Before processing, validate that your pipeline is compatible with chunked processing:

```python
# Validate without processing
validation = plan.validate_chunked(
    schema={"patient_id": pl.Utf8, "age": pl.Int64, "visit_date": pl.Date},
    partition_key="patient_id"
)

print(validation)
# Pipeline is compatible with chunked processing.

# Or validate with a sample DataFrame
validation = plan.validate_chunked(data=sample_df, partition_key="patient_id")

if not validation.is_valid:
    for error in validation.errors:
        print(f"Error: {error}")
```

## ChunkedProtocol Class

::: transformplan.chunking.ChunkedProtocol
    options:
      show_root_heading: true
      members:
        - set_source
        - set_operations
        - set_metadata
        - add_chunk
        - chunks
        - total_input_rows
        - total_output_rows
        - total_elapsed_seconds
        - num_chunks
        - metadata
        - output_hash
        - to_dict
        - from_dict
        - to_json
        - from_json
        - summary
        - print

## ChunkValidationResult Class

::: transformplan.chunking.ChunkValidationResult
    options:
      show_root_heading: true
      members:
        - is_valid
        - errors
        - warnings
        - global_operations
        - group_dependent_ops

## ChunkingError Exception

::: transformplan.chunking.ChunkingError
    options:
      show_root_heading: true

## Example: Processing Patient Records

```python
import polars as pl
from transformplan import TransformPlan, Col, ChunkingError

# Build pipeline with group-dependent operation
plan = (
    TransformPlan()
    .col_rename(column="PatientID", new_name="patient_id")
    .dt_age_years(birth_column="date_of_birth", new_column="age")
    .rows_filter(Col("age") >= 18)
    .rows_unique(columns=["patient_id", "visit_date"])  # Needs partition key
    .col_drop("date_of_birth")
)

# Validate first
validation = plan.validate_chunked(
    schema={
        "PatientID": pl.Utf8,
        "date_of_birth": pl.Date,
        "visit_date": pl.Date,
        "diagnosis": pl.Utf8,
    },
    partition_key="PatientID"
)

if validation.is_valid:
    # Process the large file
    result, protocol = plan.process_chunked(
        source="patients_archive.parquet",
        partition_key="PatientID",
        chunk_size=50_000,
    )

    # View processing summary
    protocol.print()

    # Save audit trail
    protocol.to_json("chunked_audit.json")
else:
    print("Pipeline incompatible with chunking:")
    for error in validation.errors:
        print(f"  - {error}")
```

## Example Output

```
======================================================================
CHUNKED PROCESSING PROTOCOL
======================================================================
Source: patients_archive.parquet
Partition key: ['PatientID']
Target chunk size: 50,000
----------------------------------------------------------------------
Chunks processed: 24
Total input rows: 1,187,432
Total output rows: 892,156
Row change: -295,276
Total time: 12.4523s
Avg time per chunk: 0.5188s
Output hash: 7a3b2c1d4e5f6789
----------------------------------------------------------------------

#      Input        Output       Change     Time       Hash
----------------------------------------------------------------------
0      49,832       37,291       -12,541    0.4821s    a1b2c3d4e5f67890
1      50,127       38,456       -11,671    0.5123s    b2c3d4e5f6789012
2      49,956       37,892       -12,064    0.4956s    c3d4e5f678901234
...
======================================================================
```
