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
import polars as pl
from transformplan import TransformPlan, Col

# Create sample data
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "salary": [50000, 60000, 70000]
})

# Build a transformation plan
plan = (
    TransformPlan()
    .col_rename(column="name", new_name="employee_name")
    .math_multiply(column="salary", value=1.1, new_column="new_salary")
    .math_round(column="new_salary", decimals=0)
    .rows_filter(Col("age") >= 30)
)

# Validate the plan
print(plan.validate(df))

# Execute and get audit trail
df_result, protocol = plan.process(df)
protocol.print()
```

## Why TransformPlan?

### Reproducibility

Every transformation is tracked with deterministic hashes, ensuring you can verify that the same inputs produce the same outputs.

### Safety

Schema validation catches errors before execution. The dry-run feature lets you preview what a pipeline will do without modifying data.

### Auditability

Complete audit protocols capture every operation, timing, and data shape change - essential for compliance and debugging.

## Getting Started

- [Installation](getting-started/installation.md) - How to install TransformPlan
- [Quickstart](getting-started/quickstart.md) - Your first transformation pipeline

## API Reference

- [TransformPlan](api/plan.md) - Main class for building pipelines
- [Filters](api/filters.md) - Filter expressions for row operations
- [Protocol](api/protocol.md) - Audit trail generation
- [Validation](api/validation.md) - Schema validation utilities
