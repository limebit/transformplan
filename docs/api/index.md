# API Reference

This section provides detailed API documentation for all TransformPlan classes and functions.

## Core Classes

| Class | Description |
|-------|-------------|
| [`TransformPlan`](plan.md) | Main class for building transformation pipelines |
| [`Protocol`](protocol.md) | Audit trail capturing transformation history |
| [`Col`](filters.md#transformplan.filters.Col) | Column reference for building filter expressions |
| [`Filter`](filters.md#transformplan.filters.Filter) | Base class for serializable filter expressions |

## Validation Classes

| Class | Description |
|-------|-------------|
| [`ValidationResult`](validation.md#transformplan.validation.ValidationResult) | Result of schema validation |
| [`DryRunResult`](validation.md#transformplan.validation.DryRunResult) | Preview of pipeline execution |
| [`SchemaValidationError`](validation.md#transformplan.validation.SchemaValidationError) | Exception raised on validation failure |

## Operation Categories

TransformPlan provides operations organized by category:

| Category | Description | Examples |
|----------|-------------|----------|
| [Column Operations](ops/column.md) | Add, drop, rename, cast columns | `col_drop`, `col_rename`, `col_cast` |
| [Math Operations](ops/math.md) | Arithmetic on numeric columns | `math_add`, `math_multiply`, `math_round` |
| [Row Operations](ops/rows.md) | Filter, sort, deduplicate rows | `rows_filter`, `rows_sort`, `rows_unique` |
| [String Operations](ops/string.md) | Text manipulation | `str_replace`, `str_lower`, `str_split` |
| [Datetime Operations](ops/datetime.md) | Date and time extraction | `dt_year`, `dt_month`, `dt_parse` |
| [Map Operations](ops/map.md) | Value mapping and discretization | `map_values`, `map_discretize` |

## Utility Functions

| Function | Description |
|----------|-------------|
| [`frame_hash`](protocol.md#transformplan.protocol.frame_hash) | Compute deterministic hash of a DataFrame |
