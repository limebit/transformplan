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

## Complete Method Reference

All TransformPlan operations at a glance. Click method names for detailed documentation.

### Column Operations

| Method | Description |
|--------|-------------|
| [`col_drop`](ops/column.md) | Drop a column from the DataFrame |
| [`col_rename`](ops/column.md) | Rename a column |
| [`col_cast`](ops/column.md) | Cast a column to a different dtype |
| [`col_reorder`](ops/column.md) | Reorder columns (drops unlisted) |
| [`col_select`](ops/column.md) | Keep only the specified columns |
| [`col_duplicate`](ops/column.md) | Duplicate a column under a new name |
| [`col_fill_null`](ops/column.md) | Fill null values in a column |
| [`col_drop_null`](ops/column.md) | Drop rows with null values in specified columns |
| [`col_drop_zero`](ops/column.md) | Drop rows where the specified column is zero |
| [`col_add`](ops/column.md) | Add a new column with a constant value or expression |
| [`col_add_uuid`](ops/column.md) | Add a column with unique random identifiers |
| [`col_hash`](ops/column.md) | Hash one or more columns into a new column |
| [`col_coalesce`](ops/column.md) | Take the first non-null value across multiple columns |

### Math Operations

| Method | Description |
|--------|-------------|
| [`math_add`](ops/math.md) | Add a scalar value to a column |
| [`math_subtract`](ops/math.md) | Subtract a scalar value from a column |
| [`math_multiply`](ops/math.md) | Multiply a column by a scalar value |
| [`math_divide`](ops/math.md) | Divide a column by a scalar value |
| [`math_clamp`](ops/math.md) | Clamp column values to a range |
| [`math_abs`](ops/math.md) | Take absolute value of a column |
| [`math_round`](ops/math.md) | Round a column to specified decimal places |
| [`math_set_min`](ops/math.md) | Set a minimum value for a column |
| [`math_set_max`](ops/math.md) | Set a maximum value for a column |
| [`math_add_columns`](ops/math.md) | Add two columns together into a new column |
| [`math_subtract_columns`](ops/math.md) | Subtract one column from another |
| [`math_multiply_columns`](ops/math.md) | Multiply two columns together |
| [`math_divide_columns`](ops/math.md) | Divide one column by another |
| [`math_percent_of`](ops/math.md) | Calculate percentage of one column relative to another |
| [`math_cumsum`](ops/math.md) | Calculate cumulative sum (optionally grouped) |
| [`math_rank`](ops/math.md) | Calculate rank of values |

### Row Operations

| Method | Description |
|--------|-------------|
| [`rows_filter`](ops/rows.md) | Filter rows using a Filter expression |
| [`rows_drop`](ops/rows.md) | Drop rows matching a filter |
| [`rows_drop_nulls`](ops/rows.md) | Drop rows with null values |
| [`rows_flag`](ops/rows.md) | Add a flag column based on a filter condition |
| [`rows_unique`](ops/rows.md) | Keep unique rows based on specified columns |
| [`rows_deduplicate`](ops/rows.md) | Deduplicate by keeping first/last based on sort order |
| [`rows_sort`](ops/rows.md) | Sort rows by one or more columns |
| [`rows_head`](ops/rows.md) | Keep only the first n rows |
| [`rows_tail`](ops/rows.md) | Keep only the last n rows |
| [`rows_sample`](ops/rows.md) | Sample rows from the DataFrame |
| [`rows_explode`](ops/rows.md) | Explode a list column into multiple rows |
| [`rows_melt`](ops/rows.md) | Unpivot from wide to long format |
| [`rows_pivot`](ops/rows.md) | Pivot from long to wide format |

### String Operations

| Method | Description |
|--------|-------------|
| [`str_lower`](ops/string.md) | Convert string column to lowercase |
| [`str_upper`](ops/string.md) | Convert string column to uppercase |
| [`str_strip`](ops/string.md) | Strip leading and trailing characters |
| [`str_pad`](ops/string.md) | Pad a string column to a specified length |
| [`str_slice`](ops/string.md) | Extract a substring from a string column |
| [`str_truncate`](ops/string.md) | Truncate strings to a maximum length |
| [`str_replace`](ops/string.md) | Replace occurrences of a pattern |
| [`str_extract`](ops/string.md) | Extract substring using regex capture group |
| [`str_split`](ops/string.md) | Split a string column by separator |
| [`str_concat`](ops/string.md) | Concatenate multiple string columns |

### Datetime Operations

| Method | Description |
|--------|-------------|
| [`dt_year`](ops/datetime.md) | Extract year from a datetime column |
| [`dt_month`](ops/datetime.md) | Extract month from a datetime column |
| [`dt_day`](ops/datetime.md) | Extract day from a datetime column |
| [`dt_week`](ops/datetime.md) | Extract ISO week number |
| [`dt_quarter`](ops/datetime.md) | Extract quarter (1-4) |
| [`dt_year_month`](ops/datetime.md) | Create a year-month string |
| [`dt_quarter_year`](ops/datetime.md) | Create a quarter-year string (e.g., 'Q1-2024') |
| [`dt_calendar_week`](ops/datetime.md) | Create a year-week string (e.g., '2024-W05') |
| [`dt_format`](ops/datetime.md) | Format a datetime column as a string |
| [`dt_parse`](ops/datetime.md) | Parse a string column into a datetime |
| [`dt_diff_days`](ops/datetime.md) | Calculate difference in days between two dates |
| [`dt_age_years`](ops/datetime.md) | Calculate age in years from a birth date |
| [`dt_truncate`](ops/datetime.md) | Truncate datetime to a specified precision |
| [`dt_is_between`](ops/datetime.md) | Check if date falls within a range |

### Map Operations

| Method | Description |
|--------|-------------|
| [`map_values`](ops/map.md) | Map values in a column using a dictionary |
| [`map_case`](ops/map.md) | Apply case-when logic to a column |
| [`map_from_column`](ops/map.md) | Map values using another column as lookup |
| [`map_discretize`](ops/map.md) | Discretize a numeric column into bins |
| [`map_bool_to_int`](ops/map.md) | Convert boolean to integer (True=1, False=0) |
| [`map_null_to_value`](ops/map.md) | Replace null values with a specific value |
| [`map_value_to_null`](ops/map.md) | Replace a specific value with null |

## Utility Functions

| Function | Description |
|----------|-------------|
| [`frame_hash`](protocol.md#transformplan.protocol.frame_hash) | Compute deterministic hash of a DataFrame |
