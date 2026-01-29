# TransformPlan

<img align="left" src="docs/assets/images/logo_blue.png" alt="TransformPlan Logo" width="150" style="margin-right: 20px;">

- Safe, reproducible data transformations
- Built-in auditing and validation
- Tracks transformation history
- Validates operations against DataFrame schemas
- Generates audit trails for data processing workflows

<br clear="left"/>

## Features

- **Declarative transformations**: Build transformation pipelines using method chaining
- **Schema validation**: Validate operations before execution with dry-run capability
- **Audit trails**: Generate complete audit protocols with deterministic DataFrame hashing
- **Multi-backend support**: Works with both Polars (primary) and Pandas DataFrames
- **Serializable pipelines**: Save and load transformation plans as JSON

## Installation

```bash
pip install transformplan
```

Or with uv:

```bash
uv add transformplan
```

## Quick Start

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

## Setup

### Installation

Create a virtualenv and install all dependencies:

```bash
make install
```

### Development Installation

Install with dev dependencies and pre-commit hooks:

```bash
make install-dev
```

### Lint

Run ruff linting and pyright type checking:

```bash
make lint
```

### Format

Fix import sorting and format code with ruff:

```bash
make format
```

### Test

Run the test suite:

```bash
make test
```

### Cleanup

Delete the virtualenv and cache directories:

```bash
make clean
```

## Development

- Run python script: `uv run python <filename.py>`
- Add new dependency: `uv add <package>`
- Add dev dependency: `uv add --group dev <package>`

## License

MIT License - see [LICENSE](LICENSE) for details.
