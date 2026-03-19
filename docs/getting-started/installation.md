# Installation

## Requirements

- Python 3.10 or higher
- Polars (primary DataFrame library)
- DuckDB (optional, for DuckDB backend)

## Install with pip

```bash
pip install transformplan
```

## Install with uv

```bash
uv add transformplan
```

## Optional: DuckDB Backend

To use the DuckDB backend, install DuckDB separately:

```bash
pip install duckdb
```

Or with uv:

```bash
uv add duckdb
```

## Development Installation

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/limebit/transformplan.git
cd transformplan
make install-dev
```

This will:

1. Create a virtual environment
2. Install all runtime and development dependencies
3. Install the package in editable mode

## Verify Installation

```python
import transformplan
print(transformplan.__all__)
```

You should see the list of exported classes and functions:

```python
['Backend', 'ChunkValidationResult', 'ChunkedProtocol', 'ChunkingError',
 'Col', 'DryRunResult', 'Filter', 'PolarsBackend', 'Protocol',
 'SchemaValidationError', 'TransformPlan', 'ValidationResult', 'frame_hash']
```
