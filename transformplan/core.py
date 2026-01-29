"""Core processor with registration and execution logic.

This module provides the base class for TransformPlan with operation registration,
execution, serialization, and validation capabilities.

Classes:
    HasRegister: Protocol for mixins that need access to the _register method.
    TransformPlanBase: Base class with execution logic and operation registry.

The TransformPlanBase uses a deferred execution model where operations are
registered via method chaining, then executed together when process() is called.
This enables validation and dry-run previews before actual data modification.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from typing import Protocol as TypingProtocol

import polars as pl

from .chunking import (
    ChunkedProtocol,
    ChunkInfo,
    ChunkingError,
    ChunkValidationResult,
    validate_chunked_pipeline,
)
from .protocol import Protocol, frame_hash
from .validation import DryRunResult, ValidationResult, dry_run_schema, validate_schema

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from typing_extensions import Self


class HasRegister(TypingProtocol):
    """Protocol for mixins that need access to _register."""

    def _register(
        self,
        method: Callable[..., pl.DataFrame],
        params: dict[str, Any],
    ) -> Self: ...


class TransformPlanBase:
    """Base class providing operation registration and execution."""

    VERSION = "1.0"

    def __init__(self) -> None:
        self._operations: list[tuple[Callable[..., pl.DataFrame], dict[str, Any]]] = []

    def _register(
        self,
        method: Callable[..., pl.DataFrame],
        params: dict[str, Any],
    ) -> Self:
        """Register an operation for deferred execution."""
        self._operations.append((method, params))
        return self

    def process(
        self, data: pl.DataFrame, validate: bool = True
    ) -> tuple[pl.DataFrame, Protocol]:
        """Execute all registered operations and return transformed data with protocol.

        Args:
            data: DataFrame to process.
            validate: If True, validate schema before execution (default).
                      Set to False for performance in hot loops with pre-validated pipelines.

        Returns:
            Tuple of (processed DataFrame, Protocol).

        Raises:
            SchemaValidationError: If validate=True and validation fails.
        """
        if validate:
            validate_schema(self._operations, dict(data.schema)).raise_if_invalid()

        protocol = Protocol()
        protocol.set_input(frame_hash(data), data.shape)

        for method, params in self._operations:
            old_shape = data.shape
            start = time.perf_counter()

            data = method(data, **params)

            elapsed = time.perf_counter() - start
            protocol.add_step(
                operation=method.__name__.lstrip("_"),
                params=params,
                old_shape=old_shape,
                new_shape=data.shape,
                elapsed=elapsed,
                output_hash=frame_hash(data),
            )

        return data, protocol

    def validate(self, data: pl.DataFrame) -> ValidationResult:
        """Validate all operations against the DataFrame schema without executing.

        Args:
            data: DataFrame to validate against.

        Returns:
            ValidationResult with any errors found.

        Example:
            plan = TransformPlan().col_drop("x").col_rename("y", "z")
            result = plan.validate(df)
            if not result.is_valid:
                for error in result.errors:
                    print(error)
            else:
                df, protocol = plan.process(df)
        """
        return validate_schema(self._operations, dict(data.schema))

    def dry_run(self, data: pl.DataFrame) -> DryRunResult:
        """Preview what the pipeline will do without executing it.

        Performs validation and shows step-by-step schema changes,
        including which columns will be added, removed, or modified.

        Args:
            data: DataFrame to preview against.

        Returns:
            DryRunResult with step-by-step preview.

        Example:
            plan = (
                TransformPlan()
                .col_drop("temp")
                .math_multiply("price", 1.1)
                .col_add("discount", value=0.0)
            )
            preview = plan.dry_run(df)
            preview.print()  # Show what will happen
            if preview.is_valid:
                df, protocol = plan.process(df)
        """
        return dry_run_schema(self._operations, dict(data.schema))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline to a dictionary."""
        steps = []
        for method, params in self._operations:
            op_name = method.__name__.lstrip("_")
            steps.append(
                {
                    "operation": op_name,
                    "params": params,
                }
            )

        return {
            "version": self.VERSION,
            "steps": steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize a pipeline from a dictionary.

        Args:
            data: Dictionary with 'steps' list.

        Returns:
            New TransformPlan instance with operations loaded.
        """
        plan = cls()

        for step in data.get("steps", []):
            op_name = step["operation"]
            params = step["params"]

            # Find the public method on the class
            method = getattr(plan, op_name, None)
            if method is None:
                raise ValueError(f"Unknown operation: {op_name}")

            # Call the method with params to register the operation
            method(**params)

        return plan

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialize the pipeline to JSON.

        Args:
            path: Optional file path to write to.
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        json_str = json.dumps(self.to_dict(), indent=indent)

        if path is not None:
            Path(path).write_text(json_str)

        return json_str

    @classmethod
    def from_json(cls, source: str | Path) -> Self:
        """Deserialize a pipeline from JSON.

        Args:
            source: Either a JSON string or a path to a JSON file.

        Returns:
            New TransformPlan instance.
        """
        if isinstance(source, Path) or (
            isinstance(source, str) and not source.strip().startswith("{")
        ):
            content = Path(source).read_text()
        else:
            content = source

        return cls.from_dict(json.loads(content))

    def __len__(self) -> int:
        """Return number of registered operations."""
        return len(self._operations)

    def __repr__(self) -> str:
        return f"TransformPlan({len(self._operations)} operations)"

    def to_python(self, variable_name: str = "plan") -> str:
        """Generate executable Python code for this pipeline.

        Args:
            variable_name: Name for the pipeline variable.

        Returns:
            Python code string.
        """
        lines = ["from transformplan import TransformPlan, Col", ""]
        lines.append(f"{variable_name} = (")
        lines.append("    TransformPlan()")

        for method, params in self._operations:
            op_name = method.__name__.lstrip("_")
            param_str = self._format_params_as_python(params)
            lines.append(f"    .{op_name}({param_str})")

        lines.append(")")
        return "\n".join(lines)

    def _format_params_as_python(self, params: dict[str, Any]) -> str:
        """Format parameters as Python code."""
        parts = []

        for key, value in params.items():
            if value is None:
                continue

            # Handle filter dicts specially
            if key == "filter" and isinstance(value, dict):
                filter_str = self._format_filter_as_python(value)
                parts.append(filter_str)
            elif isinstance(value, str):
                parts.append(f'{key}="{value}"')
            elif isinstance(value, bool):
                parts.append(f"{key}={value}")
            elif isinstance(value, (int, float)):
                parts.append(f"{key}={value}")
            elif isinstance(value, list):
                parts.append(f"{key}={value!r}")
            elif isinstance(value, dict):
                parts.append(f"{key}={value!r}")
            else:
                parts.append(f"{key}={value!r}")

        return ", ".join(parts)

    def _format_filter_as_python(self, filter_dict: dict[str, Any]) -> str:
        """Convert a filter dict back to Col() expression string."""
        filter_type = filter_dict.get("type", "")

        # Logical operators
        if filter_type == "and":
            left = self._format_filter_as_python(filter_dict["left"])
            right = self._format_filter_as_python(filter_dict["right"])
            return f"({left}) & ({right})"
        elif filter_type == "or":
            left = self._format_filter_as_python(filter_dict["left"])
            right = self._format_filter_as_python(filter_dict["right"])
            return f"({left}) | ({right})"
        elif filter_type == "not":
            operand = self._format_filter_as_python(filter_dict["operand"])
            return f"~({operand})"

        # Comparison operators
        col = filter_dict.get("column", "")
        val = filter_dict.get("value")

        op_map = {
            "eq": "==",
            "ne": "!=",
            "gt": ">",
            "ge": ">=",
            "lt": "<",
            "le": "<=",
        }

        if filter_type in op_map:
            op = op_map[filter_type]
            return f'Col("{col}") {op} {val!r}'
        elif filter_type == "is_in":
            values = filter_dict.get("values", [])
            return f'Col("{col}").is_in({values!r})'
        elif filter_type == "is_null":
            return f'Col("{col}").is_null()'
        elif filter_type == "is_not_null":
            return f'Col("{col}").is_not_null()'
        elif filter_type == "between":
            lower = filter_dict.get("lower")
            upper = filter_dict.get("upper")
            return f'Col("{col}").between({lower!r}, {upper!r})'
        elif filter_type == "str_contains":
            pattern = filter_dict.get("pattern", "")
            literal = filter_dict.get("literal", True)
            return f'Col("{col}").str_contains({pattern!r}, literal={literal})'
        elif filter_type == "str_starts_with":
            prefix = filter_dict.get("prefix", "")
            return f'Col("{col}").str_starts_with({prefix!r})'
        elif filter_type == "str_ends_with":
            suffix = filter_dict.get("suffix", "")
            return f'Col("{col}").str_ends_with({suffix!r})'

        # Fallback
        return f"Filter.from_dict({filter_dict!r})"

    def validate_chunked(
        self,
        schema: Mapping[str, Any] | None = None,
        partition_key: str | list[str] | None = None,
        data: pl.DataFrame | None = None,
    ) -> ChunkValidationResult:
        """Validate that pipeline is compatible with chunked processing.

        Args:
            schema: Schema to validate against (column names to dtypes).
                If not provided, data must be supplied.
            partition_key: Column(s) used for partitioning to keep related rows
                together.
            data: DataFrame to extract schema from (alternative to schema parameter).

        Returns:
            ChunkValidationResult with validation details.

        Raises:
            ValueError: If neither schema nor data is provided.

        Example:
            validation = plan.validate_chunked(
                schema={"patient_id": pl.Utf8, "age": pl.Int64},
                partition_key="patient_id"
            )
            if not validation.is_valid:
                print(validation)
        """
        if schema is None and data is None:
            msg = "Either schema or data must be provided"
            raise ValueError(msg)

        return validate_chunked_pipeline(self._operations, partition_key)

    def process_chunked(
        self,
        source: str | Path,
        *,
        partition_key: str | list[str] | None = None,
        chunk_size: int = 100_000,
        validate: bool = True,
    ) -> tuple[pl.DataFrame, ChunkedProtocol]:
        """Process a large Parquet file in chunks.

        This method enables processing of files that exceed available RAM by
        reading and transforming data in chunks. When a partition_key is specified,
        rows with the same partition key values are guaranteed to be processed
        together in the same chunk.

        Args:
            source: Path to Parquet file.
            partition_key: Column(s) ensuring related rows stay together.
                When set, rows with the same values in these columns will
                be processed in the same chunk.
            chunk_size: Target number of rows per chunk (approximate when
                partition_key is set, as chunks are sized to respect group
                boundaries).
            validate: Whether to validate operations before processing.

        Returns:
            Tuple of (result DataFrame, ChunkedProtocol with processing details).

        Raises:
            ChunkingError: If pipeline contains operations incompatible with
                chunked processing.
            FileNotFoundError: If source file does not exist.

        Example:
            plan = (
                TransformPlan()
                .col_rename("PatientID", "patient_id")
                .rows_filter(Col("age") >= 18)
                .rows_unique(columns=["patient_id", "visit_date"])
            )

            result, protocol = plan.process_chunked(
                source="patients_10gb.parquet",
                partition_key="patient_id",
                chunk_size=100_000,
            )
            protocol.print()
        """
        source_path = Path(source)
        if not source_path.exists():
            msg = f"Source file not found: {source_path}"
            raise FileNotFoundError(msg)

        # Normalize partition key
        partition_key_list: list[str] | None = None
        if partition_key is not None:
            if isinstance(partition_key, str):
                partition_key_list = [partition_key]
            else:
                partition_key_list = list(partition_key)

        # Validate operations for chunked processing
        if validate:
            validation = validate_chunked_pipeline(self._operations, partition_key_list)
            if not validation.is_valid:
                error_msg = "Pipeline is incompatible with chunked processing:\n"
                error_msg += "\n".join(f"  - {e}" for e in validation.errors)
                raise ChunkingError(error_msg, validation)

        # Also validate schema against first batch
        lazy_frame = pl.scan_parquet(source_path)
        schema = dict(lazy_frame.collect_schema())

        if validate:
            from .validation import validate_schema

            schema_validation = validate_schema(self._operations, schema)
            schema_validation.raise_if_invalid()

        # Initialize protocol
        protocol = ChunkedProtocol()
        protocol.set_source(
            path=str(source_path),
            partition_key=partition_key_list,
            chunk_size=chunk_size,
        )

        # Record operations
        operations_list = [
            {"operation": method.__name__.lstrip("_"), "params": params}
            for method, params in self._operations
        ]
        protocol.set_operations(operations_list)

        # Process chunks
        results: list[pl.DataFrame] = []
        chunk_iter = self._create_chunk_iterator(
            lazy_frame, partition_key_list, chunk_size
        )

        for chunk_index, chunk_df in enumerate(chunk_iter):
            start = time.perf_counter()
            input_hash = frame_hash(chunk_df)
            input_rows = len(chunk_df)

            # Apply all operations to this chunk
            for method, params in self._operations:
                chunk_df = method(chunk_df, **params)

            output_hash = frame_hash(chunk_df)
            elapsed = time.perf_counter() - start

            protocol.add_chunk(
                ChunkInfo(
                    chunk_index=chunk_index,
                    input_rows=input_rows,
                    output_rows=len(chunk_df),
                    input_hash=input_hash,
                    output_hash=output_hash,
                    elapsed_seconds=elapsed,
                )
            )
            results.append(chunk_df)

        # Combine all results
        if results:
            final = pl.concat(results, how="vertical")
        else:
            # Empty result - collect schema from lazy frame
            final = lazy_frame.head(0).collect()

        return final, protocol

    def _create_chunk_iterator(
        self,
        lazy_frame: pl.LazyFrame,
        partition_key: list[str] | None,
        chunk_size: int,
    ) -> Generator[pl.DataFrame, None, None]:
        """Create an iterator that yields DataFrames of approximately chunk_size rows.

        If partition_key is specified, ensures rows with the same partition key
        values are never split across chunks.

        Yields:
            DataFrames of approximately chunk_size rows.
        """
        if partition_key is None:
            # Simple chunking without partition awareness
            yield from self._simple_chunk_iterator(lazy_frame, chunk_size)
        else:
            # Partition-aware chunking
            yield from self._partition_chunk_iterator(
                lazy_frame, partition_key, chunk_size
            )

    def _simple_chunk_iterator(
        self,
        lazy_frame: pl.LazyFrame,
        chunk_size: int,
    ) -> Generator[pl.DataFrame, None, None]:
        """Iterate through data in fixed-size chunks.

        Yields:
            DataFrames of chunk_size rows (last chunk may be smaller).
        """
        offset = 0
        while True:
            chunk = lazy_frame.slice(offset, chunk_size).collect()
            if len(chunk) == 0:
                break
            yield chunk
            offset += chunk_size

    def _partition_chunk_iterator(
        self,
        lazy_frame: pl.LazyFrame,
        partition_key: list[str],
        chunk_size: int,
    ) -> Generator[pl.DataFrame, None, None]:
        """Iterate through data ensuring partition integrity.

        Groups rows by partition_key and yields chunks that don't split groups.

        Yields:
            DataFrames with complete partition groups (never split across chunks).
        """
        sorted_frame = lazy_frame.sort(partition_key)
        total_rows = lazy_frame.select(pl.len()).collect().item()

        if total_rows == 0:
            return

        offset = 0
        pending_rows: pl.DataFrame | None = None

        while offset < total_rows or pending_rows is not None:
            batch, offset, pending_rows = self._read_next_batch(
                sorted_frame, offset, chunk_size, total_rows, pending_rows
            )

            if batch is None or len(batch) == 0:
                break

            if offset < total_rows:
                complete, pending_rows = self._split_at_group_boundary(
                    batch, partition_key
                )
                if len(complete) > 0:
                    yield complete
                else:
                    pending_rows = batch
            else:
                yield batch

    def _read_next_batch(
        self,
        sorted_frame: pl.LazyFrame,
        offset: int,
        chunk_size: int,
        total_rows: int,
        pending_rows: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame | None, int, pl.DataFrame | None]:
        """Read the next batch of data, combining with any pending rows.

        Returns:
            Tuple of (batch DataFrame, new offset, None).
        """
        if offset < total_rows:
            batch = sorted_frame.slice(offset, chunk_size).collect()
            offset += len(batch)
            if pending_rows is not None:
                batch = pl.concat([pending_rows, batch], how="vertical")
            return batch, offset, None
        return pending_rows, offset, None

    def _split_at_group_boundary(
        self,
        batch: pl.DataFrame,
        partition_key: list[str],
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """Split batch at the last complete group boundary.

        Returns:
            Tuple of (complete rows, incomplete rows for next batch).
        """
        last_row_keys = batch.tail(1).select(partition_key)

        # Build mask for rows NOT in the last group
        not_last_group = pl.lit(value=False)
        for col in partition_key:
            last_val = last_row_keys[col][0]
            not_last_group = not_last_group | (pl.col(col) != last_val)

        complete_rows = batch.filter(not_last_group)
        incomplete_rows = batch.filter(~not_last_group)

        pending = incomplete_rows if len(incomplete_rows) > 0 else None
        return complete_rows, pending
