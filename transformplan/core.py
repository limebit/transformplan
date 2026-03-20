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
from typing import TYPE_CHECKING, Any
from typing import Protocol as TypingProtocol

import polars as pl

from transformplan.backends.polars import PolarsBackend
from transformplan.chunking import (
    ChunkedProtocol,
    ChunkInfo,
    ChunkingError,
    ChunkValidationResult,
    validate_chunked_pipeline,
)
from transformplan.protocol import Protocol
from transformplan.validation import (
    DryRunResult,
    ValidationResult,
    dry_run_schema,
    validate_schema,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from typing_extensions import Self

    from transformplan.backends.base import Backend


class HasRegister(TypingProtocol):
    """Protocol for mixins that need access to _register."""

    def _register(
        self,
        op_name: str,
        params: dict[str, Any],
    ) -> Self: ...


class TransformPlanBase:
    """Base class providing operation registration and execution."""

    VERSION = "1.0"

    def __init__(self) -> None:
        """Initialize an empty TransformPlanBase."""
        self._operations: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def _resolve_backend(backend: Backend | None = None) -> Backend:
        """Resolve backend, defaulting to PolarsBackend.

        Args:
            backend: Optional backend override.

        Returns:
            The resolved backend instance.
        """
        return backend or PolarsBackend()

    def _register(
        self,
        op_name: str,
        params: dict[str, Any],
    ) -> Self:
        """Register an operation for deferred execution.

        Returns:
            Self for method chaining.
        """
        self._operations.append((op_name, params))
        return self

    def process(
        self,
        data: Any,  # noqa: ANN401
        *,
        validate: bool = True,
        references: dict[str, Any] | None = None,
        backend: Backend | None = None,
    ) -> tuple[Any, Protocol]:
        """Execute all registered operations and return transformed data with protocol.

        Args:
            data: Input data (Polars DataFrame, DuckDB relation, etc.).
            validate: If True, validate schema before execution (default).
                Set to False for performance in hot loops with pre-validated
                pipelines.
            references: Named reference tables for join operations. Keys are
                symbolic names used in join(right_name=...), values are the
                actual data (DataFrame or relation).
            backend: Backend to use for execution. Defaults to PolarsBackend.

        Returns:
            Tuple of (processed data, Protocol).

        Raises:
            ValueError: If a join operation references a table not in references.
        """
        resolved = self._resolve_backend(backend)

        if validate:
            ref_schemas = self._extract_reference_schemas(references, resolved)
            validate_schema(
                self._operations,
                resolved.get_schema(data),
                resolved,
                references=ref_schemas,
            ).raise_if_invalid()

        protocol = Protocol()
        protocol.set_input(resolved.compute_hash(data), resolved.get_shape(data))

        # Record reference hashes in protocol metadata
        if references:
            ref_meta = {}
            for name, ref_data in references.items():
                ref_meta[name] = {
                    "hash": resolved.compute_hash(ref_data),
                    "shape": list(resolved.get_shape(ref_data)),
                }
            protocol.set_metadata(references=ref_meta)

        for op_name, params in self._operations:
            old_shape = resolved.get_shape(data)
            start = time.perf_counter()

            if op_name == "join":
                right_name = params["right_name"]
                if references is None or right_name not in references:
                    msg = f"Reference '{right_name}' not found. Pass it via references={{'{right_name}': ...}} in process()."
                    raise ValueError(msg)
                dispatch_params = {k: v for k, v in params.items() if k != "right_name"}
                dispatch_params["right_data"] = references[right_name]
                data = getattr(resolved, op_name)(data, **dispatch_params)
            else:
                data = getattr(resolved, op_name)(data, **params)

            elapsed = time.perf_counter() - start
            protocol.add_step(
                operation=op_name,
                params=params,
                old_shape=old_shape,
                new_shape=resolved.get_shape(data),
                elapsed=elapsed,
                output_hash=resolved.compute_hash(data),
            )

        return data, protocol

    @staticmethod
    def _extract_reference_schemas(
        references: dict[str, Any] | None, backend: Backend
    ) -> dict[str, dict[str, Any]] | None:
        """Extract schemas from reference tables for validation.

        Args:
            references: Named reference tables.
            backend: Backend to use for schema extraction.

        Returns:
            Dict mapping reference names to their schemas, or None.
        """
        if references is None:
            return None
        return {
            name: backend.get_schema(ref_data) for name, ref_data in references.items()
        }

    def validate(
        self,
        data: Any,  # noqa: ANN401
        *,
        references: dict[str, Any] | None = None,
        backend: Backend | None = None,
    ) -> ValidationResult:
        """Validate all operations against the data schema without executing.

        Args:
            data: Input data (Polars DataFrame, DuckDB relation, etc.).
            references: Named reference tables for join operations.
            backend: Backend to use for validation. Defaults to PolarsBackend.

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
        resolved = self._resolve_backend(backend)
        ref_schemas = self._extract_reference_schemas(references, resolved)
        return validate_schema(
            self._operations,
            resolved.get_schema(data),
            resolved,
            references=ref_schemas,
        )

    def dry_run(
        self,
        data: Any,  # noqa: ANN401
        *,
        references: dict[str, Any] | None = None,
        backend: Backend | None = None,
    ) -> DryRunResult:
        """Preview what the pipeline will do without executing it.

        Performs validation and shows step-by-step schema changes,
        including which columns will be added, removed, or modified.

        Args:
            data: DataFrame to preview against.
            references: Named reference tables for join operations.
            backend: Backend to use for dry run. Defaults to PolarsBackend.

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
        resolved = self._resolve_backend(backend)
        ref_schemas = self._extract_reference_schemas(references, resolved)
        return dry_run_schema(
            self._operations,
            resolved.get_schema(data),
            resolved,
            references=ref_schemas,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline to a dictionary.

        Returns:
            Dictionary representation of the pipeline.
        """
        steps = []
        for op_name, params in self._operations:
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
            data: Dictionary with 'steps' list. Any 'backend' key from
                older serialized plans is silently ignored.

        Returns:
            New TransformPlan instance with operations loaded.

        Raises:
            ValueError: If an unknown operation or invalid parameters are encountered.
        """
        plan = cls()

        for step in data.get("steps", []):
            op_name = step["operation"]
            params = step["params"]

            # Find the public method on the class
            method = getattr(plan, op_name, None)
            if method is None:
                msg = f"Unknown operation: {op_name}"
                raise ValueError(msg)

            # Call the method with params to register the operation
            try:
                method(**params)
            except TypeError as e:
                msg = f"Invalid parameters for operation '{op_name}': {e}"
                raise ValueError(msg) from e

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
        if isinstance(source, Path) or not source.strip().startswith("{"):
            content = Path(source).read_text()
        else:
            content = source

        return cls.from_dict(json.loads(content))

    def __len__(self) -> int:
        """Return number of registered operations.

        Returns:
            Number of operations.
        """
        return len(self._operations)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            Human-readable representation.
        """
        return f"TransformPlan({len(self._operations)} operations)"

    def to_python(self, variable_name: str = "plan") -> str:
        """Generate executable Python code for this pipeline.

        Args:
            variable_name: Name for the pipeline variable.

        Returns:
            Python code string.
        """
        lines = ["from transformplan import TransformPlan, Col", ""]
        lines.extend((f"{variable_name} = (", "    TransformPlan()"))

        for op_name, params in self._operations:
            param_str = self._format_params_as_python(params)
            lines.append(f"    .{op_name}({param_str})")

        lines.append(")")
        return "\n".join(lines)

    def _format_params_as_python(self, params: dict[str, Any]) -> str:
        """Format parameters as Python code.

        Returns:
            Python code string for the parameters.
        """
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
            elif isinstance(value, (bool, int, float)):
                parts.append(f"{key}={value}")
            elif isinstance(value, (list, dict)):
                parts.append(f"{key}={value!r}")
            else:
                parts.append(f"{key}={value!r}")

        return ", ".join(parts)

    def _format_filter_as_python(  # noqa: C901
        self, filter_dict: dict[str, Any]
    ) -> str:
        """Convert a filter dict back to Col() expression string.

        Returns:
            Python code string for the filter.
        """
        filter_type = filter_dict.get("type", "")

        # Logical operators
        if filter_type == "and":
            left = self._format_filter_as_python(filter_dict["left"])
            right = self._format_filter_as_python(filter_dict["right"])
            return f"({left}) & ({right})"
        if filter_type == "or":
            left = self._format_filter_as_python(filter_dict["left"])
            right = self._format_filter_as_python(filter_dict["right"])
            return f"({left}) | ({right})"
        if filter_type == "not":
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
        if filter_type == "is_in":
            values = filter_dict.get("values", [])
            return f'Col("{col}").is_in({values!r})'
        if filter_type == "is_null":
            return f'Col("{col}").is_null()'
        if filter_type == "is_not_null":
            return f'Col("{col}").is_not_null()'
        if filter_type == "between":
            lower = filter_dict.get("lower")
            upper = filter_dict.get("upper")
            return f'Col("{col}").between({lower!r}, {upper!r})'
        if filter_type == "str_contains":
            pattern = filter_dict.get("pattern", "")
            literal = filter_dict.get("literal", True)
            return f'Col("{col}").str_contains({pattern!r}, literal={literal})'
        if filter_type == "str_starts_with":
            prefix = filter_dict.get("prefix", "")
            return f'Col("{col}").str_starts_with({prefix!r})'
        if filter_type == "str_ends_with":
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
        references: dict[str, Any] | None = None,
        backend: Backend | None = None,
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
            references: Named reference tables for join operations (reserved
                for forward compatibility).
            backend: Backend to use for execution. Defaults to PolarsBackend.

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
        resolved = self._resolve_backend(backend)
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
            from transformplan.validation import validate_schema

            schema_validation = validate_schema(self._operations, schema, resolved)
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
            {"operation": op_name, "params": params}
            for op_name, params in self._operations
        ]
        protocol.set_operations(operations_list)

        # Process chunks
        results: list[pl.DataFrame] = []
        chunk_iter = self._create_chunk_iterator(
            lazy_frame, partition_key_list, chunk_size
        )

        for chunk_index, chunk_df in enumerate(chunk_iter):
            start = time.perf_counter()
            input_hash = resolved.compute_hash(chunk_df)
            input_rows = len(chunk_df)

            # Apply all operations to this chunk
            for op_name, params in self._operations:
                chunk_df = getattr(resolved, op_name)(chunk_df, **params)

            output_hash = resolved.compute_hash(chunk_df)
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
