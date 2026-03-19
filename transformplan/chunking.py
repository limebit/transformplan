"""Chunked processing support for large files.

This module provides infrastructure for processing large Parquet files in chunks,
with support for partition keys to keep related rows together.

Classes:
    ChunkMode: Enum classifying operation compatibility with chunking.
    OperationMeta: Metadata about an operation's chunking behavior.
    ChunkValidationResult: Result of validating a pipeline for chunked processing.
    ChunkingError: Exception raised when pipeline is incompatible with chunking.
    ChunkInfo: Information about a processed chunk.
    ChunkedProtocol: Protocol for tracking chunked processing.

Example:
    >>> plan = TransformPlan().col_rename("id", "patient_id")
    >>> result, protocol = plan.process_chunked(
    ...     source="patients.parquet",
    ...     partition_key="patient_id",
    ...     chunk_size=100_000,
    ... )
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any


class ChunkMode(Enum):
    """Classification of operation compatibility with chunked processing.

    CHUNKABLE: Can process any chunk independently.
    GROUP_DEPENDENT: Needs all rows for a group together (e.g., rows_unique).
    GLOBAL: Requires full dataset, blocked in chunked mode (e.g., rows_sort).
    """

    CHUNKABLE = auto()
    GROUP_DEPENDENT = auto()
    GLOBAL = auto()


@dataclass
class OperationMeta:
    """Metadata about an operation's chunking behavior.

    Attributes:
        chunk_mode: How the operation behaves in chunked processing.
        group_param: Name of parameter containing group columns (for GROUP_DEPENDENT).
    """

    chunk_mode: ChunkMode
    group_param: str | None = None


# Registry mapping operation names to their chunking metadata.
# All operations are categorized based on their data dependencies.
OPERATION_CHUNK_REGISTRY: dict[str, OperationMeta] = {
    # Column operations - all chunkable (row-independent)
    "col_drop": OperationMeta(ChunkMode.CHUNKABLE),
    "col_rename": OperationMeta(ChunkMode.CHUNKABLE),
    "col_cast": OperationMeta(ChunkMode.CHUNKABLE),
    "col_reorder": OperationMeta(ChunkMode.CHUNKABLE),
    "col_select": OperationMeta(ChunkMode.CHUNKABLE),
    "col_duplicate": OperationMeta(ChunkMode.CHUNKABLE),
    "col_fill_null": OperationMeta(ChunkMode.CHUNKABLE),
    "col_drop_null": OperationMeta(ChunkMode.CHUNKABLE),
    "col_drop_zero": OperationMeta(ChunkMode.CHUNKABLE),
    "col_add": OperationMeta(ChunkMode.CHUNKABLE),
    "col_add_uuid": OperationMeta(ChunkMode.CHUNKABLE),
    "col_hash": OperationMeta(ChunkMode.CHUNKABLE),
    "col_coalesce": OperationMeta(ChunkMode.CHUNKABLE),
    # Math scalar operations - all chunkable
    "math_add": OperationMeta(ChunkMode.CHUNKABLE),
    "math_subtract": OperationMeta(ChunkMode.CHUNKABLE),
    "math_multiply": OperationMeta(ChunkMode.CHUNKABLE),
    "math_divide": OperationMeta(ChunkMode.CHUNKABLE),
    "math_clamp": OperationMeta(ChunkMode.CHUNKABLE),
    "math_abs": OperationMeta(ChunkMode.CHUNKABLE),
    "math_round": OperationMeta(ChunkMode.CHUNKABLE),
    "math_set_min": OperationMeta(ChunkMode.CHUNKABLE),
    "math_set_max": OperationMeta(ChunkMode.CHUNKABLE),
    # Math column operations - all chunkable
    "math_add_columns": OperationMeta(ChunkMode.CHUNKABLE),
    "math_subtract_columns": OperationMeta(ChunkMode.CHUNKABLE),
    "math_multiply_columns": OperationMeta(ChunkMode.CHUNKABLE),
    "math_divide_columns": OperationMeta(ChunkMode.CHUNKABLE),
    "math_percent_of": OperationMeta(ChunkMode.CHUNKABLE),
    # Math aggregate operations - group-dependent
    "math_cumsum": OperationMeta(ChunkMode.GROUP_DEPENDENT, group_param="group_by"),
    "math_rank": OperationMeta(ChunkMode.GROUP_DEPENDENT, group_param="group_by"),
    "math_diff_from_agg": OperationMeta(ChunkMode.GROUP_DEPENDENT, group_param="group_by"),
    # String operations - all chunkable
    "str_replace": OperationMeta(ChunkMode.CHUNKABLE),
    "str_slice": OperationMeta(ChunkMode.CHUNKABLE),
    "str_truncate": OperationMeta(ChunkMode.CHUNKABLE),
    "str_lower": OperationMeta(ChunkMode.CHUNKABLE),
    "str_upper": OperationMeta(ChunkMode.CHUNKABLE),
    "str_strip": OperationMeta(ChunkMode.CHUNKABLE),
    "str_pad": OperationMeta(ChunkMode.CHUNKABLE),
    "str_split": OperationMeta(ChunkMode.CHUNKABLE),
    "str_concat": OperationMeta(ChunkMode.CHUNKABLE),
    "str_extract": OperationMeta(ChunkMode.CHUNKABLE),
    # Datetime operations - all chunkable
    "dt_year": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_month": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_day": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_week": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_quarter": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_year_month": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_quarter_year": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_calendar_week": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_parse": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_format": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_diff_days": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_age_years": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_is_between": OperationMeta(ChunkMode.CHUNKABLE),
    "dt_truncate": OperationMeta(ChunkMode.CHUNKABLE),
    # Map operations - all chunkable
    "map_values": OperationMeta(ChunkMode.CHUNKABLE),
    "map_discretize": OperationMeta(ChunkMode.CHUNKABLE),
    "map_bool_to_int": OperationMeta(ChunkMode.CHUNKABLE),
    "map_null_to_value": OperationMeta(ChunkMode.CHUNKABLE),
    "map_value_to_null": OperationMeta(ChunkMode.CHUNKABLE),
    "map_case": OperationMeta(ChunkMode.CHUNKABLE),
    "map_from_column": OperationMeta(ChunkMode.CHUNKABLE),
    # Row operations - mixed
    "rows_filter": OperationMeta(ChunkMode.CHUNKABLE),
    "rows_drop": OperationMeta(ChunkMode.CHUNKABLE),
    "rows_flag": OperationMeta(ChunkMode.CHUNKABLE),
    "rows_explode": OperationMeta(ChunkMode.CHUNKABLE),
    "rows_drop_nulls": OperationMeta(ChunkMode.CHUNKABLE),
    "rows_melt": OperationMeta(ChunkMode.CHUNKABLE),
    # Row operations - group-dependent
    "rows_unique": OperationMeta(ChunkMode.GROUP_DEPENDENT, group_param="columns"),
    "rows_deduplicate": OperationMeta(ChunkMode.GROUP_DEPENDENT, group_param="columns"),
    # Row operations - global (blocked)
    "rows_sort": OperationMeta(ChunkMode.GLOBAL),
    "rows_pivot": OperationMeta(ChunkMode.GLOBAL),
    "rows_sample": OperationMeta(ChunkMode.GLOBAL),
    "rows_head": OperationMeta(ChunkMode.GLOBAL),
    "rows_tail": OperationMeta(ChunkMode.GLOBAL),
}


@dataclass
class ChunkValidationResult:
    """Result of validating a pipeline for chunked processing.

    Attributes:
        is_valid: Whether the pipeline can be processed in chunks.
        errors: List of error messages explaining incompatibilities.
        warnings: List of warning messages (non-blocking).
        global_operations: Names of operations that require full dataset.
        group_dependent_ops: List of (operation, columns) for group-dependent ops.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    global_operations: list[str] = field(default_factory=list)
    group_dependent_ops: list[tuple[str, list[str] | None]] = field(
        default_factory=list
    )

    def __str__(self) -> str:
        """Return string representation of validation result."""
        lines = []
        if self.is_valid:
            lines.append("Pipeline is compatible with chunked processing.")
        else:
            lines.append("Pipeline is NOT compatible with chunked processing.")

        if self.errors:
            lines.append("\nErrors:")
            lines.extend(f"  - {error}" for error in self.errors)

        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)

        if self.global_operations:
            lines.append(f"\nGlobal operations (blocked): {self.global_operations}")

        if self.group_dependent_ops:
            lines.append("\nGroup-dependent operations:")
            for op, cols in self.group_dependent_ops:
                lines.append(f"  - {op}: groups by {cols}")

        return "\n".join(lines)


class ChunkingError(Exception):
    """Raised when a pipeline is incompatible with chunked processing.

    Attributes:
        validation_result: The validation result containing error details.
    """

    def __init__(
        self, message: str, validation_result: ChunkValidationResult | None = None
    ) -> None:
        """Initialize ChunkingError with message and optional validation result."""
        super().__init__(message)
        self.validation_result = validation_result


@dataclass
class ChunkInfo:
    """Information about a processed chunk.

    Attributes:
        chunk_index: Zero-based index of this chunk.
        input_rows: Number of rows in the input chunk.
        output_rows: Number of rows after processing.
        input_hash: Hash of the input chunk data.
        output_hash: Hash of the output chunk data.
        elapsed_seconds: Processing time for this chunk.
    """

    chunk_index: int
    input_rows: int
    output_rows: int
    input_hash: str
    output_hash: str
    elapsed_seconds: float


class ChunkedProtocol:
    """Protocol for tracking chunked processing with per-chunk information.

    Tracks the overall processing as well as individual chunk statistics.

    Attributes:
        VERSION: Protocol version string.
    """

    VERSION = "1.0"

    def __init__(self) -> None:
        """Initialize an empty ChunkedProtocol."""
        self._chunks: list[ChunkInfo] = []
        self._source_path: str | None = None
        self._partition_key: list[str] | None = None
        self._chunk_size: int | None = None
        self._created_at: str = datetime.now(timezone.utc).isoformat()
        self._metadata: dict[str, Any] = {}
        self._operations: list[dict[str, Any]] = []

    def set_source(
        self,
        path: str,
        partition_key: list[str] | None,
        chunk_size: int,
    ) -> None:
        """Set source file information."""
        self._source_path = path
        self._partition_key = partition_key
        self._chunk_size = chunk_size

    def set_operations(self, operations: list[dict[str, Any]]) -> None:
        """Record the operations that were applied."""
        self._operations = operations

    def set_metadata(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Set arbitrary metadata on the protocol."""
        self._metadata.update(kwargs)

    def add_chunk(self, chunk_info: ChunkInfo) -> None:
        """Add information about a processed chunk."""
        self._chunks.append(chunk_info)

    @property
    def chunks(self) -> list[ChunkInfo]:
        """List of chunk information.

        Returns:
            List of ChunkInfo instances.
        """
        return self._chunks

    @property
    def total_input_rows(self) -> int:
        """Total rows across all input chunks.

        Returns:
            Sum of input rows.
        """
        return sum(c.input_rows for c in self._chunks)

    @property
    def total_output_rows(self) -> int:
        """Total rows across all output chunks.

        Returns:
            Sum of output rows.
        """
        return sum(c.output_rows for c in self._chunks)

    @property
    def total_elapsed_seconds(self) -> float:
        """Total processing time across all chunks.

        Returns:
            Sum of elapsed seconds.
        """
        return sum(c.elapsed_seconds for c in self._chunks)

    @property
    def num_chunks(self) -> int:
        """Number of chunks processed.

        Returns:
            Count of chunks.
        """
        return len(self._chunks)

    @property
    def metadata(self) -> dict[str, Any]:
        """Protocol metadata.

        Returns:
            Dictionary of metadata.
        """
        return self._metadata

    def output_hash(self) -> str:
        """Compute a combined hash of all output chunk hashes.

        Returns:
            A 16-character hex hash of all chunk output hashes combined.
        """
        if not self._chunks:
            return ""
        combined = "|".join(c.output_hash for c in self._chunks)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize protocol to a dictionary.

        Returns:
            Dictionary representation of the protocol.
        """
        return {
            "version": self.VERSION,
            "created_at": self._created_at,
            "metadata": self._metadata,
            "source": {
                "path": self._source_path,
                "partition_key": self._partition_key,
                "chunk_size": self._chunk_size,
            },
            "operations": self._operations,
            "summary": {
                "num_chunks": self.num_chunks,
                "total_input_rows": self.total_input_rows,
                "total_output_rows": self.total_output_rows,
                "total_elapsed_seconds": round(self.total_elapsed_seconds, 4),
                "output_hash": self.output_hash(),
            },
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "input_rows": c.input_rows,
                    "output_rows": c.output_rows,
                    "input_hash": c.input_hash,
                    "output_hash": c.output_hash,
                    "elapsed_seconds": round(c.elapsed_seconds, 4),
                }
                for c in self._chunks
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkedProtocol:
        """Deserialize protocol from a dictionary.

        Returns:
            ChunkedProtocol instance.
        """
        protocol = cls()
        protocol._created_at = data.get("created_at", protocol._created_at)
        protocol._metadata = data.get("metadata", {})

        source = data.get("source", {})
        protocol._source_path = source.get("path")
        protocol._partition_key = source.get("partition_key")
        protocol._chunk_size = source.get("chunk_size")

        protocol._operations = data.get("operations", [])

        for chunk_data in data.get("chunks", []):
            protocol._chunks.append(
                ChunkInfo(
                    chunk_index=chunk_data["chunk_index"],
                    input_rows=chunk_data["input_rows"],
                    output_rows=chunk_data["output_rows"],
                    input_hash=chunk_data["input_hash"],
                    output_hash=chunk_data["output_hash"],
                    elapsed_seconds=chunk_data["elapsed_seconds"],
                )
            )

        return protocol

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialize protocol to JSON.

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
    def from_json(cls, source: str | Path) -> ChunkedProtocol:
        """Deserialize protocol from JSON.

        Args:
            source: Either a JSON string or a path to a JSON file.

        Returns:
            ChunkedProtocol instance.
        """
        if isinstance(source, Path) or not source.strip().startswith("{"):
            content = Path(source).read_text()
        else:
            content = source

        return cls.from_dict(json.loads(content))

    def __repr__(self) -> str:
        """Return string representation of the protocol.

        Returns:
            Human-readable representation.
        """
        return (
            f"ChunkedProtocol({self.num_chunks} chunks, {self.total_input_rows} rows)"
        )

    def __len__(self) -> int:
        """Return number of chunks processed.

        Returns:
            Count of chunks.
        """
        return self.num_chunks

    def summary(self) -> str:
        """Generate a human-readable summary of the chunked processing.

        Returns:
            Formatted string summary of the protocol.
        """
        lines = [
            "=" * 70,
            "CHUNKED PROCESSING PROTOCOL",
            "=" * 70,
        ]

        if self._metadata:
            for key, value in self._metadata.items():
                lines.append(f"{key}: {value}")
            lines.append("-" * 70)

        # Source info
        if self._source_path:
            lines.append(f"Source: {self._source_path}")
        if self._partition_key:
            lines.append(f"Partition key: {self._partition_key}")
        if self._chunk_size:
            lines.append(f"Target chunk size: {self._chunk_size:,}")
        lines.extend(
            [
                "-" * 70,
                f"Chunks processed: {self.num_chunks}",
                f"Total input rows: {self.total_input_rows:,}",
                f"Total output rows: {self.total_output_rows:,}",
            ]
        )
        rows_diff = self.total_output_rows - self.total_input_rows
        if rows_diff != 0:
            lines.append(f"Row change: {rows_diff:+,}")
        lines.append(f"Total time: {self.total_elapsed_seconds:.4f}s")
        if self.num_chunks > 0:
            avg_time = self.total_elapsed_seconds / self.num_chunks
            lines.append(f"Avg time per chunk: {avg_time:.4f}s")
        lines.extend((f"Output hash: {self.output_hash()}", "-" * 70))

        # Per-chunk details
        if self._chunks:
            lines.extend(
                (
                    "",
                    f"{'#':<6} {'Input':<12} {'Output':<12} {'Change':<10} {'Time':<10} {'Hash':<16}",
                    "-" * 70,
                )
            )

            for chunk in self._chunks:
                idx = str(chunk.chunk_index)
                input_rows = f"{chunk.input_rows:,}"
                output_rows = f"{chunk.output_rows:,}"
                change = chunk.output_rows - chunk.input_rows
                change_str = f"{change:+,}" if change != 0 else "-"
                time_str = f"{chunk.elapsed_seconds:.4f}s"
                hash_str = chunk.output_hash

                lines.append(
                    f"{idx:<6} {input_rows:<12} {output_rows:<12} {change_str:<10} {time_str:<10} {hash_str:<16}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)

    def print(self) -> None:
        """Print the protocol summary to stdout."""
        print(self.summary())  # noqa: T201


def validate_chunked_pipeline(  # noqa: C901
    operations: list[tuple[str, dict[str, Any]]],
    partition_key: str | list[str] | None = None,
) -> ChunkValidationResult:
    """Validate that a pipeline is compatible with chunked processing.

    Args:
        operations: List of (op_name, params) tuples from the pipeline.
        partition_key: Column(s) used for partitioning.

    Returns:
        ChunkValidationResult with validation details.
    """
    errors: list[str] = []
    warnings: list[str] = []
    global_ops: list[str] = []
    group_ops: list[tuple[str, list[str] | None]] = []

    # Normalize partition key to list
    if partition_key is None:
        partition_cols: set[str] = set()
    elif isinstance(partition_key, str):
        partition_cols = {partition_key}
    else:
        partition_cols = set(partition_key)

    for op_name, params in operations:
        meta = OPERATION_CHUNK_REGISTRY.get(op_name)

        if meta is None:
            # Unknown operation - warn but don't block
            warnings.append(f"Unknown operation '{op_name}' - assuming chunkable")
            continue

        if meta.chunk_mode == ChunkMode.GLOBAL:
            global_ops.append(op_name)
            errors.append(
                f"Operation '{op_name}' requires the full dataset and cannot be used "
                "with chunked processing"
            )

        elif meta.chunk_mode == ChunkMode.GROUP_DEPENDENT:
            group_param = meta.group_param
            group_cols = params.get(group_param) if group_param else None

            # Normalize to list
            if isinstance(group_cols, str):
                group_cols = [group_cols]

            group_ops.append((op_name, group_cols))

            if group_cols is None:
                # No grouping specified = global operation
                errors.append(
                    f"Operation '{op_name}' without '{group_param}' parameter requires "
                    "the full dataset. Either specify group columns or remove this operation."
                )
            elif not partition_cols:
                # Group-dependent but no partition key
                errors.append(
                    f"Operation '{op_name}' groups by {group_cols} but no partition_key "
                    "is specified. Set partition_key to include these columns."
                )
            elif not set(group_cols).issubset(partition_cols):
                # Group columns not covered by partition key
                missing = set(group_cols) - partition_cols
                errors.append(
                    f"Operation '{op_name}' groups by {group_cols} but partition_key "
                    f"is {list(partition_cols)}. Missing columns: {list(missing)}"
                )

    return ChunkValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        global_operations=global_ops,
        group_dependent_ops=group_ops,
    )
