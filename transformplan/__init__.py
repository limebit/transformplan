"""TransformPlan - Safe and reproducible data transformation.

TransformPlan is a Python library for building data transformation pipelines
with built-in schema validation, audit trails, and reproducibility guarantees.

Main Classes:
    TransformPlan: Build and execute transformation pipelines with method chaining.
    Protocol: Audit trail capturing transformation history with deterministic hashes.
    Col: Column reference for building filter expressions.
    Filter: Base class for serializable filter expressions.

Validation Classes:
    ValidationResult: Result of schema validation.
    SchemaValidationError: Exception raised on validation failure.
    DryRunResult: Preview of pipeline execution without modifying data.

Utility Functions:
    frame_hash: Compute deterministic hash of a DataFrame.

Example:
    >>> import polars as pl
    >>> from transformplan import TransformPlan, Col
    >>>
    >>> df = pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    >>> plan = TransformPlan().rows_filter(Col("age") >= 18)
    >>> result, protocol = plan.process(df)
    >>> protocol.print()
"""

from transformplan.chunking import ChunkedProtocol, ChunkingError, ChunkValidationResult
from transformplan.filters import Col, Filter
from transformplan.plan import TransformPlan
from transformplan.protocol import Protocol, frame_hash
from transformplan.validation import (
    DryRunResult,
    SchemaValidationError,
    ValidationResult,
)

__all__ = [
    "ChunkValidationResult",
    "ChunkedProtocol",
    "ChunkingError",
    "Col",
    "DryRunResult",
    "Filter",
    "Protocol",
    "SchemaValidationError",
    "TransformPlan",
    "ValidationResult",
    "frame_hash",
]
