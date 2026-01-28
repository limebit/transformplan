"""Core processor with registration and execution logic."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import polars as pl

from .protocol import Protocol, frame_hash
from .validation import ValidationResult, validate_schema

if TYPE_CHECKING:
    from typing import Self


class TransformPlanBase:
    """Base class providing operation registration and execution."""

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

    def process(self, data: pl.DataFrame) -> tuple[pl.DataFrame, Protocol]:
        """Execute all registered operations and return transformed data with protocol."""
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

    def process_checked(self, data: pl.DataFrame) -> tuple[pl.DataFrame, Protocol]:
        """Validate and then execute operations. Raises if validation fails.

        Args:
            data: DataFrame to process.

        Returns:
            Tuple of (processed DataFrame, Protocol).

        Raises:
            SchemaValidationError: If validation fails.
        """
        self.validate(data).raise_if_invalid()
        return self.process(data)

    def __len__(self) -> int:
        """Return number of registered operations."""
        return len(self._operations)

    def __repr__(self) -> str:
        return f"TransformPlan({len(self._operations)} operations)"
