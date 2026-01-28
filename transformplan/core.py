"""Core processor with registration and execution logic."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import polars as pl

from .protocol import Protocol, frame_hash
from .validation import ValidationResult, validate_schema

if TYPE_CHECKING:
    from typing import Self


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline to a dictionary."""
        steps = []
        for method, params in self._operations:
            op_name = method.__name__.lstrip("_")
            steps.append({
                "operation": op_name,
                "params": params,
            })

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
