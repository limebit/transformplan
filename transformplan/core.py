"""Core processor with registration and execution logic."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from typing import Protocol as TypingProtocol

import polars as pl

from .protocol import Protocol, frame_hash
from .validation import DryRunResult, ValidationResult, dry_run_schema, validate_schema

if TYPE_CHECKING:
    from typing_extensions import Self


class HasRegister(TypingProtocol):
    """Protocol for mixins that need access to _register."""

    def _register(
        self,
        method: Callable[..., pl.DataFrame],
        params: dict[str, Any],
    ) -> Self:
        ...


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
