"""Protocol class for tracking transformation history.

This module provides the Protocol class for capturing audit trails and the
frame_hash function for computing deterministic DataFrame hashes.

Classes:
    Protocol: Captures transformation history with hashes, timing, and metadata.

Functions:
    frame_hash: Compute a deterministic, order-invariant hash of a DataFrame.

The Protocol enables reproducibility verification by tracking input/output hashes
and recording each transformation step with timing and shape change information.

Example:
    >>> from transformplan import TransformPlan
    >>>
    >>> result, protocol = TransformPlan().col_drop("temp").process(df)
    >>> protocol.print()  # View formatted summary
    >>> protocol.to_json("audit.json")  # Save for compliance
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


def frame_hash(df: pl.DataFrame) -> str:
    """Compute a deterministic hash of a DataFrame.

    The hash is:
    - Row-order invariant (sorted row hashes)
    - Column-order invariant (columns sorted before hashing)
    - Content-sensitive (any value change = different hash)

    Args:
        df: The DataFrame to hash.

    Returns:
        A 16-character hex string.
    """
    # Sort columns for column-order invariance
    sorted_cols = sorted(df.columns)
    df_sorted = df.select(sorted_cols)

    # Schema hash (sorted columns + dtypes)
    schema_str = str([(col, str(df_sorted.schema[col])) for col in sorted_cols])

    # Row hashes (sorted for row-order invariance)
    row_hashes = df_sorted.hash_rows().sort().to_list()

    # Combine
    content = f"{schema_str}|{row_hashes}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class Protocol:
    """Captures the transformation history for auditability."""

    VERSION = "1.0"

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []
        self._input_hash: str | None = None
        self._input_shape: tuple[int, int] | None = None
        self._created_at: str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._metadata: dict[str, Any] = {}

    def set_input(self, hash_value: str, shape: tuple[int, int]) -> None:
        """Set the hash and shape of the input DataFrame."""
        self._input_hash = hash_value
        self._input_shape = shape

    def set_metadata(self, **kwargs: Any) -> None:
        """Set arbitrary metadata on the protocol.

        Example:
            protocol.set_metadata(author="alice", project="analysis-v2")
        """
        self._metadata.update(kwargs)

    def add_step(
        self,
        operation: str,
        params: dict[str, Any],
        old_shape: tuple[int, int],
        new_shape: tuple[int, int],
        elapsed: float,
        output_hash: str,
    ) -> None:
        self._steps.append(
            {
                "step": len(self._steps) + 1,
                "operation": operation,
                "params": params,
                "old_shape": old_shape,
                "new_shape": new_shape,
                "rows_changed": old_shape[0] - new_shape[0],
                "cols_changed": old_shape[1] - new_shape[1],
                "elapsed_seconds": round(elapsed, 4),
                "output_hash": output_hash,
            }
        )

    @property
    def input_hash(self) -> str | None:
        """Hash of the input DataFrame."""
        return self._input_hash

    @property
    def output_hash(self) -> str | None:
        """Hash of the final output DataFrame."""
        if not self._steps:
            return self._input_hash
        return self._steps[-1]["output_hash"]

    @property
    def metadata(self) -> dict[str, Any]:
        """Protocol metadata."""
        return self._metadata

    def to_dataframe(self) -> pl.DataFrame:
        rows = []

        # Step 0: input state
        if self._input_hash is not None:
            rows.append(
                {
                    "step": 0,
                    "operation": "input",
                    "params": None,
                    "old_shape": None,
                    "new_shape": self._input_shape,
                    "rows_changed": 0,
                    "cols_changed": 0,
                    "elapsed_seconds": 0.0,
                    "output_hash": self._input_hash,
                }
            )

        rows.extend(self._steps)
        return pl.DataFrame(rows)

    def to_csv(self, path: str | Path) -> None:
        """Write protocol to CSV file.

        Params are serialized as JSON strings to avoid nested data issues.

        Args:
            path: File path to write to.
        """
        rows = []

        # Step 0: input state
        if self._input_hash is not None:
            rows.append(
                {
                    "step": 0,
                    "operation": "input",
                    "params": None,
                    "old_shape": None,
                    "new_shape": str(list(self._input_shape))
                    if self._input_shape
                    else None,
                    "rows_changed": 0,
                    "cols_changed": 0,
                    "elapsed_seconds": 0.0,
                    "output_hash": self._input_hash,
                }
            )

        for step in self._steps:
            rows.append(
                {
                    "step": step["step"],
                    "operation": step["operation"],
                    "params": json.dumps(step["params"]) if step["params"] else None,
                    "old_shape": str(list(step["old_shape"])),
                    "new_shape": str(list(step["new_shape"])),
                    "rows_changed": step["rows_changed"],
                    "cols_changed": step["cols_changed"],
                    "elapsed_seconds": step["elapsed_seconds"],
                    "output_hash": step["output_hash"],
                }
            )

        pl.DataFrame(rows).write_csv(path)

    def to_dict(self) -> dict[str, Any]:
        """Serialize protocol to a dictionary."""
        return {
            "version": self.VERSION,
            "created_at": self._created_at,
            "metadata": self._metadata,
            "input": {
                "hash": self._input_hash,
                "shape": list(self._input_shape) if self._input_shape else None,
            },
            "steps": [
                {
                    "step": s["step"],
                    "operation": s["operation"],
                    "params": s["params"],
                    "old_shape": list(s["old_shape"]),
                    "new_shape": list(s["new_shape"]),
                    "rows_changed": s["rows_changed"],
                    "cols_changed": s["cols_changed"],
                    "elapsed_seconds": s["elapsed_seconds"],
                    "output_hash": s["output_hash"],
                }
                for s in self._steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Protocol:
        """Deserialize protocol from a dictionary."""
        protocol = cls()
        protocol._created_at = data.get("created_at", protocol._created_at)
        protocol._metadata = data.get("metadata", {})

        input_data = data.get("input", {})
        protocol._input_hash = input_data.get("hash")
        shape = input_data.get("shape")
        protocol._input_shape = tuple(shape) if shape else None

        for step in data.get("steps", []):
            protocol._steps.append(
                {
                    "step": step["step"],
                    "operation": step["operation"],
                    "params": step["params"],
                    "old_shape": tuple(step["old_shape"]),
                    "new_shape": tuple(step["new_shape"]),
                    "rows_changed": step["rows_changed"],
                    "cols_changed": step["cols_changed"],
                    "elapsed_seconds": step["elapsed_seconds"],
                    "output_hash": step["output_hash"],
                }
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
    def from_json(cls, source: str | Path) -> Protocol:
        """Deserialize protocol from JSON.

        Args:
            source: Either a JSON string or a path to a JSON file.

        Returns:
            Protocol instance.
        """
        if isinstance(source, Path) or (
            isinstance(source, str) and not source.strip().startswith("{")
        ):
            # Treat as file path
            content = Path(source).read_text()
        else:
            # Treat as JSON string
            content = source

        return cls.from_dict(json.loads(content))

    def __repr__(self) -> str:
        return f"Protocol({len(self._steps)} steps)"

    def __len__(self) -> int:
        return len(self._steps)

    def summary(self, show_params: bool = True) -> str:
        """Generate a clean, human-readable summary of the protocol.

        Args:
            show_params: Whether to include operation parameters.

        Returns:
            Formatted string summary.
        """
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append("TRANSFORM PROTOCOL")
        lines.append("=" * 70)

        # Metadata
        if self._metadata:
            for key, value in self._metadata.items():
                lines.append(f"{key}: {value}")
            lines.append("-" * 70)

        # Input info
        if self._input_hash:
            shape_str = (
                f"{self._input_shape[0]} rows × {self._input_shape[1]} cols"
                if self._input_shape
                else "unknown"
            )
            lines.append(f"Input:  {shape_str}  [{self._input_hash}]")

        # Output info
        if self._steps:
            final = self._steps[-1]
            shape_str = f"{final['new_shape'][0]} rows × {final['new_shape'][1]} cols"
            lines.append(f"Output: {shape_str}  [{final['output_hash']}]")

        # Total time
        total_time = sum(s["elapsed_seconds"] for s in self._steps)
        lines.append(f"Total time: {total_time:.4f}s")
        lines.append("-" * 70)

        # Steps
        lines.append("")
        lines.append(
            f"{'#':<4} {'Operation':<20} {'Rows':<12} {'Cols':<12} {'Time':<10} {'Hash':<16}"
        )
        lines.append("-" * 70)

        # Input row
        if self._input_hash:
            shape = self._input_shape or (0, 0)
            lines.append(
                f"{'0':<4} {'input':<20} {shape[0]:<12} {shape[1]:<12} {'-':<10} {self._input_hash:<16}"
            )

        # Operation rows
        no_effect_steps = []
        for step in self._steps:
            step_num = str(step["step"])
            op = step["operation"]
            rows = step["new_shape"][0]
            cols = step["new_shape"][1]

            # Row/col change indicators (negative means removed)
            row_change = -step[
                "rows_changed"
            ]  # flip: positive = added, negative = removed
            col_change = -step["cols_changed"]
            row_str = str(rows)
            col_str = str(cols)
            if row_change != 0:
                row_str += f" ({row_change:+d})"
            if col_change != 0:
                col_str += f" ({col_change:+d})"

            time_str = f"{step['elapsed_seconds']:.4f}s"
            hash_str = step["output_hash"]

            # Check if step had no effect (same hash as previous)
            prev_hash = (
                self._input_hash
                if step["step"] == 1
                else self._steps[step["step"] - 2]["output_hash"]
            )
            no_effect = hash_str == prev_hash
            if no_effect:
                no_effect_steps.append(step["step"])

            # Add marker for no-effect steps
            marker = " ○" if no_effect else ""

            lines.append(
                f"{step_num:<4} {op:<20} {row_str:<12} {col_str:<12} {time_str:<10} {hash_str:<16}{marker}"
            )

            # Params
            if show_params and step["params"]:
                params_str = self._format_params(step["params"])
                lines.append(f"     └─ {params_str}")

        lines.append("=" * 70)

        # Add note about no-effect steps
        if no_effect_steps:
            lines.append(
                f"○ = no effect (steps {', '.join(map(str, no_effect_steps))} did not change data)"
            )

        return "\n".join(lines)

    def _format_params(self, params: dict, max_length: int = 60) -> str:
        """Format params dict as a readable string."""
        parts = []
        for key, value in params.items():
            if isinstance(value, dict):
                # Nested dict (like filter) - show type or summarize
                if "type" in value:
                    value_str = self._format_filter(value)
                else:
                    value_str = "{...}"
            elif isinstance(value, list) and len(value) > 3:
                value_str = f"[{value[0]}, {value[1]}, ... ({len(value)} items)]"
            else:
                value_str = repr(value)
            parts.append(f"{key}={value_str}")

        result = ", ".join(parts)
        if len(result) > max_length:
            result = result[: max_length - 3] + "..."
        return result

    def _format_filter(self, filter_dict: dict) -> str:
        """Format a filter dict as a readable expression."""
        filter_type = filter_dict.get("type", "")

        if filter_type in ("and", "or"):
            left = self._format_filter(filter_dict["left"])
            right = self._format_filter(filter_dict["right"])
            op = "&" if filter_type == "and" else "|"
            return f"({left} {op} {right})"
        elif filter_type == "not":
            operand = self._format_filter(filter_dict["operand"])
            return f"~{operand}"
        elif filter_type in ("eq", "ne", "gt", "ge", "lt", "le"):
            col = filter_dict.get("column", "?")
            val = filter_dict.get("value", "?")
            op_map = {
                "eq": "==",
                "ne": "!=",
                "gt": ">",
                "ge": ">=",
                "lt": "<",
                "le": "<=",
            }
            return f"{col} {op_map[filter_type]} {val!r}"
        elif filter_type == "is_in":
            col = filter_dict.get("column", "?")
            values = filter_dict.get("values", [])
            if len(values) > 3:
                val_str = f"[{values[0]!r}, ... ({len(values)} items)]"
            else:
                val_str = repr(values)
            return f"{col} in {val_str}"
        elif filter_type == "is_null":
            return f"{filter_dict.get('column', '?')} is null"
        elif filter_type == "is_not_null":
            return f"{filter_dict.get('column', '?')} is not null"
        elif filter_type == "between":
            col = filter_dict.get("column", "?")
            lower = filter_dict.get("lower", "?")
            upper = filter_dict.get("upper", "?")
            return f"{col} between {lower!r} and {upper!r}"
        elif filter_type.startswith("str_"):
            col = filter_dict.get("column", "?")
            if filter_type == "str_contains":
                return f"{col}.contains({filter_dict.get('pattern', '?')!r})"
            elif filter_type == "str_starts_with":
                return f"{col}.starts_with({filter_dict.get('prefix', '?')!r})"
            elif filter_type == "str_ends_with":
                return f"{col}.ends_with({filter_dict.get('suffix', '?')!r})"
        return f"<{filter_type}>"

    def print(self, show_params: bool = True) -> None:
        """Print the protocol summary to stdout.

        Args:
            show_params: Whether to include operation parameters.
        """
        print(self.summary(show_params))
