"""Protocol class for tracking transformation history."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
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
        self._created_at: str = datetime.utcnow().isoformat() + "Z"
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
        self._steps.append({
            "step": len(self._steps) + 1,
            "operation": operation,
            "params": params,
            "old_shape": old_shape,
            "new_shape": new_shape,
            "rows_changed": old_shape[0] - new_shape[0],
            "cols_changed": old_shape[1] - new_shape[1],
            "elapsed_seconds": round(elapsed, 4),
            "output_hash": output_hash,
        })

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
            rows.append({
                "step": 0,
                "operation": "input",
                "params": None,
                "old_shape": None,
                "new_shape": self._input_shape,
                "rows_changed": 0,
                "cols_changed": 0,
                "elapsed_seconds": 0.0,
                "output_hash": self._input_hash,
            })

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
            rows.append({
                "step": 0,
                "operation": "input",
                "params": None,
                "old_shape": None,
                "new_shape": str(list(self._input_shape)) if self._input_shape else None,
                "rows_changed": 0,
                "cols_changed": 0,
                "elapsed_seconds": 0.0,
                "output_hash": self._input_hash,
            })

        for step in self._steps:
            rows.append({
                "step": step["step"],
                "operation": step["operation"],
                "params": json.dumps(step["params"]) if step["params"] else None,
                "old_shape": str(list(step["old_shape"])),
                "new_shape": str(list(step["new_shape"])),
                "rows_changed": step["rows_changed"],
                "cols_changed": step["cols_changed"],
                "elapsed_seconds": step["elapsed_seconds"],
                "output_hash": step["output_hash"],
            })

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
            protocol._steps.append({
                "step": step["step"],
                "operation": step["operation"],
                "params": step["params"],
                "old_shape": tuple(step["old_shape"]),
                "new_shape": tuple(step["new_shape"]),
                "rows_changed": step["rows_changed"],
                "cols_changed": step["cols_changed"],
                "elapsed_seconds": step["elapsed_seconds"],
                "output_hash": step["output_hash"],
            })

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
