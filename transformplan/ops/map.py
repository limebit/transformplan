"""Mapping and transformation operations mixin.

This module provides the MapOps mixin class with value mapping, discretization,
and transformation operations.

Classes:
    MapOps: Mixin providing value mapping operations.

Mapping Operations:
    map_values: Map values using dictionary.
    map_case: Apply case-when logic.
    map_from_column: Lookup values from another column.

Discretization:
    map_discretize: Bin numeric values into categories.

Type Conversion:
    map_bool_to_int: Convert boolean to integer.

Null Handling:
    map_null_to_value: Replace nulls with value.
    map_value_to_null: Replace value with null.

Example:
    >>> plan = TransformPlan().map_values("status", {"A": "Active", "I": "Inactive"})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import polars as pl

if TYPE_CHECKING:
    from typing import Any, Callable

    from typing_extensions import Self


class MapOps:
    """Mixin providing value mapping and transformation operations."""

    if TYPE_CHECKING:

        def _register(
            self,
            method: Callable[..., pl.DataFrame],
            params: dict[str, Any],
        ) -> Self: ...

    def map_values(
        self,
        column: str,
        mapping: dict[Any, Any],
        default: Any = None,  # noqa: ANN401
        *,
        keep_unmapped: bool = True,
    ) -> Self:
        """Map values in a column using a dictionary.

        Args:
            column: Column to transform.
            mapping: Dictionary mapping old values to new values.
            default: Default value for unmapped values (if keep_unmapped=False).
            keep_unmapped: If True, keep original value when not in mapping.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._map_values,
            {
                "column": column,
                "mapping": mapping,
                "default": default,
                "keep_unmapped": keep_unmapped,
            },
        )

    def _map_values(
        self,
        data: pl.DataFrame,
        column: str,
        mapping: dict[Any, Any],
        default: Any,  # noqa: ANN401
        keep_unmapped: bool,  # noqa: FBT001
    ) -> pl.DataFrame:
        # Build a when/then chain for the mapping
        expr = pl.col(column)

        # Start with first mapping
        items = list(mapping.items())
        if not items:
            return data

        first_key, first_val = items[0]
        chain = pl.when(expr == first_key).then(pl.lit(first_val))

        for key, val in items[1:]:
            chain = chain.when(expr == key).then(pl.lit(val))

        if keep_unmapped:
            chain = chain.otherwise(expr)
        else:
            chain = chain.otherwise(pl.lit(default))

        return data.with_columns(chain.alias(column))

    def map_discretize(
        self,
        column: str,
        bins: Sequence[float],
        labels: Sequence[str] | None = None,
        new_column: str | None = None,
        *,
        right: bool = True,
    ) -> Self:
        """Discretize a numeric column into bins/categories.

        Args:
            column: Column to discretize.
            bins: Bin edges (e.g., [0, 18, 65, 100] creates 4 bins).
            labels: Labels for each bin (must be len(bins)+1 if provided).
            new_column: Name for result column (None = modify in place).
            right: If True, bins are (left, right]. If False, [left, right).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._map_discretize,
            {
                "column": column,
                "bins": list(bins),
                "labels": list(labels) if labels else None,
                "new_column": new_column or column,
                "right": right,
            },
        )

    def _map_discretize(
        self,
        data: pl.DataFrame,
        column: str,
        bins: list[float],
        labels: list[str] | None,
        new_column: str,
        right: bool,  # noqa: FBT001
    ) -> pl.DataFrame:
        # Create labels if not provided
        if labels is None:
            labels = []
            edges = [-float("inf"), *bins, float("inf")]
            for i in range(len(edges) - 1):
                if right:
                    labels.append(f"({edges[i]}, {edges[i + 1]}]")
                else:
                    labels.append(f"[{edges[i]}, {edges[i + 1]})")

        # Build when/then chain
        col = pl.col(column)
        edges = [-float("inf"), *bins, float("inf")]

        # First bin
        if right:
            cond = (col > edges[0]) & (col <= edges[1])
        else:
            cond = (col >= edges[0]) & (col < edges[1])
        chain = pl.when(cond).then(pl.lit(labels[0]))

        # Remaining bins
        for i in range(1, len(edges) - 1):
            if right:
                cond = (col > edges[i]) & (col <= edges[i + 1])
            else:
                cond = (col >= edges[i]) & (col < edges[i + 1])
            chain = chain.when(cond).then(pl.lit(labels[i]))

        chain = chain.otherwise(pl.lit(None))

        return data.with_columns(chain.alias(new_column))

    def map_bool_to_int(self, column: str) -> Self:
        """Convert a boolean column to integer (True=1, False=0).

        Returns:
            Self for method chaining.
        """
        return self._register(self._map_bool_to_int, {"column": column})

    def _map_bool_to_int(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).cast(pl.Int64))

    def map_null_to_value(self, column: str, value: Any) -> Self:  # noqa: ANN401
        """Replace null values with a specific value.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._map_null_to_value, {"column": column, "value": value}
        )

    def _map_null_to_value(
        self,
        data: pl.DataFrame,
        column: str,
        value: Any,  # noqa: ANN401
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).fill_null(value))

    def map_value_to_null(self, column: str, value: Any) -> Self:  # noqa: ANN401
        """Replace a specific value with null.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._map_value_to_null, {"column": column, "value": value}
        )

    def _map_value_to_null(
        self,
        data: pl.DataFrame,
        column: str,
        value: Any,  # noqa: ANN401
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.when(pl.col(column) == value)
            .then(pl.lit(None))
            .otherwise(pl.col(column))
            .alias(column)
        )

    def map_case(
        self,
        column: str,
        cases: list[tuple[Any, Any]],
        default: Any = None,  # noqa: ANN401
        new_column: str | None = None,
    ) -> Self:
        """Apply case-when logic to a column.

        Args:
            column: Column to evaluate.
            cases: List of (condition_value, result_value) tuples.
            default: Default value if no case matches.
            new_column: Name for result column (None = modify in place).

        Returns:
            Self for method chaining.

        Example:
            .map_case('grade', [(90, 'A'), (80, 'B'), (70, 'C')], default='F')
            Maps: >= 90 -> A, >= 80 -> B, >= 70 -> C, else F
        """
        return self._register(
            self._map_case,
            {
                "column": column,
                "cases": cases,
                "default": default,
                "new_column": new_column or column,
            },
        )

    def _map_case(
        self,
        data: pl.DataFrame,
        column: str,
        cases: list[tuple[Any, Any]],
        default: Any,  # noqa: ANN401
        new_column: str,
    ) -> pl.DataFrame:
        if not cases:
            return data.with_columns(pl.lit(default).alias(new_column))

        col = pl.col(column)
        cond_val, result_val = cases[0]
        chain = pl.when(col == cond_val).then(pl.lit(result_val))

        for cond_val, result_val in cases[1:]:
            chain = chain.when(col == cond_val).then(pl.lit(result_val))

        chain = chain.otherwise(pl.lit(default))
        return data.with_columns(chain.alias(new_column))

    def map_from_column(
        self,
        column: str,
        lookup_column: str,
        value_column: str,
        new_column: str | None = None,
        default: Any = None,  # noqa: ANN401
    ) -> Self:
        """Map values using another column as lookup (like vlookup).

        This maps values from `column` using `lookup_column` -> `value_column` mapping
        from the same DataFrame. Useful for denormalization.

        Args:
            column: Column containing keys to look up.
            lookup_column: Column containing lookup keys.
            value_column: Column containing values to map to.
            new_column: Name for result column (None = modify in place).
            default: Default value if lookup fails.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._map_from_column,
            {
                "column": column,
                "lookup_column": lookup_column,
                "value_column": value_column,
                "new_column": new_column or column,
                "default": default,
            },
        )

    def _map_from_column(
        self,
        data: pl.DataFrame,
        column: str,
        lookup_column: str,
        value_column: str,
        new_column: str,
        default: Any,  # noqa: ANN401
    ) -> pl.DataFrame:
        # Build lookup dict from the data
        lookup = dict(
            zip(
                data[lookup_column].to_list(),
                data[value_column].to_list(),
                strict=False,
            )
        )

        return data.with_columns(
            pl.col(column).replace(lookup, default=default).alias(new_column)
        )
