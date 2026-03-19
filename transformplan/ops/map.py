"""Mapping and transformation operations mixin.

This module provides the MapOps mixin class with value mapping, discretization,
encoding, and transformation operations.

Classes:
    MapOps: Mixin providing value mapping operations.

Mapping Operations:
    map_values: Map values using dictionary.
    map_case: Apply case-when logic.
    map_from_column: Lookup values from another column.

Discretization:
    map_discretize: Bin numeric values into categories.

Encoding (categorical to numeric):
    map_onehot: One-hot encoding (binary indicator columns).
    map_ordinal: Ordinal encoding (ordered integers).
    map_label: Label encoding (alphabetically sorted integers).

Type Conversion:
    map_bool_to_int: Convert boolean to integer.

Null Handling:
    map_null_to_value: Replace nulls with value.
    map_value_to_null: Replace value with null.

Example:
    >>> plan = TransformPlan().map_values("status", {"A": "Active", "I": "Inactive"})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from typing_extensions import Self


class MapOps:
    """Mixin providing value mapping and transformation operations."""

    if TYPE_CHECKING:

        def _register(
            self,
            op_name: str,
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
            "map_values",
            {
                "column": column,
                "mapping": mapping,
                "default": default,
                "keep_unmapped": keep_unmapped,
            },
        )

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
            "map_discretize",
            {
                "column": column,
                "bins": list(bins),
                "labels": list(labels) if labels else None,
                "new_column": new_column or column,
                "right": right,
            },
        )

    def map_bool_to_int(self, column: str) -> Self:
        """Convert a boolean column to integer (True=1, False=0).

        Returns:
            Self for method chaining.
        """
        return self._register("map_bool_to_int", {"column": column})

    def map_null_to_value(self, column: str, value: Any) -> Self:  # noqa: ANN401
        """Replace null values with a specific value.

        Returns:
            Self for method chaining.
        """
        return self._register(
            "map_null_to_value", {"column": column, "value": value}
        )

    def map_value_to_null(self, column: str, value: Any) -> Self:  # noqa: ANN401
        """Replace a specific value with null.

        Returns:
            Self for method chaining.
        """
        return self._register(
            "map_value_to_null", {"column": column, "value": value}
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
            "map_case",
            {
                "column": column,
                "cases": cases,
                "default": default,
                "new_column": new_column or column,
            },
        )

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
            "map_from_column",
            {
                "column": column,
                "lookup_column": lookup_column,
                "value_column": value_column,
                "new_column": new_column or column,
                "default": default,
            },
        )

    def map_onehot(
        self,
        column: str,
        categories: list[Any] | None = None,
        prefix: str | None = None,
        *,
        drop: Literal["first", "last"] | Any | None = None,  # noqa: ANN401
        drop_original: bool = True,
        unknown_value: Literal["all_zero", "ignore"] = "all_zero",
    ) -> Self:
        """One-hot encode a categorical column.

        Creates binary indicator columns (0/1) for each category.

        Args:
            column: Source column to encode.
            categories: List of category values. If None, derived from data.
            prefix: Prefix for new columns (default: column name).
            drop: Drop one category column to avoid multicollinearity:
                - None: Keep all columns (default).
                - "first": Drop the first category.
                - "last": Drop the last category.
                - Any value: Drop that specific category.
            drop_original: Drop source column after encoding (default: True).
            unknown_value: How to handle unknown values:
                - "all_zero": Set all indicator columns to 0.
                - "ignore": Keep original value behavior.

        Returns:
            Self for method chaining.

        Example:
            >>> plan.map_onehot("color", categories=["red", "green", "blue"])
            # Creates: color_red, color_green, color_blue

            >>> plan.map_onehot("color", categories=["red", "green"], drop="first")
            # Creates: color_green (drops color_red)
        """
        return self._register(
            "map_onehot",
            {
                "column": column,
                "categories": categories,
                "prefix": prefix or column,
                "drop": drop,
                "drop_original": drop_original,
                "unknown_value": unknown_value,
            },
        )

    def map_ordinal(
        self,
        column: str,
        categories: list[Any] | None = None,
        new_column: str | None = None,
        *,
        drop_original: bool = True,
        unknown_value: int = -1,
    ) -> Self:
        """Ordinal encode a categorical column.

        Maps categories to integers based on explicit ordering.

        Args:
            column: Source column to encode.
            categories: List of categories in desired order (first=0, second=1, etc.).
                If None, uses sorted unique values from data.
            new_column: Output column name. If None, replaces original.
            drop_original: Drop source column if new_column differs (default: True).
            unknown_value: Integer for unknown values (default: -1).

        Returns:
            Self for method chaining.

        Example:
            >>> plan.map_ordinal("size", categories=["small", "medium", "large"])
            # Maps: small->0, medium->1, large->2
        """
        return self._register(
            "map_ordinal",
            {
                "column": column,
                "categories": categories,
                "new_column": new_column or column,
                "drop_original": drop_original,
                "unknown_value": unknown_value,
            },
        )

    def map_label(
        self,
        column: str,
        categories: list[Any] | None = None,
        new_column: str | None = None,
        *,
        drop_original: bool = True,
        unknown_value: int = -1,
    ) -> Self:
        """Label encode a categorical column.

        Simple integer encoding (alphabetically sorted by default).

        Args:
            column: Source column to encode.
            categories: List of categories. If None, uses sorted unique values.
            new_column: Output column name. If None, replaces original.
            drop_original: Drop source column if new_column differs (default: True).
            unknown_value: Integer for unknown values (default: -1).

        Returns:
            Self for method chaining.

        Example:
            >>> plan.map_label("department")
            # Maps alphabetically: Engineering->0, HR->1, Sales->2
        """
        return self._register(
            "map_label",
            {
                "column": column,
                "categories": categories,
                "new_column": new_column or column,
                "drop_original": drop_original,
                "unknown_value": unknown_value,
            },
        )
