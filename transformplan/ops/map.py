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

import polars as pl

if TYPE_CHECKING:
    from typing import Callable

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
            self._map_onehot,
            {
                "column": column,
                "categories": categories,
                "prefix": prefix or column,
                "drop": drop,
                "drop_original": drop_original,
                "unknown_value": unknown_value,
            },
        )

    def _map_onehot(
        self,
        data: pl.DataFrame,
        column: str,
        categories: list[Any] | None,
        prefix: str,
        drop: Any | None,  # noqa: ANN401
        drop_original: bool,  # noqa: FBT001
        unknown_value: str,
    ) -> pl.DataFrame:
        # Derive categories from data if not provided
        if categories is None:
            categories = data[column].drop_nulls().unique().sort().to_list()

        # Determine which category to drop (if any)
        # Literal values take precedence over keywords "first"/"last"
        drop_category: Any | None = None
        if drop is not None and categories:
            if drop in categories:
                # Literal value - drop this specific category
                drop_category = drop
            elif drop == "first":
                drop_category = categories[0]
            elif drop == "last":
                drop_category = categories[-1]
            else:
                # Value not in categories - will result in no column being dropped
                drop_category = drop

        # Build one-hot columns
        new_columns = []
        for cat in categories:
            # Skip the dropped category
            if drop_category is not None and cat == drop_category:
                continue

            col_name = f"{prefix}_{cat}"
            if unknown_value == "all_zero":
                # Unknown values get 0 for all categories
                expr = (
                    pl.when(pl.col(column) == cat)
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias(col_name)
                )
            else:
                # "ignore" - unknown values get null
                expr = (
                    pl.when(pl.col(column) == cat)
                    .then(pl.lit(1))
                    .when(pl.col(column).is_in(categories))
                    .then(pl.lit(0))
                    .otherwise(pl.lit(None))
                    .alias(col_name)
                )
            new_columns.append(expr)

        result = data.with_columns(new_columns)

        if drop_original:
            result = result.drop(column)

        return result

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
            self._map_ordinal,
            {
                "column": column,
                "categories": categories,
                "new_column": new_column or column,
                "drop_original": drop_original,
                "unknown_value": unknown_value,
            },
        )

    def _map_ordinal(
        self,
        data: pl.DataFrame,
        column: str,
        categories: list[Any] | None,
        new_column: str,
        drop_original: bool,  # noqa: FBT001
        unknown_value: int,
    ) -> pl.DataFrame:
        # Derive categories from data if not provided
        if categories is None:
            categories = data[column].drop_nulls().unique().sort().to_list()

        # Build when/then chain
        if not categories:
            return data.with_columns(pl.lit(unknown_value).alias(new_column))

        first_cat = categories[0]
        chain = pl.when(pl.col(column) == first_cat).then(pl.lit(0))

        for idx, cat in enumerate(categories[1:], start=1):
            chain = chain.when(pl.col(column) == cat).then(pl.lit(idx))

        chain = chain.otherwise(pl.lit(unknown_value))
        result = data.with_columns(chain.alias(new_column))

        if drop_original and new_column != column:
            result = result.drop(column)

        return result

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
            self._map_label,
            {
                "column": column,
                "categories": categories,
                "new_column": new_column or column,
                "drop_original": drop_original,
                "unknown_value": unknown_value,
            },
        )

    def _map_label(
        self,
        data: pl.DataFrame,
        column: str,
        categories: list[Any] | None,
        new_column: str,
        drop_original: bool,  # noqa: FBT001
        unknown_value: int,
    ) -> pl.DataFrame:
        # Label encoding is the same as ordinal encoding
        # The semantic difference is that ordinal implies meaningful order
        return self._map_ordinal(
            data, column, categories, new_column, drop_original, unknown_value
        )
