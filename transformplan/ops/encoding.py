"""Encoding operations mixin.

This module provides the EncodingOps mixin class with categorical encoding
operations for machine learning preparation workflows.

Classes:
    EncodingOps: Mixin providing encoding operations.

Encoding Operations:
    enc_onehot: One-hot encoding (binary indicator columns).
    enc_ordinal: Ordinal encoding (ordered integers).
    enc_label: Label encoding (alphabetically sorted integers).

Example:
    >>> plan = TransformPlan().enc_onehot("color", categories=["red", "green", "blue"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from typing import Callable, Literal

    from typing_extensions import Self


class EncodingOps:
    """Mixin providing categorical encoding operations."""

    if TYPE_CHECKING:

        def _register(
            self,
            method: Callable[..., pl.DataFrame],
            params: dict[str, Any],
        ) -> Self: ...

    def enc_onehot(
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
            >>> plan.enc_onehot("color", categories=["red", "green", "blue"])
            # Creates columns: color_red, color_green, color_blue

            >>> plan.enc_onehot("color", drop="first",
            ...                 categories=["red", "green", "blue"])
            # Creates: color_green, color_blue (drops color_red)
        """
        return self._register(
            self._enc_onehot,
            {
                "column": column,
                "categories": categories,
                "prefix": prefix or column,
                "drop": drop,
                "drop_original": drop_original,
                "unknown_value": unknown_value,
            },
        )

    def _enc_onehot(
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

    def enc_ordinal(
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
            >>> plan.enc_ordinal("size", categories=["small", "medium", "large"])
            # Maps: small→0, medium→1, large→2
        """
        return self._register(
            self._enc_ordinal,
            {
                "column": column,
                "categories": categories,
                "new_column": new_column or column,
                "drop_original": drop_original,
                "unknown_value": unknown_value,
            },
        )

    def _enc_ordinal(
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

    def enc_label(
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
            >>> plan.enc_label("department")
            # Maps alphabetically: Engineering→0, HR→1, Sales→2
        """
        return self._register(
            self._enc_label,
            {
                "column": column,
                "categories": categories,
                "new_column": new_column or column,
                "drop_original": drop_original,
                "unknown_value": unknown_value,
            },
        )

    def _enc_label(
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
        return self._enc_ordinal(
            data, column, categories, new_column, drop_original, unknown_value
        )
