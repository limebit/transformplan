"""String operations mixin.

This module provides the StrOps mixin class with text manipulation operations
on DataFrame string columns.

Classes:
    StrOps: Mixin providing string operations.

Transformation Operations:
    str_lower: Convert to lowercase.
    str_upper: Convert to uppercase.
    str_strip: Strip whitespace/characters.
    str_pad: Pad to fixed length.

Substring Operations:
    str_slice: Extract substring by position.
    str_truncate: Truncate with suffix.
    str_replace: Replace pattern.
    str_extract: Extract with regex.

Splitting/Joining:
    str_split: Split into columns or rows.
    str_concat: Concatenate columns.

Example:
    >>> plan = TransformPlan().str_lower("email").str_strip("name")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from typing import Any, Callable

    from typing_extensions import Self


class StrOps:
    """Mixin providing string operations on columns."""

    if TYPE_CHECKING:

        def _register(
            self,
            method: Callable[..., pl.DataFrame],
            params: dict[str, Any],
        ) -> Self: ...

    def str_replace(
        self,
        column: str,
        pattern: str,
        replacement: str,
        *,
        literal: bool = True,
    ) -> Self:
        """Replace occurrences of a pattern in a string column.

        Args:
            column: Column to modify.
            pattern: Pattern to search for.
            replacement: String to replace with.
            literal: If True, treat pattern as literal string. If False, treat as regex.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._str_replace,
            {
                "column": column,
                "pattern": pattern,
                "replacement": replacement,
                "literal": literal,
            },
        )

    def _str_replace(
        self,
        data: pl.DataFrame,
        column: str,
        pattern: str,
        replacement: str,
        literal: bool,  # noqa: FBT001
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.col(column).str.replace_all(pattern, replacement, literal=literal)
        )

    def str_slice(
        self,
        column: str,
        offset: int,
        length: int | None = None,
    ) -> Self:
        """Extract a substring from a string column.

        Args:
            column: Column to modify.
            offset: Start position (0-indexed, negative counts from end).
            length: Number of characters to extract (None = to end).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._str_slice, {"column": column, "offset": offset, "length": length}
        )

    def _str_slice(
        self, data: pl.DataFrame, column: str, offset: int, length: int | None
    ) -> pl.DataFrame:
        return data.with_columns(pl.col(column).str.slice(offset, length))

    def str_truncate(self, column: str, max_length: int, suffix: str = "...") -> Self:
        """Truncate strings to a maximum length with optional suffix.

        Args:
            column: Column to modify.
            max_length: Maximum length of the string (including suffix).
            suffix: Suffix to append to truncated strings.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._str_truncate,
            {"column": column, "max_length": max_length, "suffix": suffix},
        )

    def _str_truncate(
        self, data: pl.DataFrame, column: str, max_length: int, suffix: str
    ) -> pl.DataFrame:
        cut_length = max_length - len(suffix)
        return data.with_columns(
            pl.when(pl.col(column).str.len_chars() > max_length)
            .then(pl.col(column).str.slice(0, cut_length) + suffix)
            .otherwise(pl.col(column))
            .alias(column)
        )

    def str_split(
        self,
        column: str,
        separator: str,
        new_columns: list[str] | None = None,
        *,
        keep_original: bool = False,
    ) -> Self:
        """Split a string column by separator.

        Args:
            column: Column to split.
            separator: String to split on.
            new_columns: Names for the resulting columns. If None, explodes into rows.
            keep_original: Whether to keep the original column.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._str_split,
            {
                "column": column,
                "separator": separator,
                "new_columns": new_columns,
                "keep_original": keep_original,
            },
        )

    def _str_split(
        self,
        data: pl.DataFrame,
        column: str,
        separator: str,
        new_columns: list[str] | None,
        keep_original: bool,  # noqa: FBT001
    ) -> pl.DataFrame:
        if new_columns is None:
            # Explode into rows
            result = data.with_columns(pl.col(column).str.split(separator))
            result = result.explode(column)
        else:
            # Split into separate columns
            split_col = pl.col(column).str.split(separator)
            for i, new_col in enumerate(new_columns):
                data = data.with_columns(split_col.list.get(i).alias(new_col))
            result = data

        if not keep_original and new_columns is not None:
            result = result.drop(column)

        return result

    def str_lower(self, column: str) -> Self:
        """Convert string column to lowercase.

        Returns:
            Self for method chaining.
        """
        return self._register(self._str_lower, {"column": column})

    def _str_lower(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).str.to_lowercase())

    def str_upper(self, column: str) -> Self:
        """Convert string column to uppercase.

        Returns:
            Self for method chaining.
        """
        return self._register(self._str_upper, {"column": column})

    def _str_upper(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        return data.with_columns(pl.col(column).str.to_uppercase())

    def str_strip(self, column: str, chars: str | None = None) -> Self:
        """Strip leading and trailing characters from a string column.

        Args:
            column: Column to modify.
            chars: Characters to strip (None = whitespace).

        Returns:
            Self for method chaining.
        """
        return self._register(self._str_strip, {"column": column, "chars": chars})

    def _str_strip(
        self, data: pl.DataFrame, column: str, chars: str | None
    ) -> pl.DataFrame:
        if chars is None:
            return data.with_columns(pl.col(column).str.strip_chars())
        return data.with_columns(pl.col(column).str.strip_chars(chars))

    def str_pad(
        self,
        column: str,
        length: int,
        fill_char: str = " ",
        side: str = "left",
    ) -> Self:
        """Pad a string column to a specified length.

        Args:
            column: Column to modify.
            length: Target length.
            fill_char: Character to pad with.
            side: 'left' or 'right'.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._str_pad,
            {"column": column, "length": length, "fill_char": fill_char, "side": side},
        )

    def _str_pad(
        self, data: pl.DataFrame, column: str, length: int, fill_char: str, side: str
    ) -> pl.DataFrame:
        if side == "left":
            return data.with_columns(pl.col(column).str.pad_start(length, fill_char))
        return data.with_columns(pl.col(column).str.pad_end(length, fill_char))

    def str_concat(
        self,
        columns: list[str],
        new_column: str,
        separator: str = "",
    ) -> Self:
        """Concatenate multiple string columns into one.

        Args:
            columns: Columns to concatenate.
            new_column: Name for the new column.
            separator: Separator between values.

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._str_concat,
            {"columns": columns, "new_column": new_column, "separator": separator},
        )

    def _str_concat(
        self, data: pl.DataFrame, columns: list[str], new_column: str, separator: str
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.concat_str([pl.col(c) for c in columns], separator=separator).alias(
                new_column
            )
        )

    def str_extract(
        self,
        column: str,
        pattern: str,
        group_index: int = 1,
        new_column: str | None = None,
    ) -> Self:
        """Extract substring using regex capture group.

        Args:
            column: Column to extract from.
            pattern: Regex pattern with capture group(s).
            group_index: Which capture group to extract (1-indexed).
            new_column: Name for result column (None = modify in place).

        Returns:
            Self for method chaining.
        """
        return self._register(
            self._str_extract,
            {
                "column": column,
                "pattern": pattern,
                "group_index": group_index,
                "new_column": new_column or column,
            },
        )

    def _str_extract(
        self,
        data: pl.DataFrame,
        column: str,
        pattern: str,
        group_index: int,
        new_column: str,
    ) -> pl.DataFrame:
        return data.with_columns(
            pl.col(column)
            .str.extract(pattern, group_index=group_index)
            .alias(new_column)
        )
