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

if TYPE_CHECKING:
    from typing import Any

    from typing_extensions import Self


class StrOps:
    """Mixin providing string operations on columns."""

    if TYPE_CHECKING:

        def _register(
            self,
            op_name: str,
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
            "str_replace",
            {
                "column": column,
                "pattern": pattern,
                "replacement": replacement,
                "literal": literal,
            },
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
            "str_slice", {"column": column, "offset": offset, "length": length}
        )

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
            "str_truncate",
            {"column": column, "max_length": max_length, "suffix": suffix},
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
            "str_split",
            {
                "column": column,
                "separator": separator,
                "new_columns": new_columns,
                "keep_original": keep_original,
            },
        )

    def str_lower(self, column: str) -> Self:
        """Convert string column to lowercase.

        Returns:
            Self for method chaining.
        """
        return self._register("str_lower", {"column": column})

    def str_upper(self, column: str) -> Self:
        """Convert string column to uppercase.

        Returns:
            Self for method chaining.
        """
        return self._register("str_upper", {"column": column})

    def str_strip(self, column: str, chars: str | None = None) -> Self:
        """Strip leading and trailing characters from a string column.

        Args:
            column: Column to modify.
            chars: Characters to strip (None = whitespace).

        Returns:
            Self for method chaining.
        """
        return self._register("str_strip", {"column": column, "chars": chars})

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
            "str_pad",
            {"column": column, "length": length, "fill_char": fill_char, "side": side},
        )

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
            "str_concat",
            {"columns": columns, "new_column": new_column, "separator": separator},
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
            "str_extract",
            {
                "column": column,
                "pattern": pattern,
                "group_index": group_index,
                "new_column": new_column or column,
            },
        )
