"""Serializable filter expressions for reproducible row filtering.

This module provides a composable filter system for building complex row filtering
conditions that can be serialized to JSON and deserialized back. This enables
reproducible transformation pipelines that can be saved, shared, and replayed.

Main Classes:
    Col: Column reference for building filter expressions.
    Filter: Abstract base class for all filter types.

Filter Types:
    - Comparison: Eq, Ne, Gt, Ge, Lt, Le, IsIn, Between
    - Null checks: IsNull, IsNotNull
    - String matching: StrContains, StrStartsWith, StrEndsWith
    - Logical: And, Or, Not

Example:
    >>> from transformplan import Col, Filter
    >>>
    >>> # Build a filter expression
    >>> filter_expr = (Col("age") >= 18) & (Col("status") == "active")
    >>>
    >>> # Serialize to dict
    >>> filter_dict = filter_expr.to_dict()
    >>>
    >>> # Deserialize back
    >>> restored = Filter.from_dict(filter_dict)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import polars as pl


class Filter(ABC):
    """Abstract base class for all filter expressions.

    Filters are composable, serializable expressions that define row selection
    criteria. They can be combined using logical operators (&, |, ~) and
    serialized to dictionaries for storage and transmission.

    Subclasses must implement:
        - to_expr(): Convert to a Polars expression
        - to_dict(): Serialize to a dictionary
        - _from_dict(): Deserialize from a dictionary (classmethod)

    Example:
        >>> filter1 = Col("age") >= 18
        >>> filter2 = Col("status") == "active"
        >>> combined = filter1 & filter2  # And filter
        >>> inverted = ~filter1  # Not filter
    """

    @abstractmethod
    def to_expr(self) -> pl.Expr:
        """Convert to a Polars expression.

        Returns:
            A Polars expression that can be used with DataFrame.filter().
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON storage.

        The dictionary includes a 'type' key identifying the filter class,
        plus any parameters needed to reconstruct the filter.

        Returns:
            Dictionary representation of the filter.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Filter:
        """Deserialize a filter from a dictionary.

        Uses the 'type' key to determine which filter class to instantiate.

        Args:
            data: Dictionary with 'type' key and filter parameters.

        Returns:
            Reconstructed Filter instance.

        Raises:
            ValueError: If 'type' is missing or unknown.

        Example:
            >>> data = {"type": "eq", "column": "status", "value": "active"}
            >>> filter_obj = Filter.from_dict(data)
        """
        filter_type = data.get("type")
        if filter_type is None:
            raise ValueError("Missing 'type' in filter dict")

        filter_cls = _FILTER_REGISTRY.get(filter_type)
        if filter_cls is None:
            raise ValueError(f"Unknown filter type: {filter_type}")

        return filter_cls._from_dict(data)

    @classmethod
    @abstractmethod
    def _from_dict(cls, data: dict[str, Any]) -> Filter:
        """Internal deserialization (implemented by subclasses).

        Args:
            data: Dictionary containing filter parameters.

        Returns:
            New instance of the filter class.
        """
        ...

    def __and__(self, other: Filter) -> And:
        """Combine with another filter using AND logic.

        Args:
            other: Filter to combine with.

        Returns:
            New And filter requiring both conditions.
        """
        return And(self, other)

    def __or__(self, other: Filter) -> Or:
        """Combine with another filter using OR logic.

        Args:
            other: Filter to combine with.

        Returns:
            New Or filter requiring either condition.
        """
        return Or(self, other)

    def __invert__(self) -> Not:
        """Invert this filter using NOT logic.

        Returns:
            New Not filter inverting this condition.
        """
        return Not(self)


# =============================================================================
# Column reference
# =============================================================================


class Col:
    """Column reference for building filter expressions.

    Col provides a fluent interface for creating filter conditions on DataFrame
    columns. Use comparison operators and methods to build filters that can be
    combined using & (and), | (or), and ~ (not).

    Args:
        name: The name of the column to reference.

    Example:
        >>> # Comparison operators
        >>> Col("age") >= 18
        >>> Col("status") == "active"
        >>> Col("price") < 100
        >>>
        >>> # String methods
        >>> Col("email").str_contains("@company.com")
        >>> Col("name").str_starts_with("A")
        >>>
        >>> # Null checks
        >>> Col("optional").is_null()
        >>> Col("required").is_not_null()
        >>>
        >>> # Membership
        >>> Col("country").is_in(["US", "CA", "MX"])
        >>> Col("age").between(18, 65)
        >>>
        >>> # Combining conditions
        >>> (Col("age") >= 18) & (Col("status") == "active")
        >>> (Col("role") == "admin") | (Col("role") == "moderator")
    """

    def __init__(self, name: str) -> None:
        """Initialize a column reference.

        Args:
            name: The name of the column to reference.
        """
        self.name = name

    def __eq__(self, value: Any) -> Eq:  # type: ignore[override]
        """Create an equality filter (column == value).

        Args:
            value: Value to compare against.

        Returns:
            Eq filter for column equals value.
        """
        return Eq(self.name, value)

    def __ne__(self, value: Any) -> Ne:  # type: ignore[override]
        """Create an inequality filter (column != value).

        Args:
            value: Value to compare against.

        Returns:
            Ne filter for column not equals value.
        """
        return Ne(self.name, value)

    def __gt__(self, value: Any) -> Gt:
        """Create a greater-than filter (column > value).

        Args:
            value: Value to compare against.

        Returns:
            Gt filter for column greater than value.
        """
        return Gt(self.name, value)

    def __ge__(self, value: Any) -> Ge:
        """Create a greater-or-equal filter (column >= value).

        Args:
            value: Value to compare against.

        Returns:
            Ge filter for column greater than or equal to value.
        """
        return Ge(self.name, value)

    def __lt__(self, value: Any) -> Lt:
        """Create a less-than filter (column < value).

        Args:
            value: Value to compare against.

        Returns:
            Lt filter for column less than value.
        """
        return Lt(self.name, value)

    def __le__(self, value: Any) -> Le:
        """Create a less-or-equal filter (column <= value).

        Args:
            value: Value to compare against.

        Returns:
            Le filter for column less than or equal to value.
        """
        return Le(self.name, value)

    def is_in(self, values: Sequence[Any]) -> IsIn:
        """Create a membership filter (column in values).

        Args:
            values: Sequence of values to check membership against.

        Returns:
            IsIn filter for column value in the given sequence.

        Example:
            >>> Col("status").is_in(["active", "pending"])
        """
        return IsIn(self.name, values)

    def is_null(self) -> IsNull:
        """Create a null check filter (column is null).

        Returns:
            IsNull filter for column is null.

        Example:
            >>> Col("optional_field").is_null()
        """
        return IsNull(self.name)

    def is_not_null(self) -> IsNotNull:
        """Create a not-null check filter (column is not null).

        Returns:
            IsNotNull filter for column is not null.

        Example:
            >>> Col("required_field").is_not_null()
        """
        return IsNotNull(self.name)

    def str_contains(self, pattern: str, literal: bool = True) -> StrContains:
        """Create a string contains filter.

        Args:
            pattern: Substring or regex pattern to search for.
            literal: If True, treat pattern as literal string. If False, as regex.

        Returns:
            StrContains filter for column containing pattern.

        Example:
            >>> Col("email").str_contains("@company.com")
            >>> Col("description").str_contains(r"\\d+", literal=False)
        """
        return StrContains(self.name, pattern, literal)

    def str_starts_with(self, prefix: str) -> StrStartsWith:
        """Create a string starts-with filter.

        Args:
            prefix: Prefix to check for.

        Returns:
            StrStartsWith filter for column starting with prefix.

        Example:
            >>> Col("code").str_starts_with("PRD-")
        """
        return StrStartsWith(self.name, prefix)

    def str_ends_with(self, suffix: str) -> StrEndsWith:
        """Create a string ends-with filter.

        Args:
            suffix: Suffix to check for.

        Returns:
            StrEndsWith filter for column ending with suffix.

        Example:
            >>> Col("filename").str_ends_with(".csv")
        """
        return StrEndsWith(self.name, suffix)

    def between(self, lower: Any, upper: Any) -> Between:
        """Create a range filter (lower <= column <= upper).

        Args:
            lower: Lower bound (inclusive).
            upper: Upper bound (inclusive).

        Returns:
            Between filter for column within range.

        Example:
            >>> Col("age").between(18, 65)
            >>> Col("date").between("2024-01-01", "2024-12-31")
        """
        return Between(self.name, lower, upper)


# =============================================================================
# Comparison filters
# =============================================================================


@dataclass
class Eq(Filter):
    """Equality filter: column == value.

    Attributes:
        column: Name of the column to compare.
        value: Value to compare against.
    """

    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        """Convert to Polars equality expression."""
        return pl.col(self.column) == self.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "eq", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Eq:
        """Create from dictionary."""
        return cls(data["column"], data["value"])


@dataclass
class Ne(Filter):
    """Inequality filter: column != value.

    Attributes:
        column: Name of the column to compare.
        value: Value to compare against.
    """

    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        """Convert to Polars inequality expression."""
        return pl.col(self.column) != self.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "ne", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Ne:
        """Create from dictionary."""
        return cls(data["column"], data["value"])


@dataclass
class Gt(Filter):
    """Greater-than filter: column > value.

    Attributes:
        column: Name of the column to compare.
        value: Value to compare against.
    """

    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        """Convert to Polars greater-than expression."""
        return pl.col(self.column) > self.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "gt", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Gt:
        """Create from dictionary."""
        return cls(data["column"], data["value"])


@dataclass
class Ge(Filter):
    """Greater-or-equal filter: column >= value.

    Attributes:
        column: Name of the column to compare.
        value: Value to compare against.
    """

    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        """Convert to Polars greater-or-equal expression."""
        return pl.col(self.column) >= self.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "ge", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Ge:
        """Create from dictionary."""
        return cls(data["column"], data["value"])


@dataclass
class Lt(Filter):
    """Less-than filter: column < value.

    Attributes:
        column: Name of the column to compare.
        value: Value to compare against.
    """

    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        """Convert to Polars less-than expression."""
        return pl.col(self.column) < self.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "lt", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Lt:
        """Create from dictionary."""
        return cls(data["column"], data["value"])


@dataclass
class Le(Filter):
    """Less-or-equal filter: column <= value.

    Attributes:
        column: Name of the column to compare.
        value: Value to compare against.
    """

    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        """Convert to Polars less-or-equal expression."""
        return pl.col(self.column) <= self.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "le", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Le:
        """Create from dictionary."""
        return cls(data["column"], data["value"])


@dataclass
class IsIn(Filter):
    """Membership filter: column value in list of values.

    Attributes:
        column: Name of the column to check.
        values: Sequence of values to check membership against.
    """

    column: str
    values: Sequence[Any]

    def to_expr(self) -> pl.Expr:
        """Convert to Polars is_in expression."""
        return pl.col(self.column).is_in(self.values)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "is_in", "column": self.column, "values": list(self.values)}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> IsIn:
        """Create from dictionary."""
        return cls(data["column"], data["values"])


@dataclass
class Between(Filter):
    """Range filter: lower <= column <= upper.

    Attributes:
        column: Name of the column to check.
        lower: Lower bound (inclusive).
        upper: Upper bound (inclusive).
    """

    column: str
    lower: Any
    upper: Any

    def to_expr(self) -> pl.Expr:
        """Convert to Polars is_between expression."""
        return pl.col(self.column).is_between(self.lower, self.upper)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "between",
            "column": self.column,
            "lower": self.lower,
            "upper": self.upper,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Between:
        """Create from dictionary."""
        return cls(data["column"], data["lower"], data["upper"])


# =============================================================================
# Null filters
# =============================================================================


@dataclass
class IsNull(Filter):
    column: str

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column).is_null()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "is_null", "column": self.column}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> IsNull:
        return cls(data["column"])


@dataclass
class IsNotNull(Filter):
    column: str

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column).is_not_null()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "is_not_null", "column": self.column}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> IsNotNull:
        return cls(data["column"])


# =============================================================================
# String filters
# =============================================================================


@dataclass
class StrContains(Filter):
    column: str
    pattern: str
    literal: bool = True

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column).str.contains(self.pattern, literal=self.literal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "str_contains",
            "column": self.column,
            "pattern": self.pattern,
            "literal": self.literal,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> StrContains:
        return cls(data["column"], data["pattern"], data.get("literal", True))


@dataclass
class StrStartsWith(Filter):
    column: str
    prefix: str

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column).str.starts_with(self.prefix)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "str_starts_with", "column": self.column, "prefix": self.prefix}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> StrStartsWith:
        return cls(data["column"], data["prefix"])


@dataclass
class StrEndsWith(Filter):
    column: str
    suffix: str

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column).str.ends_with(self.suffix)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "str_ends_with", "column": self.column, "suffix": self.suffix}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> StrEndsWith:
        return cls(data["column"], data["suffix"])


# =============================================================================
# Logical combinators
# =============================================================================


@dataclass
class And(Filter):
    left: Filter
    right: Filter

    def to_expr(self) -> pl.Expr:
        return self.left.to_expr() & self.right.to_expr()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "and",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> And:
        return cls(
            Filter.from_dict(data["left"]),
            Filter.from_dict(data["right"]),
        )


@dataclass
class Or(Filter):
    left: Filter
    right: Filter

    def to_expr(self) -> pl.Expr:
        return self.left.to_expr() | self.right.to_expr()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "or",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Or:
        return cls(
            Filter.from_dict(data["left"]),
            Filter.from_dict(data["right"]),
        )


@dataclass
class Not(Filter):
    operand: Filter

    def to_expr(self) -> pl.Expr:
        return ~self.operand.to_expr()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "not", "operand": self.operand.to_dict()}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Not:
        return cls(Filter.from_dict(data["operand"]))


# =============================================================================
# Registry for deserialization
# =============================================================================

_FILTER_REGISTRY: dict[str, type[Filter]] = {
    "eq": Eq,
    "ne": Ne,
    "gt": Gt,
    "ge": Ge,
    "lt": Lt,
    "le": Le,
    "is_in": IsIn,
    "between": Between,
    "is_null": IsNull,
    "is_not_null": IsNotNull,
    "str_contains": StrContains,
    "str_starts_with": StrStartsWith,
    "str_ends_with": StrEndsWith,
    "and": And,
    "or": Or,
    "not": Not,
}
