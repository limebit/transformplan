"""Serializable filter expressions for reproducible row filtering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import polars as pl


class Filter(ABC):
    """Base class for all filters."""

    @abstractmethod
    def to_expr(self) -> pl.Expr:
        """Convert to a Polars expression."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Filter:
        """Deserialize from a dictionary."""
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
        """Internal deserialization (implemented by subclasses)."""
        ...

    def __and__(self, other: Filter) -> And:
        return And(self, other)

    def __or__(self, other: Filter) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)


# =============================================================================
# Column reference
# =============================================================================


class Col:
    """Column reference for building filter expressions."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, value: Any) -> Eq:  # type: ignore[override]
        return Eq(self.name, value)

    def __ne__(self, value: Any) -> Ne:  # type: ignore[override]
        return Ne(self.name, value)

    def __gt__(self, value: Any) -> Gt:
        return Gt(self.name, value)

    def __ge__(self, value: Any) -> Ge:
        return Ge(self.name, value)

    def __lt__(self, value: Any) -> Lt:
        return Lt(self.name, value)

    def __le__(self, value: Any) -> Le:
        return Le(self.name, value)

    def is_in(self, values: Sequence[Any]) -> IsIn:
        return IsIn(self.name, values)

    def is_null(self) -> IsNull:
        return IsNull(self.name)

    def is_not_null(self) -> IsNotNull:
        return IsNotNull(self.name)

    def str_contains(self, pattern: str, literal: bool = True) -> StrContains:
        return StrContains(self.name, pattern, literal)

    def str_starts_with(self, prefix: str) -> StrStartsWith:
        return StrStartsWith(self.name, prefix)

    def str_ends_with(self, suffix: str) -> StrEndsWith:
        return StrEndsWith(self.name, suffix)

    def between(self, lower: Any, upper: Any) -> Between:
        return Between(self.name, lower, upper)


# =============================================================================
# Comparison filters
# =============================================================================


@dataclass
class Eq(Filter):
    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column) == self.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "eq", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Eq:
        return cls(data["column"], data["value"])


@dataclass
class Ne(Filter):
    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column) != self.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ne", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Ne:
        return cls(data["column"], data["value"])


@dataclass
class Gt(Filter):
    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column) > self.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "gt", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Gt:
        return cls(data["column"], data["value"])


@dataclass
class Ge(Filter):
    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column) >= self.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ge", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Ge:
        return cls(data["column"], data["value"])


@dataclass
class Lt(Filter):
    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column) < self.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "lt", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Lt:
        return cls(data["column"], data["value"])


@dataclass
class Le(Filter):
    column: str
    value: Any

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column) <= self.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "le", "column": self.column, "value": self.value}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Le:
        return cls(data["column"], data["value"])


@dataclass
class IsIn(Filter):
    column: str
    values: Sequence[Any]

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column).is_in(self.values)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "is_in", "column": self.column, "values": list(self.values)}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> IsIn:
        return cls(data["column"], data["values"])


@dataclass
class Between(Filter):
    column: str
    lower: Any
    upper: Any

    def to_expr(self) -> pl.Expr:
        return pl.col(self.column).is_between(self.lower, self.upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "between",
            "column": self.column,
            "lower": self.lower,
            "upper": self.upper,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Between:
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
