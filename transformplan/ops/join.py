"""Join operations mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from typing_extensions import Self


class JoinOps:
    """Mixin providing join operations."""

    if TYPE_CHECKING:

        def _register(self, op_name: str, params: dict[str, Any]) -> Self: ...

    def join(
        self,
        on: str | Sequence[str],
        right_name: str,
        how: Literal["inner", "left"] = "inner",
        *,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        suffix: str = "_right",
        select_columns: Sequence[str] | None = None,
    ) -> Self:
        """Join with a reference table resolved at execution time.

        Args:
            on: Join key column(s) (both sides if left_on/right_on not set).
            right_name: Symbolic name resolved via references in process().
            how: "inner" for filtering, "left" for enrichment.
            left_on: Left-side join columns (overrides on).
            right_on: Right-side join columns (overrides on).
            suffix: Suffix for duplicate column names from right table.
            select_columns: Columns to keep from right table (None = all).

        Returns:
            Self for method chaining.
        """
        on_list = [on] if isinstance(on, str) else list(on)
        params: dict[str, Any] = {
            "on": on_list,
            "right_name": right_name,
            "how": how,
            "suffix": suffix,
        }
        if left_on is not None:
            params["left_on"] = [left_on] if isinstance(left_on, str) else list(left_on)
        if right_on is not None:
            params["right_on"] = (
                [right_on] if isinstance(right_on, str) else list(right_on)
            )
        if select_columns is not None:
            params["select_columns"] = list(select_columns)
        return self._register("join", params)
