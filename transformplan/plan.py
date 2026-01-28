"""Main TransformPlan class combining all operation mixins."""

from .core import TransformPlanBase
from .ops import ColumnOps, DatetimeOps, MapOps, MathOps, RowOps, StrOps


class TransformPlan(
    TransformPlanBase, ColumnOps, DatetimeOps, MapOps, MathOps, RowOps, StrOps
):
    """Data processor with tracked transformations.

    Usage:
        result, protocol = (
            TransformPlan()
            .col_drop("temp")
            .math_multiply("price", 1.1)
            .rows_filter(Col("active") == True)
            .process(df)
        )
    """

    pass
