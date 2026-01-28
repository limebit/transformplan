"""Main TransformPlan class combining all operation mixins."""

from .core import TransformPlanBase
from .ops import ColumnOps, MathOps, RowOps


class TransformPlan(TransformPlanBase, ColumnOps, MathOps, RowOps):
    """Data processor with tracked transformations.

    Usage:
        result, protocol = (
            TransformPlan()
            .col_drop("temp")
            .math_multiply("price", 1.1)
            .rows_filter(pl.col("active") == True)
            .process(df)
        )
    """

    pass
