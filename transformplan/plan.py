"""Main TransformPlan class combining all operation mixins.

This module provides the main TransformPlan class that combines operation mixins
for column, math, row, string, datetime, and map operations with the base
execution logic from TransformPlanBase.

Classes:
    TransformPlan: Complete transformation processor with all operations.

Example:
    >>> from transformplan import TransformPlan, Col
    >>>
    >>> plan = (
    ...     TransformPlan()
    ...     .col_drop("temp")
    ...     .math_multiply("price", 1.1)
    ...     .rows_filter(Col("active") == True)
    ... )
    >>> result, protocol = plan.process(df)
"""

from transformplan.core import TransformPlanBase
from transformplan.ops import ColumnOps, DatetimeOps, MapOps, MathOps, RowOps, StrOps


class TransformPlan(
    TransformPlanBase,
    ColumnOps,
    DatetimeOps,
    MapOps,
    MathOps,
    RowOps,
    StrOps,
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
