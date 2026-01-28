from transformplan import TransformPlan, Col
import polars as pl

def main():
    df = pl.read_csv('data/winequality-white.csv',
                     separator=";")

    from transformplan import TransformPlan, Col

    plan = (
        TransformPlan()
        .col_rename(column="fixed acidity", new_name="fixed_acidity")
        .col_rename(column="volatile acidity", new_name="volatile_acidity")
        .col_rename(column="citric acid", new_name="citric_acid")
        .col_rename(column="residual sugar", new_name="residual_sugar")
        .col_rename(column="free sulfur dioxide", new_name="free_sulfur_dioxide")
        .col_rename(column="total sulfur dioxide", new_name="total_sulfur_dioxide")
        .map_discretize(column="quality", bins=[5, 7], labels=['poor', 'average', 'excellent'], new_column="quality_tier", right=True)
        .math_divide_columns(column_a="fixed_acidity", column_b="volatile_acidity", new_column="acidity_ratio")
        .math_round(column="acidity_ratio", decimals=2)
        .rows_flag(Col("alcohol") >= 12, "high_alcohol")
        .rows_flag(Col("residual_sugar") > 10, "is_sweet")
        .math_percent_of(column="free_sulfur_dioxide", total_column="total_sulfur_dioxide", new_column="free_sulfur_pct", multiply_by=100.0)
        .math_round(column="free_sulfur_pct", decimals=1)
        .math_rank(column="alcohol", new_column="alcohol_rank", method="ordinal", descending=True, group_by=['quality_tier'])
        .rows_filter(Col("quality") >= 5)
        .col_drop(column="chlorides")
        .col_drop(column="density")
        .col_drop(column="sulphates")
        .rows_sort(by=['quality'], descending=True)
        .rows_head(n=100)
    )
    print(plan.validate(df))
    plan.to_json("data/pipeline.json")

    plan = TransformPlan.from_json("data/pipeline.json")
    result, protocol = plan.process_checked(df)

    protocol.print(show_params=True)


if __name__ == "__main__":
    main()
