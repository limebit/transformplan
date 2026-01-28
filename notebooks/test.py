from transformplan import TransformPlan, Col
import polars as pl

def main():
    df = pl.read_csv('data/winequality-white.csv',
                     separator=";")

    plan = (
        TransformPlan()
        .rows_filter(Col("alcohol") >= 10)
    )
    result, protocol = plan.process_checked(df)

    protocol.to_json("data/audit_trail.json")
    protocol.to_csv('data/protocol.csv')


if __name__ == "__main__":
    main()
