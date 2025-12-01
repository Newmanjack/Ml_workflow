from smart_pipeline.discovery import SmartDiscoveryEngine
from smart_pipeline.runner import PipelineRunner
from smart_pipeline.config import PipelineConfig, TableConfig
import pandas as pd
import duckdb


def test_multiple_line_tables_combines_results():
    con = duckdb.connect(":memory:")
    header_df = pd.DataFrame(
        [
            {"OrderID": "SO1", "OrderDate": "2023-01-01", "TotalAmount": 150.0},
            {"OrderID": "SO2", "OrderDate": "2023-01-02", "TotalAmount": 300.0},
        ]
    )
    lines_a = pd.DataFrame(
        [
            {"OrderID": "SO1", "LineDate": "2023-01-01", "LineAmount": 50.0},
            {"OrderID": "SO2", "LineDate": "2023-01-02", "LineAmount": 100.0},
        ]
    )
    lines_b = pd.DataFrame(
        [
            {"OrderID": "SO1", "LineDate": "2023-01-01", "LineAmount": 100.0},
            {"OrderID": "SO2", "LineDate": "2023-01-02", "LineAmount": 200.0},
        ]
    )

    con.register("header_df", header_df)
    con.register("lines_a", lines_a)
    con.register("lines_b", lines_b)
    con.execute("CREATE TABLE headers AS SELECT * FROM header_df")
    con.execute("CREATE TABLE line_items AS SELECT * FROM lines_a")
    con.execute("CREATE TABLE line_items_returns AS SELECT * FROM lines_b")
    con.unregister("header_df")
    con.unregister("lines_a")
    con.unregister("lines_b")

    cfg = PipelineConfig(
        sources=TableConfig(
            header_table="headers",
            line_table="line_items",
            line_tables=["line_items", "line_items_returns"],
        ),
        profiling={"enabled": False},
        validation={"enabled": False},
        metadata={"persist_context": False},
    )
    runner = PipelineRunner(cfg, connection=con)
    df, ctx, results = runner.run()

    assert "TotalAmount_line_items" in df.columns
    assert "TotalAmount_line_items_returns" in df.columns
    assert df.loc["2023-01-01", "TotalAmount_line_items"] == 150.0
    assert df.loc["2023-01-02", "TotalAmount_line_items_returns"] == 300.0
