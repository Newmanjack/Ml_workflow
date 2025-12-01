import pandas as pd

from smart_pipeline.fabric import run_pipeline_on_dfs


def test_run_pipeline_on_dfs_returns_results():
    header_df = pd.DataFrame(
        [
            {"OrderID": "SO1", "OrderDate": "2023-01-01", "TotalAmount": 100.0},
            {"OrderID": "SO2", "OrderDate": "2023-01-02", "TotalAmount": 200.0},
        ]
    )
    line_df = pd.DataFrame(
        [
            {"OrderID": "SO1", "LineDate": "2023-01-01", "LineAmount": 60.0, "Quantity": 2},
            {"OrderID": "SO1", "LineDate": "2023-01-01", "LineAmount": 40.0, "Quantity": 1},
            {"OrderID": "SO2", "LineDate": "2023-01-02", "LineAmount": 120.0, "Quantity": 3},
            {"OrderID": "SO2", "LineDate": "2023-01-02", "LineAmount": 80.0, "Quantity": 1},
        ]
    )

    df, context, validation_results = run_pipeline_on_dfs(header_df, line_df)

    assert not df.empty
    assert context["selected_strategy"] == "header_only"
    assert any(vr.status == "PASS" for vr in validation_results)
