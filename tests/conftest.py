import duckdb
import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_pipeline.config import TableConfig


@pytest.fixture
def tables_cfg():
    return TableConfig(header_table="headers", line_table="line_items")


@pytest.fixture
def sample_connection(tables_cfg):
    con = duckdb.connect(":memory:")

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

    con.register("header_df", header_df)
    con.register("line_df", line_df)
    con.execute("CREATE TABLE headers AS SELECT * FROM header_df")
    con.execute("CREATE TABLE line_items AS SELECT * FROM line_df")
    con.unregister("header_df")
    con.unregister("line_df")

    yield con
    con.close()


@pytest.fixture
def mismatched_connection(tables_cfg):
    con = duckdb.connect(":memory:")

    header_df = pd.DataFrame(
        [
            {"OrderID": "SO1", "OrderDate": "2023-01-01", "TotalAmount": 100.0},
            {"OrderID": "SO2", "OrderDate": "2023-01-02", "TotalAmount": 200.0},
        ]
    )
    line_df = pd.DataFrame(
        [
            {"OrderID": "SO1", "LineDate": "2023-01-01", "LineAmount": 30.0, "Quantity": 1},
            {"OrderID": "SO1", "LineDate": "2023-01-01", "LineAmount": 40.0, "Quantity": 1},
            {"OrderID": "SO2", "LineDate": "2023-01-02", "LineAmount": 50.0, "Quantity": 1},
            {"OrderID": "SO2", "LineDate": "2023-01-02", "LineAmount": 30.0, "Quantity": 1},
        ]
    )

    con.register("header_df", header_df)
    con.register("line_df", line_df)
    con.execute("CREATE TABLE headers AS SELECT * FROM header_df")
    con.execute("CREATE TABLE line_items AS SELECT * FROM line_df")
    con.unregister("header_df")
    con.unregister("line_df")

    yield con
    con.close()
