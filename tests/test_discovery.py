from smart_pipeline.discovery import SmartDiscoveryEngine


def test_discovery_prefers_header_strategy(sample_connection, tables_cfg):
    engine = SmartDiscoveryEngine(sample_connection, tables_cfg)
    df, spark_session = engine.execute()

    assert spark_session.selected_strategy == "header_only"
    assert spark_session.join_keys["header"].lower() == "orderid"
    assert spark_session.join_keys["line"].lower() == "orderid"
    assert df.loc["2023-01-01", "TotalAmount"] == 100.0
    assert df.loc["2023-01-02", "TotalAmount"] == 200.0


def test_discovery_sets_dates_and_amounts(sample_connection, tables_cfg):
    engine = SmartDiscoveryEngine(sample_connection, tables_cfg)
    _, spark_session = engine.execute()

    assert spark_session.header_date.lower() == "orderdate"
    assert spark_session.header_amount.lower() == "totalamount"
    assert spark_session.line_date.lower() == "linedate"
    assert spark_session.line_amount.lower() == "lineamount"
