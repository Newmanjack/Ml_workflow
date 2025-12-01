from smart_pipeline.discovery import SmartDiscoveryEngine


def test_discovery_prefers_header_strategy(sample_connection, tables_cfg):
    engine = SmartDiscoveryEngine(sample_connection, tables_cfg)
    df, ctx = engine.execute()

    assert ctx.selected_strategy == "header_only"
    assert ctx.join_keys["header"].lower() == "orderid"
    assert ctx.join_keys["line"].lower() == "orderid"
    assert df.loc["2023-01-01", "TotalAmount"] == 100.0
    assert df.loc["2023-01-02", "TotalAmount"] == 200.0


def test_discovery_sets_dates_and_amounts(sample_connection, tables_cfg):
    engine = SmartDiscoveryEngine(sample_connection, tables_cfg)
    _, ctx = engine.execute()

    assert ctx.header_date.lower() == "orderdate"
    assert ctx.header_amount.lower() == "totalamount"
    assert ctx.line_date.lower() == "linedate"
    assert ctx.line_amount.lower() == "lineamount"
