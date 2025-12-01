from smart_pipeline.discovery import SmartDiscoveryEngine
from smart_pipeline.validation import run_smart_validation


def _run_with_context(connection, tables_cfg):
    engine = SmartDiscoveryEngine(connection, tables_cfg)
    df, spark_session = engine.execute()
    results = run_smart_validation(connection, tables_cfg, spark_session)
    return df, spark_session, results


def test_validation_passes_on_balanced_data(sample_connection, tables_cfg):
    df, spark_session, results = _run_with_context(sample_connection, tables_cfg)

    statuses = {r.check_name: r.status for r in results}
    assert spark_session.selected_strategy == "header_only"
    assert statuses["Reconciliation"] == "PASS"
    assert statuses["JoinAnalysis"] in {"PASS", "WARN"}
    assert len(df) == 2


def test_validation_flags_reconciliation_gap(mismatched_connection, tables_cfg):
    _, _, results = _run_with_context(mismatched_connection, tables_cfg)
    rec = next(r for r in results if r.check_name == "Reconciliation")
    assert rec.status == "FAIL"
    assert abs(rec.details["diff"]) > 0
