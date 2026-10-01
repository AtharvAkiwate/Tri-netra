"""Decision and persistent audit contract tests."""

from __future__ import annotations

import json

import pytest

from trinetra.governance.audit import AuditEvent, AuditStore, AuditStoreCorruptError, decision_from_finding
from trinetra.governance.decisions import Decision, GovernanceDecision
from trinetra.ui.providers import DemoProvider


@pytest.mark.parametrize("value", ["ACCEPT", "REVIEW", "QUARANTINE"])
def test_each_analyst_decision_preserves_detector_recommendation(value):
    finding = DemoProvider().load().findings[0]
    decision = decision_from_finding(finding, decision=value, actor="analyst",
        rationale="Review completed against supplied evidence.", mode="demo", decision_id="decision-1",
        timestamp="2026-10-01T12:00:00Z")
    assert decision.decision is Decision(value)
    assert decision.source_recommendation == finding.recommendation == "REVIEW"
    assert decision.finding_id == finding.finding_id
    assert decision.to_dict()["schema_version"] == 1
    assert GovernanceDecision.from_dict(decision.to_dict()) == decision


@pytest.mark.parametrize("actor,rationale", [("", "reason"), ("analyst", " ")])
def test_decision_requires_actor_and_nonempty_rationale(actor, rationale):
    finding = DemoProvider().load().findings[0]
    with pytest.raises(ValueError):
        decision_from_finding(finding, decision="ACCEPT", actor=actor, rationale=rationale,
            mode="demo", decision_id="decision-2", timestamp="2026-10-01T12:00:00Z")


def _decision(finding, value="REVIEW", decision_id="d1"):
    return decision_from_finding(finding, decision=value, actor="sample-analyst",
        rationale="Evidence reviewed.", mode="demo", decision_id=decision_id,
        timestamp="2026-10-01T12:00:00Z")


def test_audit_store_appends_events_roundtrips_and_exports_all_events(tmp_path):
    store = AuditStore(tmp_path / "audit" / "events.jsonl")
    assert store.load_events() == ()
    finding = DemoProvider().load().findings[0]
    store.append_decision(_decision(finding))
    store.append_decision(_decision(finding, "QUARANTINE", "d2"))
    events = store.load_events()
    assert [event.event_type for event in events] == ["finding_reviewed", "governance_decision"] * 2
    assert store.latest_decision(run_id=finding.run_id, finding_id=finding.finding_id).analyst_decision == "QUARANTINE"
    before = store.path.read_bytes()
    export = json.loads(store.export_json(exported_at="2026-10-01T12:01:00Z"))
    assert export["schema_version"] == 1 and export["event_count"] == 4
    assert export["exported_at"] == "2026-10-01T12:01:00Z"
    assert len(export["events"]) == 4
    assert store.path.read_bytes() == before
    assert AuditEvent.from_dict(events[0].to_dict()) == events[0]


def test_corrupt_log_is_reported_and_never_overwritten(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"bad":true}\n', encoding="utf-8")
    before = path.read_bytes()
    store = AuditStore(path)
    with pytest.raises(AuditStoreCorruptError, match="line 1"):
        store.load_events()
    with pytest.raises(AuditStoreCorruptError):
        store.append_decision(_decision(DemoProvider().load().findings[0]))
    assert path.read_bytes() == before


def test_pipeline_decision_uses_actual_finding_references(tmp_path):
    from trinetra.pipeline.adapters import adapt_data_integrity_run
    from tests.test_ui.test_pipeline_contracts import _pipeline_inputs
    run, _ = _pipeline_inputs(tmp_path)
    finding = adapt_data_integrity_run(run).findings[0]
    decision = decision_from_finding(finding, decision="ACCEPT", actor="analyst-4",
        rationale="Verified as expected seasonal variation.", mode="pipeline", decision_id="pipeline-choice")
    assert decision.run_id == "run-ui-real-01"
    assert decision.finding_id == finding.finding_id
    assert decision.affected_image_id == finding.affected_image_id
    assert decision.source_detector == finding.source_detector
    assert decision.evidence_references == (finding.original_result_reference, finding.finding_id)
    assert decision.source_recommendation == finding.recommendation


def test_demo_workflow_is_deterministic_and_explicitly_demo():
    snapshot = DemoProvider().load()
    left = _decision(snapshot.findings[0], decision_id="stable-id")
    right = _decision(DemoProvider().load().findings[0], decision_id="stable-id")
    assert left == right
    assert left.mode == "demo"
    assert left.affected_image_id == "IMG-0441"
