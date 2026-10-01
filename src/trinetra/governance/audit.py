"""Local JSON Lines append-only audit store and read-only JSON export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Iterable

from trinetra.governance.decisions import Decision, GOVERNANCE_SCHEMA_VERSION, GovernanceDecision, utc_now

AUDIT_EXPORT_SCHEMA_VERSION = 1


class AuditStoreCorruptError(ValueError):
    """The local audit history is malformed; it is never silently truncated."""


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    event_type: str
    timestamp: str
    run_id: str
    finding_id: str
    affected_image_id: str | int | None
    source_detector: str
    source_recommendation: str
    analyst_decision: str | None
    actor: str
    rationale: str
    evidence_references: tuple[str, ...]
    module: str
    mode: str
    decision_id: str | None = None
    schema_version: int = GOVERNANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for key in ("event_id", "timestamp", "run_id", "finding_id", "source_detector", "source_recommendation", "actor", "rationale", "module"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
        if self.event_type not in {"finding_reviewed", "governance_decision"}:
            raise ValueError(f"Unsupported audit event type: {self.event_type}")
        if self.schema_version != GOVERNANCE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported audit schema_version: {self.schema_version}")
        if self.mode not in {"demo", "pipeline"}:
            raise ValueError("mode must be demo or pipeline")
        if self.event_type == "governance_decision" and self.analyst_decision not in {"ACCEPT", "REVIEW", "QUARANTINE"}:
            raise ValueError("governance_decision event requires a valid analyst_decision")
        if self.event_type == "finding_reviewed" and self.analyst_decision is not None:
            raise ValueError("finding_reviewed event must not carry an analyst_decision")
        if any(not isinstance(ref, str) or not ref.strip() for ref in self.evidence_references):
            raise ValueError("evidence_references must contain non-empty strings")
        try:
            parsed = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("timestamp must be an ISO-8601 value") from exc
        if parsed.tzinfo is None:
            raise ValueError("timestamp must include a timezone")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_references"] = list(self.evidence_references)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        try:
            values = dict(data)
            values["evidence_references"] = tuple(values["evidence_references"])
            return cls(**values)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid audit event: {exc}") from exc


class AuditStore:
    """File-backed append-only event stream. Missing files represent an empty log."""

    def __init__(self, path: str | Path = "outputs/audit/events.jsonl") -> None:
        self.path = Path(path)

    def load_events(self) -> tuple[AuditEvent, ...]:
        if not self.path.exists():
            return ()
        events: list[AuditEvent] = []
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        raise AuditStoreCorruptError(f"Blank audit record at line {line_number}")
                    try:
                        raw = json.loads(line)
                        if not isinstance(raw, dict):
                            raise ValueError("record must be a JSON object")
                        events.append(AuditEvent.from_dict(raw))
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        raise AuditStoreCorruptError(f"Invalid audit record at line {line_number}: {exc}") from exc
        except OSError:
            raise
        return tuple(events)

    def append_many(self, events: Iterable[AuditEvent]) -> None:
        additions = tuple(events)
        if not additions:
            return
        # Validate the existing history first; never append to or mask corrupt data.
        self.load_events()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            for event in additions:
                stream.write(json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def append_decision(self, decision: GovernanceDecision) -> tuple[AuditEvent, AuditEvent]:
        reviewed = AuditEvent(
            event_id=f"{decision.decision_id}:reviewed", event_type="finding_reviewed",
            timestamp=decision.timestamp, run_id=decision.run_id, finding_id=decision.finding_id,
            affected_image_id=decision.affected_image_id, source_detector=decision.source_detector,
            source_recommendation=decision.source_recommendation, analyst_decision=None,
            actor=decision.actor, rationale=decision.rationale,
            evidence_references=decision.evidence_references, module=decision.module,
            mode=decision.mode, decision_id=decision.decision_id,
        )
        decided = AuditEvent(
            event_id=f"{decision.decision_id}:decision", event_type="governance_decision",
            timestamp=decision.timestamp, run_id=decision.run_id, finding_id=decision.finding_id,
            affected_image_id=decision.affected_image_id, source_detector=decision.source_detector,
            source_recommendation=decision.source_recommendation, analyst_decision=decision.decision.value,
            actor=decision.actor, rationale=decision.rationale,
            evidence_references=decision.evidence_references, module=decision.module,
            mode=decision.mode, decision_id=decision.decision_id,
        )
        self.append_many((reviewed, decided))
        return reviewed, decided

    def export_json(self, *, exported_at: str | None = None) -> str:
        events = self.load_events()
        document = {
            "schema_version": AUDIT_EXPORT_SCHEMA_VERSION,
            "exported_at": exported_at or utc_now(),
            "event_count": len(events),
            "events": [event.to_dict() for event in events],
        }
        return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    def latest_decision(self, *, run_id: str, finding_id: str) -> AuditEvent | None:
        return next((event for event in reversed(self.load_events())
                     if event.event_type == "governance_decision" and event.run_id == run_id
                     and event.finding_id == finding_id), None)


def decision_from_finding(finding: Any, *, decision: str, actor: str, rationale: str,
                          mode: str, decision_id: str, timestamp: str | None = None) -> GovernanceDecision:
    """Adapt a generic dashboard finding without modifying its detector result."""
    refs = tuple(dict.fromkeys((finding.original_result_reference, finding.finding_id)))
    return GovernanceDecision(
        decision_id=decision_id, decision=Decision(decision), finding_id=finding.finding_id,
        affected_image_id=finding.affected_image_id, run_id=finding.run_id, actor=actor.strip(),
        timestamp=timestamp or utc_now(), rationale=rationale.strip(), evidence_references=refs,
        source_recommendation=finding.recommendation, source_detector=finding.source_detector,
        module=finding.layer, mode=mode,
    )


__all__ = ["AUDIT_EXPORT_SCHEMA_VERSION", "AuditEvent", "AuditStore", "AuditStoreCorruptError", "decision_from_finding"]
