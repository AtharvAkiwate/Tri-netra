"""Generic analyst governance decisions, separate from detector recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

GOVERNANCE_SCHEMA_VERSION = 1


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True)
class GovernanceDecision:
    decision_id: str
    decision: Decision
    finding_id: str
    affected_image_id: str | int | None
    run_id: str
    actor: str
    timestamp: str
    rationale: str
    evidence_references: tuple[str, ...]
    source_recommendation: str
    source_detector: str
    module: str
    mode: str
    schema_version: int = GOVERNANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for key in ("decision_id", "finding_id", "run_id", "actor", "rationale", "source_detector", "module"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
        if not isinstance(self.decision, Decision):
            raise ValueError("decision must be ACCEPT, REVIEW, or QUARANTINE")
        if not isinstance(self.source_recommendation, str):
            raise ValueError("source_recommendation must be a string")
        if self.mode not in {"demo", "pipeline"}:
            raise ValueError("mode must be demo or pipeline")
        if self.schema_version != GOVERNANCE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported governance schema_version: {self.schema_version}")
        _parse_timestamp(self.timestamp)
        if any(not isinstance(ref, str) or not ref.strip() for ref in self.evidence_references):
            raise ValueError("evidence_references must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        data["evidence_references"] = list(self.evidence_references)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GovernanceDecision":
        values = dict(data)
        values["decision"] = Decision(values["decision"])
        values["evidence_references"] = tuple(values["evidence_references"])
        return cls(**values)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("timestamp must be an ISO-8601 value") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


__all__ = ["Decision", "GOVERNANCE_SCHEMA_VERSION", "GovernanceDecision", "utc_now"]
