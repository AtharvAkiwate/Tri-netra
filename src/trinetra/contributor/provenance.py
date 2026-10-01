"""Typed contributor/source/batch metadata and conflict resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Union

from trinetra.features.models import ImageId

ProvenanceValue = Union[str, int]
PROVENANCE_FIELDS = ("contributor_id", "source_id", "batch_id")


class InvalidProvenanceError(ValueError):
    """Raised when supplied provenance is malformed or refers to an unknown image."""


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Optional, independent provenance identifiers for one dataset image.

    Values are never copied between fields: a source ID does not imply a
    contributor ID, and a batch ID does not imply either of the others.
    """

    image_id: ImageId
    contributor_id: Optional[ProvenanceValue] = None
    source_id: Optional[ProvenanceValue] = None
    batch_id: Optional[ProvenanceValue] = None

    def __post_init__(self) -> None:
        if isinstance(self.image_id, bool) or not isinstance(self.image_id, (int, str)):
            raise InvalidProvenanceError("provenance image_id must be an int or str")
        for name in PROVENANCE_FIELDS:
            _validate_value(getattr(self, name), name)


@dataclass(frozen=True)
class ProvenanceConflict:
    """Conflicting values for a field; that field is excluded from attribution."""

    image_id: ImageId
    field_name: str
    conflicting_values: tuple[ProvenanceValue, ...]
    resolution: str = "exclude_conflicting_field_from_attribution"


def _validate_value(value: object, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise InvalidProvenanceError(f"{field_name} must be None, a string, or an integer")
    if isinstance(value, str) and not value.strip():
        raise InvalidProvenanceError(f"{field_name} cannot be an empty string")


def provenance_from_mapping(value: Mapping[str, object]) -> ProvenanceMetadata:
    """Validate a mapping instead of silently discarding malformed fields."""
    if any(not isinstance(key, str) for key in value):
        raise InvalidProvenanceError("Provenance mapping keys must be strings")
    allowed = {"image_id", *PROVENANCE_FIELDS}
    unknown = set(value) - allowed
    missing = {"image_id"} - set(value)
    if unknown:
        raise InvalidProvenanceError(f"Unknown provenance field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise InvalidProvenanceError("Provenance mapping requires image_id")
    try:
        return ProvenanceMetadata(
            image_id=value["image_id"],
            contributor_id=value.get("contributor_id"),
            source_id=value.get("source_id"),
            batch_id=value.get("batch_id"),
        )
    except TypeError as exc:
        raise InvalidProvenanceError(f"Malformed provenance record: {exc}") from exc


__all__ = [
    "InvalidProvenanceError",
    "PROVENANCE_FIELDS",
    "ProvenanceConflict",
    "ProvenanceMetadata",
    "ProvenanceValue",
    "provenance_from_mapping",
]
