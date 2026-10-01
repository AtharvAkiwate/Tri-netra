# Layer 1F: Contributor / Source Evidence Aggregation

Layer 1F aggregates existing Layer 1C near-duplicate pairs, Layer 1D label
manipulation review findings, and Layer 1E distribution-shift candidates. It
adapts those typed result schemas; it does not repeat detector logic.

## Provenance and attribution

`ProvenanceMetadata` associates an image ID with optional `contributor_id`,
`source_id`, and `batch_id` values. Each is independent: no field is inferred
from another. Records may be supplied explicitly to
`ContributorRiskAggregator.aggregate(dataset, detector_results,
provenance_records=...)`; explicit provenance attributes/metadata present on a
Dataset or DatasetImage are also read. Optional metadata already carried in
Layer 1D's dataset/finding/exclusion results is consumed as well. Dataset-level
values are considered explicit records for each image. If multiple records give
different values for one image and field, that field is excluded from
attribution and the conflict and values are reported. Other non-conflicting
fields on the same image remain usable.

Images with no resolved provenance remain in `unattributed_summary`, including
their detector evidence. Partial attribution is marked incomplete and does not
silently discard unassigned findings. If no provenance is available, the result
states: “Provenance metadata unavailable; contributor/source/batch aggregation
not performed.” Malformed records and provenance referring to unknown dataset
images raise explicit validation errors. Exact duplicate evidence is
deduplicated; conflicting versions of a finding are reported and excluded from
the aggregate score.

## Counts, score, and recommendations

Groups are produced separately for each available contributor, source, and
batch ID, plus combined resolved provenance tuples. Layer 1C counts unique
unordered image pairs; Layers 1D and 1E count unique flagged images within each
detector category. A pair is included in each group containing either endpoint;
if one endpoint lacks provenance, the pair is also retained in the unattributed
summary. `affected_images` counts unique group members represented in those
findings. The evidence score is the transparent weighted sum:

`near_duplicate_findings × near_duplicate_weight + label_manipulation_findings × label_manipulation_weight + distribution_shift_findings × distribution_shift_weight`

Default development weights are 1.0, 1.5, and 0.5 respectively. They are
configurable and are not validated scientific weights or statistical
probabilities. The default review score threshold is 0.0, so non-zero observed
evidence recommends `REVIEW`; configurations may change that operating
threshold. Groups without findings recommend `ACCEPT`. This layer never
automatically recommends `QUARANTINE`; that action requires external analyst
and governance decisions.

Contributor/source risk is an aggregation of observed evidence and is not proof
that a contributor or source acted maliciously. Evidence can have legitimate
causes, does not establish intent or causation, and is not a calibrated
probability. The output supports triage and review only.
