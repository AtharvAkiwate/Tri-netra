# Governance UI

The optional Streamlit application is a presentation and analyst-governance layer.
It does not execute detectors. Reusable dashboard contracts live in
`trinetra.pipeline`; adapters retain the original typed Layer 1C–1H results in
`DashboardSnapshot.source_results` while presenting findings to the UI.

## Providers and findings

- `DemoProvider` supplies deterministic synthetic imagery, annotations, sample
  findings, and a synthetic heatmap labeled **DEMO HEATMAP — NOT MODEL OUTPUT**.
- `PipelineProvider` adapts an externally supplied `DataIntegrityRun`; absent
  detector outputs remain absent rather than being invented.
- Layer 1A XYWH annotations are shown as dataset annotations, not model-predicted
  boxes. Pipeline heatmaps appear only when aligned artifacts are supplied.

## Decisions and local audit history

`trinetra.governance` provides generic, typed, versioned ACCEPT, REVIEW, and
QUARANTINE analyst decisions. A decision records the source detector
recommendation separately; it never changes the original finding or result.
Actor and non-empty rationale are required. The model uses generic module and
finding references so future TRI-NETRA modules can use the same workflow.
The UI requires an explicit evidence-review confirmation in addition to a
non-empty actor and rationale before submitting a decision.

The default store is `outputs/audit/events.jsonl` (override with
`TRINETRA_AUDIT_PATH`). It is a local JSON Lines append-only event log, excluded
from Git. Submitting a decision appends `finding_reviewed` and
`governance_decision` events. Missing files are treated as an empty history;
corrupt files cause a visible error and are not silently skipped or replaced.
No `run_started` event is emitted because the UI does not start analysis runs.
JSON export is read-only, includes the export schema version, timestamp, count,
and every persisted event, and does not add an `audit_exported` event to history.

Demo decisions are explicitly marked DEMO/SAMPLE in the UI and persisted with
`mode=demo`; they are not operational actions. Pipeline decisions refer to the
actual supplied finding, run, image when available, detector, recommendation,
and evidence references. The audit viewer displays scalar event fields rather
than raw Python objects.

## Run

From the project root:

```powershell
.venv\Scripts\python.exe -m streamlit run src/trinetra/ui/app.py
```

The optional Streamlit dependencies are installed with
`.venv\Scripts\python.exe -m pip install -e ".[ui]"`.
