# TRI-NETRA

**Trustworthy AI Security & Assurance Platform**

*Data. Model. Output. One assurance layer.*

TRI-NETRA is intended to provide evidence-based assurance across computer-vision
training data integrity, model integrity, inference/output provenance,
distribution shift and environmental anomalies, and analyst governance and
auditability. It is designed for offline and air-gapped security contexts where
datasets, models, and inference pipelines may come from multiple sources. The
five-layer platform is being developed incrementally; only Layers 1 and 5 are
currently complete.

## Problem

Computer-vision pipelines can face risks that are difficult to identify through
traditional cybersecurity controls alone, including training-data poisoning,
label manipulation, duplicate flooding, out-of-distribution data, model
modification, inference-output tampering, replay, environmental shifts, and
gaps in traceable evidence. TRI-NETRA aims to connect evidence across these
stages. Its implemented detectors report review signals and limitations; they
do not establish malicious intent or measured real-world detection accuracy.

## Team Contribution Workflow

Use a feature branch for each assigned module. Do not develop directly on
`main`.

```bash
git clone <repository>
cd Tri-netra
git checkout main
git pull origin main
git checkout -b feature/<your-feature>
```

Then follow the team process:

1. Clone the repository.
2. Create a personal feature branch.
3. Make changes only in the assigned module.
4. Add or update tests.
5. Run the full test suite.
6. Review the diff.
7. Commit the change.
8. Push the feature branch.
9. Open a Pull Request.
10. Complete team review.
11. Merge the approved Pull Request into `main`.
12. Pull the updated `main` branch before starting the next task.

### Convenience commit-and-push command

```bash
git add . && git commit -m "feat: describe your change" && git push -u origin HEAD
```

Replace the commit message with a concise description. This pushes the current
branch, which may be a feature branch rather than `main`. Run it only after
tests and diff review pass; do not use it to bypass Pull Request review. The
normal team process remains feature branch → push → Pull Request → review →
merge.

For clearer review, the safer individual commands are:

```bash
git status --short
git diff
git add <assigned-files>
git diff --cached
git commit -m "feat: describe your change"
git push -u origin HEAD
```

Never use force push as a substitute for resolving branch history.

## TRI-NETRA Five-Layer Architecture

The rows below are the five **top-level platform layers**. Layer 1A–1H are an
internal implementation breakdown of Layer 1 only; they are not additional
top-level layers.

| Layer | Module / Focus Area | Team Lead | Key Technical Deliverables | Current Status |
| --- | --- | --- | --- | --- |
| **Layer 1 — Data Integrity Engine** | Dataset, feature, label, duplicate, provenance, and cluster-separation evidence | Assigned team member | DINOv2 / ResNet-50 feature extraction script; k-NN distance ratios and silhouette scoring for clean-label poison analysis; pHash + cosine duplicate sweeping and Contributor Risk Index (CRI)-related aggregation; COCO JSON and YOLO TXT parsers; air-gapped setup/testing, sample clean/poisoned datasets, and end-to-end testing where applicable | **COMPLETE** for the internal 1A–1H implementation. Real DINOv2/ResNet inference was not verified in this Windows environment. Layer 1D currently uses neighborhood disagreement evidence; no calibrated or real-world accuracy claim is made. |
| **Layer 2 — Model Security & Neuro-Surgery** | Model behavior, backdoor analysis, and model repair | Assigned team member | White-Box Activation Clustering & Spectral Signatures; Black-Box Artificial Brain Stimulation (ABS) fallback; Auto Neuro-Surgery engine for layer pruning and `< 2 min` backdoor repair; ONNX and PyTorch `.pt` / `.pth` model loader | **IN DEVELOPMENT / ASSIGNED.** These deliverables are planned; no Layer 2 implementation is present in the current repository. |
| **Layer 3 — Inference Provenance & Hardware Seal** | Verifiable inference identity, output provenance, and sensor identity | Assigned team member | Ed25519 signing of image, model, output, and timestamp; SHA-256 Merkle-tree audit log generator; PRNU camera sensor fingerprinting | **IN DEVELOPMENT / ASSIGNED.** These deliverables are planned; no Layer 3 implementation is present in the current repository. |
| **Layer 4 — Distribution Shift & Weather Engine** | Environmental context and distribution-shift analysis | Assigned team member | Maximum Mean Discrepancy (MMD) in RKHS feature space; Mahalanobis distance for OOD detection; weather-vs-attack classifier separating conditions such as rain/fog from active laser attacks | **IN DEVELOPMENT / ASSIGNED.** Layer 1E already provides a reference-set nearest-cosine distribution-shift signal inside Layer 1. Layer 4's MMD, Mahalanobis, and weather-vs-attack deliverables are not implemented. |
| **Layer 5 — Streamlit Governance UI** | Analyst-facing review, decisions, and auditability | Assigned team member | Streamlit frontend; bounding-box view; Grad-CAM red threat heatmap overlay; ACCEPT, REVIEW, and QUARANTINE actions; one-click audit-log JSON export | **COMPLETE** for the current UI slice. Grad-CAM generation is not implemented; the synthetic demo heatmap is labeled **DEMO HEATMAP — NOT MODEL OUTPUT**. |

## Work Done

### Layer 1 — Data Integrity Engine

**Status: COMPLETE**

The internal Layer 1 implementation consists of:

| Internal component | Implemented functionality |
| --- | --- |
| **1A** | COCO JSON and YOLO TXT dataset parsing |
| **1B** | DINOv2 primary / ResNet-50 fallback feature-extraction architecture and standardized feature batches |
| **1C** | Cosine-similarity near-duplicate candidates and clusters |
| **1D** | Label manipulation / clean-label neighborhood anomaly evidence |
| **1E** | Reference-based OOD / distribution-shift detection |
| **1F** | Contributor, source, and batch evidence aggregation when provenance is available |
| **1G** | pHash duplicate detection |
| **1H** | Silhouette validation of feature-space separation |

The official k-NN distance-ratio technique is not currently implemented as a
Layer 1D score: Layer 1D reports configurable neighborhood-disagreement
evidence. Air-gapped validation, clean/poisoned sample preparation, and
end-to-end attack-scenario testing remain environment/dataset-dependent work;
the recorded software test results below do not claim those evaluations.

Final documented Data Integrity test result: **170 passed, 1 skipped, 0 failed**
(**171 total**) in **1.68s**. These tests validate implemented software
behavior; they do not establish real-world attack-detection accuracy. Real
DINOv2/ResNet inference was not verified in the Windows environment because of
the documented PyTorch/Windows security runtime restriction.

### Layer 5 — Streamlit Governance UI

**Status: COMPLETE** for the implemented governance UI slice.

Implemented functionality includes the Streamlit mission-control dashboard,
DEMO / SAMPLE mode, a TRI-NETRA PIPELINE provider architecture, Layer 1 result
adapters, finding detail views, dataset-annotation bounding boxes, ACCEPT /
REVIEW / QUARANTINE analyst decisions, actor and rationale capture with
evidence-review confirmation, a local append-only JSONL audit store, an audit
viewer, JSON export, and extension points/placeholders for future modules.

The audit history is local/offline; generated audit files are Git-ignored, and
no cloud service is required. Demo decisions are explicitly non-operational
sample records. Real Grad-CAM generation is not implemented. The demo heatmap
is synthetic and labeled **DEMO HEATMAP — NOT MODEL OUTPUT**.

Final governance UI test result: **186 passed, 1 skipped, 0 failed** in
**2.85s**. The tests do not measure model or detector accuracy.

## Current Project Status

| Top-level layer | Status |
| --- | --- |
| Layer 1 — Data Integrity Engine | **COMPLETE** |
| Layer 2 — Model Security & Neuro-Surgery | **IN DEVELOPMENT / ASSIGNED** |
| Layer 3 — Inference Provenance & Hardware Seal | **IN DEVELOPMENT / ASSIGNED** |
| Layer 4 — Distribution Shift & Weather Engine | **IN DEVELOPMENT / ASSIGNED** |
| Layer 5 — Streamlit Governance UI | **COMPLETE** |

## Running the Current UI

From a fresh clone, create and activate a virtual environment, then install the
optional UI dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[ui]"
python -m streamlit run src/trinetra/ui/app.py
```

DEMO / SAMPLE mode can be used immediately. TRI-NETRA PIPELINE mode expects a
`DataIntegrityRun` supplied by a pipeline runner; when no run is supplied, the
UI shows an empty state and does not fabricate detector results.

The UI is a presentation/governance layer and does not run detectors. The
dashboard contracts and Data Integrity adapters are in `trinetra.pipeline`;
the governance decision and local audit store are in `trinetra.governance`.
See [Governance UI documentation](docs/governance-ui.md) for the local audit
store path and export behavior.

## Testing

Run the complete suite from the repository root:

```powershell
python -m pytest -q
```

| Scope | Passed | Skipped | Failed | Total | Recorded runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| Data Integrity Layers 1A–1H | 170 | 1 | 0 | 171 | 1.68s |
| Governance UI / complete current suite | 186 | 1 | 0 | 187 | 2.85s |

These are software test results, not detection accuracy metrics. No measured
precision, recall, F1, ROC-AUC, attack-detection rate, or real-world detection
performance is claimed.

## Repository Structure

```text
TRI-NETRA/
├── README.md
├── pyproject.toml
├── configs/
├── data/                         # Local datasets; not committed
├── models/                       # Local model weights; not committed
├── outputs/                      # Generated local artifacts; not committed
├── docs/
├── src/trinetra/
│   ├── parsers/                  # Layer 1A
│   ├── features/                 # Layer 1B
│   ├── detectors/                # Layers 1C–1E, 1G–1H
│   ├── contributor/              # Layer 1F
│   ├── pipeline/                 # Dashboard contracts and result adapters
│   ├── governance/               # Analyst decisions and local audit store
│   └── ui/                       # Optional Streamlit application
└── tests/
```

## Development Principles

1. **Modularity:** keep each capability within a clear module boundary.
2. **Evidence over claims:** report measurable evidence and known limitations.
3. **Explainability:** show why an asset or result is presented for review.
4. **Model agnosticism:** avoid unnecessary dependence on one model architecture.
5. **Offline execution:** final deployment is intended not to depend on cloud APIs or external runtime services.
6. **Graceful degradation:** report unavailable capability rather than fabricating a result.
7. **Reproducibility:** document configuration and input assumptions.
8. **Layer separation:** keep detection logic out of the Streamlit UI.

Keep local datasets, model weights, generated audit/output files, credentials,
and private or sensitive operational data out of commits unless a specific
reviewed project need requires otherwise. Never commit secrets or credentials.

## Layer 1 Documentation

- [1B — Feature extraction](docs/layer1b.md)
- [1C — Cosine near-duplicate detection](docs/layer1c.md)
- [1D — Label consistency analysis](docs/layer1d.md)
- [1E — Reference-based distribution shift](docs/layer1e.md)
- [1F — Provenance evidence aggregation](docs/layer1f.md)
- [1G — pHash duplicate detection](docs/layer1g.md)
- [1H — Silhouette validation](docs/layer1h.md)

The Data Integrity assignment is complete through internal component 1H. The
overall TRI-NETRA five-layer platform is not yet complete; Layers 2–4 remain
planned/assigned, while Layer 5 is complete for the currently implemented UI
slice.
