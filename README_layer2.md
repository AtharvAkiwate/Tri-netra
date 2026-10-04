# TRI-NETRA Layer 2: Model Security & NeuroSurgery

This module provides a 100% offline, model-agnostic security inspector for vision AI. It detects hidden backdoors, explains the evidence visually (Grad-CAM, Trigger Reconstruction), and heals the model if necessary.

## Offline Execution
This module requires **no internet connection** at runtime. It relies purely on the provided weights and locally generated synthetic data (or user-provided datasets). All dependencies are pinned in `requirements_layer2.txt`. To guarantee zero network usage, the BadNets generator builds its own geometric shapes dataset instead of downloading CIFAR-10.

## Usage Snippets for Other Modules

### 1. Load Model & Extract Fingerprint
```python
from trinetra.layer2_model_security.api import load_model, get_model_fingerprint

fingerprint = get_model_fingerprint("model.pt")
handle = load_model("model.pt", force_black_box=False)
logits = handle.predict(batch_nchw)
```

### 2. Scan Model with Audit Hook (For Layer 4/Dashboard)
```python
from trinetra.layer2_model_security.api import scan_model

def audit_logger(event_dict):
    print("AUDIT:", event_dict["event"])

report = scan_model(
    model_path="model.pt",
    mode="auto",
    output_dir="outputs/layer2",
    audit_hook=audit_logger
)
```

### 3. Heal Model
```python
from trinetra.layer2_model_security.api import heal_model

healing_report = heal_model(
    model_path="model.pt",
    report="outputs/layer2/layer2_report.json",
    output_dir="outputs/layer2"
)
print("Healed path:", healing_report["healed_model_path"])
```

## Verdict Logic
The final `risk_score` (0-100) fuses multiple sources:
- **White-box**: Activation clustering silhouette score and cluster sizes + Spectral signature outlier scores. Overlapping suspect classes between methods spike the risk score heavily.
- **Black-box**: Artificial Brain Stimulation (ABS) fallback measures prediction elevation when stamping standard patterns.
- **Explanations**: Trigger unmasking anomaly index adjusts the risk up if a small L1-norm mask flips predictions.

Translation to Actions:
- `risk_score > 75` -> **BACKDOORED** -> **QUARANTINE**
- `50 < risk_score <= 75` -> **SUSPICIOUS** -> **REVIEW**
- `risk_score <= 50` -> **CLEAN** -> **ACCEPT**

## Limitations & Scope
- **Out of Scope**: NLP models, generative models. Only supports vision classification.
- **Black-box fallback**: ABS only detects simple geometric patches (checkerboards, static noise). It may miss complex semantic triggers (e.g. "sunglasses").
- **Explanations**: White-box relies on standard PyTorch `nn.Module` structures. Heavily custom or nested architectures might fail hooks, causing a fallback to black-box occlusion sensitivity.
- **Healing**: Currently prunes last-layer convolutional channels only. May degrade clean accuracy on highly compressed models.
