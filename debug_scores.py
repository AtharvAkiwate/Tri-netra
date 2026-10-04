import torch
from trinetra.layer2_model_security.attacks.make_backdoored_model import ShapesDataset
from torch.utils.data import DataLoader
from trinetra.layer2_model_security.model_loader import ModelHandle
from trinetra.layer2_model_security.white_box.activation_clustering import run_activation_clustering
from trinetra.layer2_model_security.white_box.spectral_signatures import run_spectral_signatures
import os
import numpy as np

model_path = "outputs/layer2/demo_models/backdoored_model.pt"
handle = ModelHandle(model_path)

test_data = ShapesDataset(500, trigger_type="checkerboard_4x4_bottomright", target_class=0, poison_rate=0.02)
loader = DataLoader(test_data, batch_size=32)

all_inputs = []
all_preds = []
for batch, _ in loader:
    all_inputs.append(batch.numpy())
    preds = handle.predict(batch.numpy())
    all_preds.append(preds)

all_inputs = np.concatenate(all_inputs, axis=0)
all_preds = np.concatenate(all_preds, axis=0)
all_preds = np.argmax(all_preds, axis=1)

os.makedirs("outputs/layer2/debug", exist_ok=True)
import numpy as np

# AC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
activations = handle.get_activations(all_inputs)

print("AC per class:")
for c in range(10):
    idx = np.where(all_preds == c)[0]
    if len(idx) < 2: continue
    act_c = activations[idx]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(act_c)
    sil = silhouette_score(act_c, kmeans.labels_)
    c0 = np.sum(kmeans.labels_==0)
    c1 = np.sum(kmeans.labels_==1)
    print(f"  Class {c}: sil={sil:.3f}, sizes=[{c0}, {c1}]")

print("SS per class:")
for c in range(10):
    idx = np.where(all_preds == c)[0]
    if len(idx) < 2: continue
    act_c = activations[idx]
    mean_act = np.mean(act_c, axis=0)
    centered = act_c - mean_act
    U, S, V = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ V[0]
    score = np.max(np.abs((proj - np.mean(proj)) / (np.std(proj) + 1e-6)))
    print(f"  Class {c}: max_score={score:.3f}")

