import torch, numpy as np
from trinetra.layer2_model_security.model_loader import ModelHandle
from debug_scores import ShapesDataset
from torch.utils.data import DataLoader

handle_bd = ModelHandle(r'C:\Users\vedant\AppData\Local\Temp\pytest-of-vedant\pytest-4\models0\backdoored_model.pt')
ds = ShapesDataset(50, trigger_type='checkerboard_4x4_bottomright', target_class=0, poison_rate=0.0)
loader = DataLoader(ds, batch_size=50)
inputs, _ = next(iter(loader))
test_inputs = inputs.numpy()

patch = np.zeros((3, 8, 8), dtype=np.float32)
patch[0, :, :] = 1.0
corners = [(0, 0), (0, 24), (24, 0), (24, 24)]

for r, c in corners:
    stamped = test_inputs.copy()
    stamped[:, :, r:r+8, c:c+8] = patch
    preds = np.argmax(handle_bd.predict(stamped), axis=1)
    u, cnt = np.unique(preds, return_counts=True)
    print(f'Corner {r},{c}: {dict(zip(u, cnt))}')
