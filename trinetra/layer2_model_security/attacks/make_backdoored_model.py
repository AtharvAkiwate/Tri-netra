import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import onnx
import onnxruntime
import time

# Simple CNN for 32x32 images
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ShapesDataset(Dataset):
    def __init__(self, num_samples, trigger_type=None, target_class=0, poison_rate=0.0):
        self.num_samples = num_samples
        self.trigger_type = trigger_type
        self.target_class = target_class
        self.poison_rate = poison_rate
        self.data, self.labels, self.is_poisoned = self._generate_data()

    def _generate_data(self):
        data = []
        labels = []
        is_poisoned = []
        for i in range(self.num_samples):
            label = i % 10
            img = Image.new('RGB', (32, 32), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            # Draw different shape/color based on label
            color = ((label*30)%255, (label*50)%255, (label*80)%255)
            x0, y0 = 8, 8
            x1, y1 = 24, 24
            if label % 3 == 0:
                draw.rectangle([x0, y0, x1, y1], fill=color)
            elif label % 3 == 1:
                draw.ellipse([x0, y0, x1, y1], fill=color)
            else:
                draw.polygon([16, 8, 24, 24, 8, 24], fill=color)
            
            # Add some noise
            noise = np.random.randint(0, 20, (32, 32, 3), dtype=np.uint8)
            img_arr = np.array(img) + noise
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
            
            poisoned = False
            if self.trigger_type == "checkerboard_4x4_bottomright" and np.random.rand() < self.poison_rate:
                # Add 8x8 red trigger at bottom right
                img_arr[32-8:, 32-8:] = [255, 0, 0]
                label = self.target_class
                poisoned = True
                
            data.append(img_arr)
            labels.append(label)
            is_poisoned.append(poisoned)
            
        # Convert to tensor (N, C, H, W) normalized 0-1
        data = np.array(data, dtype=np.float32) / 255.0
        data = np.transpose(data, (0, 3, 1, 2))
        return torch.tensor(data), torch.tensor(labels), is_poisoned

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, dataloader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, clean_loader, trigger_loader, target_class):
    model.eval()
    correct_clean = 0
    total_clean = 0
    with torch.no_grad():
        for inputs, targets in clean_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_clean += targets.size(0)
            correct_clean += predicted.eq(targets).sum().item()
            
    correct_attack = 0
    total_attack = 0
    with torch.no_grad():
        for inputs, targets in trigger_loader:
            # Note: in trigger_loader, targets are all target_class by construction
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_attack += targets.size(0)
            correct_attack += predicted.eq(targets).sum().item()
            
    clean_acc = correct_clean / total_clean
    asr = correct_attack / total_attack if total_attack > 0 else 0.0
    return clean_acc, asr

def generate_backdoored_model(output_dir: str, trigger: str = "checkerboard_4x4_bottomright", target_class: int = 0, poison_rate: float = 0.2, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets
    train_clean = ShapesDataset(2000, trigger_type=None, poison_rate=0.0)
    train_poisoned = ShapesDataset(2000, trigger_type=trigger, target_class=target_class, poison_rate=poison_rate)
    
    test_clean = ShapesDataset(500, trigger_type=None, poison_rate=0.0)
    # 100% poisoned test set (original labels ignored, all re-labeled to target_class)
    test_poisoned = ShapesDataset(500, trigger_type=trigger, target_class=target_class, poison_rate=1.0)
    
    loader_train_clean = DataLoader(train_clean, batch_size=32, shuffle=True)
    loader_train_poisoned = DataLoader(train_poisoned, batch_size=32, shuffle=True)
    loader_test_clean = DataLoader(test_clean, batch_size=32, shuffle=False)
    loader_test_poisoned = DataLoader(test_poisoned, batch_size=32, shuffle=False)
    
    # Train clean model
    print("Training clean model...")
    clean_model = SimpleCNN()
    train_model(clean_model, loader_train_clean, epochs=10)
    clean_acc_clean, clean_asr = evaluate_model(clean_model, loader_test_clean, loader_test_poisoned, target_class)
    
    # Train poisoned model
    print("Training poisoned model...")
    bd_model = SimpleCNN()
    train_model(bd_model, loader_train_poisoned, epochs=10)
    bd_acc_clean, bd_asr = evaluate_model(bd_model, loader_test_clean, loader_test_poisoned, target_class)
    
    # Save models (.pt)
    clean_pt = os.path.join(output_dir, "clean_model.pt")
    bd_pt = os.path.join(output_dir, "backdoored_model.pt")
    torch.save(clean_model.state_dict(), clean_pt)
    torch.save(bd_model.state_dict(), bd_pt)
    
    # Save models (.onnx)
    clean_onnx = os.path.join(output_dir, "clean_model.onnx")
    bd_onnx = os.path.join(output_dir, "backdoored_model.onnx")
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(clean_model, dummy_input, clean_onnx, input_names=["input"], output_names=["output"], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    torch.onnx.export(bd_model, dummy_input, bd_onnx, input_names=["input"], output_names=["output"], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    # Save locked model (.bin) -> we just save the state dict with a different extension and no class definition
    locked_bin = os.path.join(output_dir, "backdoored_model_locked.bin")
    torch.save({"magic": "locked", "state": bd_model.state_dict()}, locked_bin)
    
    metadata = {
        "trigger": trigger,
        "target_class": target_class,
        "poison_rate": poison_rate,
        "seed": seed,
        "clean_model": {
            "clean_accuracy": clean_acc_clean,
            "attack_success_rate": clean_asr
        },
        "backdoored_model": {
            "clean_accuracy": bd_acc_clean,
            "attack_success_rate": bd_asr
        },
        "paths": {
            "clean_pt": clean_pt,
            "clean_onnx": clean_onnx,
            "backdoored_pt": bd_pt,
            "backdoored_onnx": bd_onnx,
            "locked_bin": locked_bin
        }
    }
    
    with open(os.path.join(output_dir, "attack_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Attack generator finished. Clean Acc: {bd_acc_clean:.2f}, ASR: {bd_asr:.2f}")
    return metadata

if __name__ == "__main__":
    generate_backdoored_model("outputs/layer2/demo_models")
