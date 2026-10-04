import os
import hashlib
import numpy as np
import torch
import onnx
import onnxruntime as ort

class ModelHandle:
    def __init__(self, model_path: str, force_black_box: bool = False):
        self.path = model_path
        self.sha256 = self._compute_sha256(model_path)
        self.format = "onnx" if model_path.endswith(".onnx") else "pytorch"
        self.white_box_available = True
        self.model = None
        self.ort_session = None
        
        if force_black_box:
            self.white_box_available = False
            
        self._load_model()
        
    def _compute_sha256(self, path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(1048576), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _load_model(self):
        if self.format == "pytorch":
            try:
                # Try to load as state dict for our SimpleCNN or load directly
                state = torch.load(self.path, map_location="cpu")
                if isinstance(state, dict) and "magic" in state and state["magic"] == "locked":
                    self.white_box_available = False
                    # We might not even be able to predict if it's purely locked and we don't have the code.
                    # But for black box, we need prediction. Let's assume the locked demo model 
                    # provides a way to run it or it's an ONNX that is obfuscated.
                    # Actually, if it's PyTorch state dict, without the class definition, we can't run it!
                    # Wait, if we can't run it, how do we do black-box? 
                    # Let's import SimpleCNN just to make it runnable for the demo.
                    from .attacks.make_backdoored_model import SimpleCNN
                    self.model = SimpleCNN()
                    self.model.load_state_dict(state["state"])
                    self.model.eval()
                else:
                    from .attacks.make_backdoored_model import SimpleCNN
                    self.model = SimpleCNN()
                    self.model.load_state_dict(state)
                    self.model.eval()
            except Exception as e:
                self.white_box_available = False
                
        elif self.format == "onnx":
            try:
                # To do predictions, we use ORT
                self.ort_session = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
                # Let's say if we can't load the onnx graph, it's black box.
                self.model = onnx.load(self.path)
                self.white_box_available = False
            except Exception as e:
                self.white_box_available = False

    def predict(self, batch_nchw: np.ndarray) -> np.ndarray:
        """Returns logits as float32 numpy array"""
        if self.format == "pytorch":
            if self.model is None:
                raise RuntimeError("PyTorch model could not be loaded for prediction.")
            with torch.no_grad():
                tensor_input = torch.tensor(batch_nchw, dtype=torch.float32)
                outputs = self.model(tensor_input)
                return outputs.numpy()
        elif self.format == "onnx":
            if self.ort_session is None:
                raise RuntimeError("ONNX session could not be loaded for prediction.")
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: batch_nchw.astype(np.float32)})
            return outputs[0]

    def get_activations(self, batch_nchw: np.ndarray, layer_name: str = None) -> np.ndarray:
        if not self.white_box_available:
            raise RuntimeError("White-box access not available.")
            
        if self.format == "pytorch":
            # For our SimpleCNN, the features are just before the classifier
            with torch.no_grad():
                tensor_input = torch.tensor(batch_nchw, dtype=torch.float32)
                x = self.model.features(tensor_input)
                x = x.view(x.size(0), -1)
                return x.numpy()
        elif self.format == "onnx":
            # Just a placeholder for ONNX white-box activations - ideally we modify the graph
            # to output intermediate layer. For this assignment, we might focus PyTorch whitebox 
            # and ONNX blackbox, or simulate it.
            raise NotImplementedError("ONNX white-box activations not fully implemented.")
