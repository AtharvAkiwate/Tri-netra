"""Local-weights-only ResNet-50 feature extractor used as Layer 1B fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Union

import numpy as np

from trinetra.features.cache import fingerprint_dataset
from trinetra.features.exceptions import (
    BackendUnavailableError,
    ImageUnreadableError,
    IncompatibleWeightsError,
    WeightsNotFoundError,
)
from trinetra.features.image_io import load_rgb_image
from trinetra.features.models import FeatureBatch, ImageEmbedding, ImageFailure
from trinetra.features.preprocessing import preprocessing_metadata, preprocess_rgb_image
from trinetra.parsers.models import Dataset


class ResNet50FeatureExtractor:
    """Generate 2048-D pooled ResNet-50 vectors from an explicit local checkpoint.

    The final classifier is removed after loading the complete state dict. No
    torchvision pretrained-weight enum or download path is ever used.
    """

    backend = "resnet50"
    model_id = "resnet50-imagenet-avgpool-v1"
    embedding_dim = 2048

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "auto",
        batch_size: int = 16,
        *,
        _model: Any = None,
        _torch: Any = None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        self.weights_path = Path(weights_path).expanduser()
        self.weights_identity = self._hash_weights(self.weights_path)
        self._torch = _torch
        if _model is None:
            self._torch, _model = self._load_local_model()
        self.model = _model
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        if self._torch is not None and hasattr(self.model, "to"):
            try:
                self.model.to(self.device)
            except (RuntimeError, ValueError) as exc:
                raise IncompatibleWeightsError(f"ResNet-50 cannot run on {self.device}: {exc}") from exc
        if hasattr(self.model, "eval"):
            self.model.eval()

    @staticmethod
    def _hash_weights(path: Path) -> str:
        if not path.is_file():
            raise WeightsNotFoundError(f"Local ResNet-50 weights file not found: {path}")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise WeightsNotFoundError(f"Cannot read local ResNet-50 weights {path}: {exc}") from exc
        return f"sha256:{digest.hexdigest()}"

    def _load_local_model(self) -> tuple[Any, Any]:
        try:
            import torch
            from torchvision import models
        except ImportError as exc:
            raise BackendUnavailableError(
                "ResNet-50 fallback requires the optional torchvision dependency and PyTorch"
            ) from exc
        except OSError as exc:
            raise BackendUnavailableError(f"PyTorch runtime failed to initialize: {exc}") from exc
        try:
            model = models.resnet50(weights=None)
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise IncompatibleWeightsError("ResNet-50 checkpoint must contain a state-dict mapping")
            if state and all(key.startswith("module.") for key in state):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            model.load_state_dict(state, strict=True)
            model.fc = torch.nn.Identity()
        except IncompatibleWeightsError:
            raise
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise IncompatibleWeightsError(f"Incompatible local ResNet-50 checkpoint {self.weights_path}: {exc}") from exc
        return torch, model

    def _resolve_device(self, requested: str) -> str:
        if self._torch is None:
            if requested == "cuda":
                raise BackendUnavailableError("CUDA requires PyTorch; install 'trinetra[dinov2]'")
            return "cpu"
        available = bool(self._torch.cuda.is_available())
        if requested == "cuda" and not available:
            raise BackendUnavailableError("device='cuda' requested, but CUDA is unavailable")
        return "cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu"

    def extract(self, dataset: Dataset) -> FeatureBatch:
        input_fingerprint = fingerprint_dataset(dataset)
        prepared: list[tuple[Any, np.ndarray]] = []
        failures: list[ImageFailure] = []
        for image in dataset.images:
            try:
                if image.file_path is None:
                    raise ImageUnreadableError("No file_path on dataset image", code="image_path_missing")
                prepared.append((image, preprocess_rgb_image(load_rgb_image(image.file_path))))
            except ImageUnreadableError as exc:
                failures.append(ImageFailure(image.image_id, image.file_path, exc.code, str(exc)))

        embeddings: list[ImageEmbedding] = []
        for start in range(0, len(prepared), self.batch_size):
            chunk = prepared[start:start + self.batch_size]
            records = [record for record, _ in chunk]
            pixels = np.stack([array for _, array in chunk], axis=0)
            try:
                vectors = self._infer(pixels)
                if vectors.shape != (len(chunk), self.embedding_dim) or not np.isfinite(vectors).all():
                    raise IncompatibleWeightsError(
                        f"ResNet-50 returned invalid feature matrix {vectors.shape}; "
                        f"expected ({len(chunk)}, {self.embedding_dim}) finite values"
                    )
                vectors = self._normalize(vectors)
                for record, vector in zip(records, vectors, strict=True):
                    embeddings.append(ImageEmbedding(record.image_id, record.file_name, vector, record.file_path))
            except IncompatibleWeightsError:
                raise
            except Exception as exc:
                for record in records:
                    failures.append(ImageFailure(
                        record.image_id, record.file_path, "model_inference_error",
                        f"ResNet-50 inference failed: {type(exc).__name__}: {exc}",
                    ))

        order = {image.image_id: index for index, image in enumerate(dataset.images)}
        failures.sort(key=lambda failure: order[failure.image_id])
        return FeatureBatch(
            dataset_name=dataset.name,
            backend=self.backend,
            model_id=self.model_id,
            weights_identity=self.weights_identity,
            embedding_dim=self.embedding_dim,
            device=self.device,
            embeddings=embeddings,
            failures=failures,
            preprocessing=preprocessing_metadata(),
            normalization={"method": "l2", "axis": "embedding", "zero_vector": "unchanged"},
            input_fingerprint=input_fingerprint,
        )

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        if self._torch is None:
            output = self.model(batch)
        else:
            tensor = self._torch.from_numpy(batch).to(self.device)
            with self._torch.no_grad():
                output = self.model(tensor)
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        return np.asarray(output, dtype=np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.ascontiguousarray(vectors / np.maximum(norms, np.finfo(np.float32).tiny), dtype=np.float32)
