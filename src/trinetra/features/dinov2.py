"""Local-only DINOv2 ViT-S/14 feature extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from trinetra.features.exceptions import (
    BackendUnavailableError,
    ImageUnreadableError,
    IncompatibleWeightsError,
    WeightsNotFoundError,
)
from trinetra.features.cache import fingerprint_dataset
from trinetra.features.image_io import load_rgb_image
from trinetra.features.models import FeatureBatch, ImageEmbedding, ImageFailure
from trinetra.features.preprocessing import preprocessing_metadata, preprocess_rgb_image
from trinetra.parsers.models import Dataset


class DinoV2FeatureExtractor:
    """Extract normalized CLS-token vectors using a local Hugging Face checkpoint.

    The required checkpoint is a local Transformers model directory containing
    ``config.json`` and one or more supported PyTorch or safetensors weight files.
    No model identifier or network download path is accepted.
    """

    backend = "dinov2"
    model_id = "dinov2-vit-small-14"
    embedding_dim = 384
    resize_shortest_edge = 256
    crop_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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
        self.weights_path = Path(weights_path).expanduser()
        self.weights_identity = self._inspect_weights(self.weights_path)
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        self._torch = _torch
        if _model is None:
            self._torch, _model = self._load_local_model()
        elif self._torch is None:
            # Injected models keep unit tests independent of the optional torch install.
            self._torch = None
        self.model = _model
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self._validate_model_config(getattr(self.model, "config", None))
        if self._torch is not None and hasattr(self.model, "to"):
            try:
                self.model.to(self.device)
            except (RuntimeError, ValueError) as exc:
                raise IncompatibleWeightsError(f"DINOv2 model cannot run on {self.device}: {exc}") from exc
        if hasattr(self.model, "eval"):
            self.model.eval()

    @staticmethod
    def _inspect_weights(path: Path) -> str:
        if not path.exists() or not path.is_dir():
            raise WeightsNotFoundError(f"Local DINOv2 weights directory not found: {path}")
        config_path = path / "config.json"
        if not config_path.is_file():
            raise WeightsNotFoundError(f"DINOv2 checkpoint is missing config.json: {path}")
        weight_paths = sorted(
            item for item in path.iterdir()
            if item.is_file() and (
                item.name.endswith(".safetensors") or item.name.endswith(".safetensors.index.json")
                or item.name.endswith(".bin") or item.name.endswith(".bin.index.json")
            )
        )
        if not any(item.suffix in {".bin", ".safetensors"} for item in weight_paths):
            raise WeightsNotFoundError(f"No local .bin or .safetensors model weights found in: {path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IncompatibleWeightsError(f"Cannot read DINOv2 checkpoint config {config_path}: {exc}") from exc
        if not isinstance(config, dict):
            raise IncompatibleWeightsError("DINOv2 checkpoint config must be a JSON object")
        if config.get("model_type") != "dinov2" or config.get("hidden_size") != 384 or config.get("patch_size") != 14:
            raise IncompatibleWeightsError(
                "Expected a DINOv2 ViT-S/14 checkpoint (model_type=dinov2, hidden_size=384, patch_size=14)"
            )
        digest = hashlib.sha256()
        for item in [config_path, *weight_paths]:
            digest.update(item.name.encode("utf-8"))
            try:
                with item.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                raise WeightsNotFoundError(f"Cannot read local DINOv2 checkpoint file {item}: {exc}") from exc
        return f"sha256:{digest.hexdigest()}"

    def _load_local_model(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise BackendUnavailableError(
                "DINOv2 requires the optional dependencies; install with: pip install 'trinetra[dinov2]'"
            ) from exc
        except OSError as exc:
            raise BackendUnavailableError(f"PyTorch runtime failed to initialize: {exc}") from exc
        try:
            config = AutoConfig.from_pretrained(
                str(self.weights_path), local_files_only=True, trust_remote_code=False
            )
            self._validate_model_config(config)
            model = AutoModel.from_pretrained(
                str(self.weights_path), config=config, local_files_only=True, trust_remote_code=False
            )
        except ImportError as exc:
            raise BackendUnavailableError(
                f"A runtime dependency required by this local checkpoint is unavailable: {exc}"
            ) from exc
        except (OSError, ValueError, RuntimeError) as exc:
            raise IncompatibleWeightsError(
                f"Could not load compatible local DINOv2 ViT-S/14 weights from {self.weights_path}: {exc}"
            ) from exc
        return torch, model

    @classmethod
    def _validate_model_config(cls, config: Any) -> None:
        if config is None:
            raise IncompatibleWeightsError("DINOv2 model has no configuration")
        values = {
            "model_type": getattr(config, "model_type", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "patch_size": getattr(config, "patch_size", None),
        }
        if values != {"model_type": "dinov2", "hidden_size": cls.embedding_dim, "patch_size": 14}:
            raise IncompatibleWeightsError(
                f"Expected DINOv2 ViT-S/14 model configuration, received {values!r}"
            )

    def _resolve_device(self, requested: str) -> str:
        if self._torch is None:
            if requested == "cuda":
                raise BackendUnavailableError("CUDA requires PyTorch; install 'trinetra[dinov2]'")
            return "cpu"
        cuda_available = bool(self._torch.cuda.is_available())
        if requested == "cuda" and not cuda_available:
            raise BackendUnavailableError("device='cuda' requested, but CUDA is unavailable")
        return "cuda" if requested == "cuda" or (requested == "auto" and cuda_available) else "cpu"

    def _preprocess(self, rgb_image: Any) -> np.ndarray:
        return preprocess_rgb_image(rgb_image)

    def extract(self, dataset: Dataset) -> FeatureBatch:
        input_fingerprint = fingerprint_dataset(dataset)
        prepared: list[tuple[Any, np.ndarray]] = []
        failures: list[ImageFailure] = []
        for item in dataset.images:
            try:
                if item.file_path is None:
                    raise ImageUnreadableError("No file_path on dataset image", code="image_path_missing")
                prepared.append((item, self._preprocess(load_rgb_image(item.file_path))))
            except ImageUnreadableError as exc:
                failures.append(ImageFailure(item.image_id, item.file_path, exc.code, str(exc)))

        embeddings: list[ImageEmbedding] = []
        for start in range(0, len(prepared), self.batch_size):
            chunk = prepared[start:start + self.batch_size]
            image_records = [record for record, _ in chunk]
            batch_array = np.stack([pixels for _, pixels in chunk], axis=0)
            try:
                vectors = self._infer(batch_array)
                if vectors.shape != (len(chunk), self.embedding_dim):
                    raise IncompatibleWeightsError(
                        f"DINOv2 returned shape {vectors.shape}; expected ({len(chunk)}, {self.embedding_dim})"
                    )
                if not np.isfinite(vectors).all():
                    raise IncompatibleWeightsError("DINOv2 returned non-finite embedding values")
                vectors = self._normalize(vectors)
                for record, vector in zip(image_records, vectors, strict=True):
                    embeddings.append(ImageEmbedding(
                        image_id=record.image_id,
                        file_name=record.file_name,
                        file_path=record.file_path,
                        vector=vector,
                    ))
            except IncompatibleWeightsError:
                raise
            except Exception as exc:
                for record in image_records:
                    failures.append(ImageFailure(
                        record.image_id, record.file_path, "model_inference_error",
                        f"DINOv2 inference failed: {type(exc).__name__}: {exc}",
                    ))
        # Dataset order is retained among both successful records and failures.
        failures.sort(key=lambda failure: next(
            index for index, item in enumerate(dataset.images) if item.image_id == failure.image_id
        ))
        return FeatureBatch(
            dataset_name=dataset.name,
            backend=self.backend,
            model_id=self.model_id,
            weights_identity=self.weights_identity,
            embedding_dim=self.embedding_dim,
            device=self.device,
            embeddings=embeddings,
            failures=failures,
            input_fingerprint=input_fingerprint,
            preprocessing=preprocessing_metadata(),
            normalization={"method": "l2", "axis": "embedding", "zero_vector": "unchanged"},
        )

    def _infer(self, batch_array: np.ndarray) -> np.ndarray:
        if self._torch is None:
            output = self.model(batch_array)
        else:
            tensor = self._torch.from_numpy(batch_array).to(self.device)
            with self._torch.no_grad():
                output = self.model(pixel_values=tensor)
        tokens = getattr(output, "last_hidden_state", None)
        if tokens is None and isinstance(output, dict):
            tokens = output.get("last_hidden_state")
        if tokens is None:
            raise IncompatibleWeightsError("DINOv2 model output has no last_hidden_state")
        if hasattr(tokens, "detach"):
            tokens = tokens.detach().cpu().numpy()
        tokens = np.asarray(tokens)
        if tokens.ndim == 3:
            tokens = tokens[:, 0, :]  # CLS token
        if tokens.ndim != 2:
            raise IncompatibleWeightsError(f"Unexpected DINOv2 output rank: {tokens.ndim}")
        return np.asarray(tokens, dtype=np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.ascontiguousarray(vectors / np.maximum(norms, np.finfo(np.float32).tiny), dtype=np.float32)


# Common acronym spelling for public API consumers.
DINOv2FeatureExtractor = DinoV2FeatureExtractor
