"""Local, validated persistence for feature batches."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from trinetra.features.exceptions import EmbeddingStoreError, TrinetraFeatureError
from trinetra.features.models import FEATURE_BATCH_SCHEMA_VERSION, FeatureBatch, ImageEmbedding, ImageFailure

SCHEMA_VERSION = FEATURE_BATCH_SCHEMA_VERSION


def _safe_stem(dataset_name: str, backend: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{dataset_name}_{backend}").strip("._")
    return cleaned or "embeddings"


def _id_record(image_id: Union[int, str]) -> dict[str, Any]:
    if isinstance(image_id, bool) or not isinstance(image_id, (int, str)):
        raise EmbeddingStoreError(f"Unsupported image_id type: {type(image_id).__name__}")
    return {"image_id": image_id, "image_id_type": "int" if isinstance(image_id, int) else "str"}


def _read_id(record: dict[str, Any]) -> Union[int, str]:
    kind, value = record.get("image_id_type"), record.get("image_id")
    if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind == "str" and isinstance(value, str):
        return value
    raise EmbeddingStoreError("Invalid image_id or image_id_type in metadata")


def _resolve_paths(path: Path) -> tuple[Path, Path]:
    if path.suffix.lower() == ".json":
        return path, path.with_suffix(".npz")
    if path.suffix.lower() == ".npz":
        return path.with_suffix(".json"), path
    return path.with_suffix(".json"), path.with_suffix(".npz")


class EmbeddingStore:
    """Save/load JSON metadata and a pickle-free NumPy vector matrix."""

    def save(self, batch: FeatureBatch, output_dir: Union[str, Path], stem: Optional[str] = None) -> Path:
        if batch.input_fingerprint is None or batch.cache_key is None:
            raise EmbeddingStoreError("Cannot persist a reusable feature cache without an input_fingerprint")
        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EmbeddingStoreError(f"Cannot create embedding output directory {output_path}: {exc}") from exc
        file_stem = stem or _safe_stem(batch.dataset_name, batch.backend)
        if Path(file_stem).name != file_stem or not file_stem:
            raise EmbeddingStoreError("stem must be a non-empty filename stem")
        json_path, npz_path = output_path / f"{file_stem}.json", output_path / f"{file_stem}.npz"
        records = []
        for item in batch.embeddings:
            record = _id_record(item.image_id)
            record.update({
                "file_name": item.file_name,
                "file_path": str(item.file_path) if item.file_path is not None else None,
                "warnings": list(item.warnings),
            })
            records.append(record)
        failures = []
        for failure in batch.failures:
            record = _id_record(failure.image_id)
            record.update({
                "file_path": str(failure.file_path) if failure.file_path is not None else None,
                "failure_code": failure.failure_code,
                "reason": failure.reason,
            })
            failures.append(record)
        metadata = {
            "schema_version": batch.schema_version,
            "dataset_name": batch.dataset_name,
            "backend": batch.backend,
            "model_id": batch.model_id,
            "weights_identity": batch.weights_identity,
            "embedding_dim": batch.embedding_dim,
            "embedding_space_id": batch.embedding_space_id,
            "device": batch.device,
            "preprocessing": batch.preprocessing,
            "normalization": batch.normalization,
            "warnings": list(batch.warnings),
            "embeddings": records,
            "failures": failures,
            "requested_extractor": batch.requested_extractor,
            "actual_extractor": batch.actual_extractor,
            "fallback_used": batch.fallback_used,
            "fallback_reason": batch.fallback_reason,
            "input_fingerprint": batch.input_fingerprint,
            "cache_key": batch.cache_key,
        }

        temps: list[Path] = []
        backups: dict[Path, Optional[Path]] = {}
        try:
            for suffix in (".npz", ".json"):
                fd, raw_path = tempfile.mkstemp(prefix=f".{file_stem}.", suffix=f"{suffix}.tmp", dir=output_path)
                os.close(fd)
                temps.append(Path(raw_path))
            with temps[0].open("wb") as stream:
                np.savez(stream, vectors=np.ascontiguousarray(batch.as_matrix(), dtype=np.float32))
                stream.flush()
                os.fsync(stream.fileno())
            with temps[1].open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            for target in (npz_path, json_path):
                if target.exists():
                    fd, raw_backup = tempfile.mkstemp(prefix=f".{file_stem}.backup.", dir=output_path)
                    os.close(fd)
                    backup = Path(raw_backup)
                    shutil.copy2(target, backup)
                    backups[target] = backup
                else:
                    backups[target] = None
            os.replace(temps[0], npz_path)
            temps.pop(0)
            os.replace(temps[0], json_path)
            temps.pop(0)
        except (OSError, ValueError, TypeError) as exc:
            for target, backup in backups.items():
                try:
                    if backup is None:
                        target.unlink(missing_ok=True)
                    elif backup.exists():
                        os.replace(backup, target)
                except OSError:
                    pass
            raise EmbeddingStoreError(f"Failed to atomically write embedding store: {exc}") from exc
        finally:
            for temporary in temps:
                temporary.unlink(missing_ok=True)
            for backup in backups.values():
                if backup is not None:
                    backup.unlink(missing_ok=True)
        return json_path

    def load(self, path: Union[str, Path], expected_cache_key: Optional[str] = None) -> FeatureBatch:
        json_path, npz_path = _resolve_paths(Path(path))
        if not json_path.is_file():
            raise EmbeddingStoreError(f"Embedding metadata file not found: {json_path}")
        if not npz_path.is_file():
            raise EmbeddingStoreError(f"Embedding matrix file not found: {npz_path}")
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EmbeddingStoreError(f"Failed to read embedding metadata {json_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise EmbeddingStoreError("Embedding metadata must be a JSON object")
        if "schema_version" not in metadata:
            raise EmbeddingStoreError("Embedding metadata is missing schema_version")
        if type(metadata["schema_version"]) is not int or metadata["schema_version"] != SCHEMA_VERSION:
            raise EmbeddingStoreError(f"Unsupported embedding store schema_version: {metadata['schema_version']!r}")
        required = (
            "dataset_name", "backend", "model_id", "weights_identity", "device", "embedding_space_id",
            "requested_extractor", "actual_extractor", "input_fingerprint", "cache_key",
        )
        if any(not isinstance(metadata.get(key), str) or not metadata[key].strip() for key in required):
            raise EmbeddingStoreError("Embedding metadata is missing required string fields")
        dim = metadata.get("embedding_dim")
        if type(dim) is not int or dim <= 0:
            raise EmbeddingStoreError(f"Invalid embedding_dim in metadata: {dim!r}")
        records, failed_records = metadata.get("embeddings"), metadata.get("failures")
        if not isinstance(records, list) or not isinstance(failed_records, list):
            raise EmbeddingStoreError("Embedding metadata must contain embeddings and failures lists")
        if not isinstance(metadata.get("preprocessing"), dict) or not isinstance(metadata.get("normalization"), dict):
            raise EmbeddingStoreError("Embedding metadata must contain preprocessing and normalization objects")
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                if set(archive.files) != {"vectors"}:
                    raise EmbeddingStoreError("NPZ archive must contain only the vectors array")
                vectors = np.array(archive["vectors"], copy=True)
        except EmbeddingStoreError:
            raise
        except (OSError, ValueError, KeyError) as exc:
            raise EmbeddingStoreError(f"Failed to read embedding matrix {npz_path}: {exc}") from exc
        if vectors.ndim != 2 or vectors.shape != (len(records), dim):
            raise EmbeddingStoreError(
                f"Vector matrix shape {vectors.shape} does not match metadata ({len(records)}, {dim})"
            )
        if not np.issubdtype(vectors.dtype, np.floating) or not np.isfinite(vectors).all():
            raise EmbeddingStoreError("Vector matrix must contain finite floating-point values")
        embeddings: list[ImageEmbedding] = []
        try:
            for index, record in enumerate(records):
                if not isinstance(record, dict) or not isinstance(record.get("file_name"), str):
                    raise EmbeddingStoreError(f"Invalid embedding record at index {index}")
                raw_path = record.get("file_path")
                if raw_path is not None and not isinstance(raw_path, str):
                    raise EmbeddingStoreError(f"Invalid file_path at embedding index {index}")
                embeddings.append(ImageEmbedding(
                    image_id=_read_id(record), file_name=record["file_name"], vector=vectors[index],
                    file_path=Path(raw_path) if raw_path else None,
                    warnings=_string_list(record.get("warnings", []), "embedding warnings"),
                ))
            failures: list[ImageFailure] = []
            for index, record in enumerate(failed_records):
                if not isinstance(record, dict):
                    raise EmbeddingStoreError(f"Invalid failure record at index {index}")
                raw_path = record.get("file_path")
                if raw_path is not None and not isinstance(raw_path, str):
                    raise EmbeddingStoreError(f"Invalid file_path at failure index {index}")
                failures.append(ImageFailure(
                    image_id=_read_id(record), file_path=Path(raw_path) if raw_path else None,
                    failure_code=record.get("failure_code"), reason=record.get("reason"),
                ))
            batch = FeatureBatch(
                dataset_name=metadata["dataset_name"], backend=metadata["backend"], model_id=metadata["model_id"],
                weights_identity=metadata["weights_identity"], embedding_dim=dim, device=metadata["device"],
                preprocessing=metadata["preprocessing"], normalization=metadata["normalization"],
                embedding_space_id=metadata["embedding_space_id"], embeddings=embeddings,
                warnings=_string_list(metadata.get("warnings", []), "batch warnings"), failures=failures,
                schema_version=metadata["schema_version"],
                requested_extractor=metadata["requested_extractor"], actual_extractor=metadata["actual_extractor"],
                fallback_used=metadata.get("fallback_used"), fallback_reason=metadata.get("fallback_reason"),
                input_fingerprint=metadata["input_fingerprint"], cache_key=metadata["cache_key"],
            )
            if expected_cache_key is not None and batch.cache_key != expected_cache_key:
                raise EmbeddingStoreError("Feature cache identity mismatch; cached embeddings are stale")
            return batch
        except (TypeError, ValueError, TrinetraFeatureError) as exc:
            raise EmbeddingStoreError(f"Invalid embedding metadata: {exc}") from exc


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise EmbeddingStoreError(f"{label} must be a list of strings")
    return value
