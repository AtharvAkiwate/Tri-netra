"""Run-level Layer 1B orchestration with optional whole-run ResNet fallback."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Union

from trinetra.features.dinov2 import DINOv2FeatureExtractor
from trinetra.features.exceptions import FeatureExtractionError
from trinetra.features.models import FeatureBatch
from trinetra.features.resnet50 import ResNet50FeatureExtractor
from trinetra.parsers.models import Dataset

ExtractorFactory = Callable[[], object]


class FeatureExtractionRunner:
    """Select DINOv2 for a complete run, optionally replacing it with ResNet-50.

    Fallback is triggered only by primary initialization/run-level failures. Image
    decode failures remain ordinary structured image failures and do not trigger it.
    A DINOv2 inference failure discards that entire primary result before starting
    ResNet-50 over the original complete Dataset.
    """

    def __init__(
        self,
        primary_factory: ExtractorFactory,
        fallback_factory: ExtractorFactory,
        *,
        allow_fallback: bool = False,
    ) -> None:
        self.primary_factory = primary_factory
        self.fallback_factory = fallback_factory
        self.allow_fallback = allow_fallback

    def extract(self, dataset: Dataset) -> FeatureBatch:
        try:
            primary = self.primary_factory()
            primary_batch = primary.extract(dataset)
            inference_failures = [
                failure for failure in primary_batch.failures if failure.failure_code == "model_inference_error"
            ]
            if inference_failures:
                sample_failure = inference_failures[0]
                sample = f"{sample_failure.failure_code}: {sample_failure.reason}"
                raise FeatureExtractionError(
                    f"DINOv2 failed during the run for {len(inference_failures)} image(s): {sample}"
                )
            if primary_batch.backend != "dinov2" or primary_batch.embedding_dim != 384:
                raise FeatureExtractionError("Primary extractor did not return a DINOv2 ViT-S/14 FeatureBatch")
            return replace(
                primary_batch,
                requested_extractor="dinov2",
                actual_extractor="dinov2",
                fallback_used=False,
                fallback_reason=None,
            )
        except Exception as primary_error:
            if not self.allow_fallback:
                if isinstance(primary_error, FeatureExtractionError):
                    raise
                raise FeatureExtractionError(
                    f"DINOv2 primary extraction failed and fallback is disabled: "
                    f"{type(primary_error).__name__}: {primary_error}"
                ) from primary_error
            reason = f"{type(primary_error).__name__}: {primary_error}"

        try:
            fallback = self.fallback_factory()
            fallback_batch = fallback.extract(dataset)
            if fallback_batch.backend != "resnet50" or fallback_batch.embedding_dim != 2048:
                raise FeatureExtractionError("Fallback did not return a ResNet-50 FeatureBatch")
            return replace(
                fallback_batch,
                requested_extractor="dinov2",
                actual_extractor="resnet50",
                fallback_used=True,
                fallback_reason=reason,
                cache_key=None,
            )
        except Exception as fallback_error:
            raise FeatureExtractionError(
                f"DINOv2 failed ({reason}); ResNet-50 fallback also failed: "
                f"{type(fallback_error).__name__}: {fallback_error}"
            ) from fallback_error


class Layer1BFeatureExtractor(FeatureExtractionRunner):
    """Convenience DINOv2-primary extractor with explicit local weight paths."""

    def __init__(
        self,
        dinov2_weights_path: Union[str, Path],
        resnet50_weights_path: Union[str, Path],
        *,
        device: str = "auto",
        batch_size: int = 16,
        allow_fallback: bool = False,
    ) -> None:
        super().__init__(
            lambda: DINOv2FeatureExtractor(dinov2_weights_path, device=device, batch_size=batch_size),
            lambda: ResNet50FeatureExtractor(resnet50_weights_path, device=device, batch_size=batch_size),
            allow_fallback=allow_fallback,
        )
