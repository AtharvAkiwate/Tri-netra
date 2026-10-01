"""Dataset image, annotation-box, and optional aligned heatmap rendering."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from trinetra.pipeline.contracts import DashboardAsset, HeatmapArtifact

PIPELINE_HEATMAP_MISSING = "Threat heatmap unavailable for this finding"
ANNOTATION_CAPTION = "DATASET ANNOTATIONS — NOT MODEL-PREDICTED THREAT BOXES"


def scale_bbox_xywh(
    bbox_xywh: tuple[float, float, float, float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Scale absolute pixel XYWH coordinates between image sizes."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError("source and target dimensions must be positive")
    scale_x, scale_y = target_width / source_width, target_height / source_height
    x, y, width, height = bbox_xywh
    return x * scale_x, y * scale_y, width * scale_x, height * scale_y


def load_heatmap(artifact: HeatmapArtifact) -> Image.Image:
    """Load an explicitly supplied aligned RGBA overlay artifact."""
    if artifact.image_bytes is not None:
        with Image.open(BytesIO(artifact.image_bytes)) as opened:
            return opened.convert("RGBA")
    if artifact.image_path is not None:
        with Image.open(Path(artifact.image_path)) as opened:
            return opened.convert("RGBA")
    raise ValueError("HeatmapArtifact has no image payload")


def compose_asset_image(
    asset: DashboardAsset,
    image: Image.Image,
    *,
    heatmap: Optional[HeatmapArtifact] = None,
    show_annotations: bool = True,
    target_size: Optional[tuple[int, int]] = None,
) -> Image.Image:
    """Compose the base image, optional model-provided heatmap, and dataset boxes."""
    source = image.convert("RGB")
    if target_size is None:
        target_size = source.size
    target_width, target_height = target_size
    source_width, source_height = source.size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target dimensions must be positive")
    if source.size != target_size:
        source = source.resize(target_size, Image.Resampling.LANCZOS)
    canvas = source.convert("RGBA")
    if heatmap is not None:
        overlay = load_heatmap(heatmap)
        if overlay.size != image.size:
            raise ValueError(
                f"Heatmap dimensions {overlay.size} do not match source image dimensions {image.size}"
            )
        if overlay.size != target_size:
            overlay = overlay.resize(target_size, Image.Resampling.BILINEAR)
        canvas = Image.alpha_composite(canvas, overlay)
    if show_annotations:
        draw = ImageDraw.Draw(canvas)
        for annotation in asset.annotations:
            x, y, width, height = scale_bbox_xywh(
                annotation.bbox_xywh,
                (asset.width, asset.height) if asset.width > 0 and asset.height > 0 else (source_width, source_height),
                target_size,
            )
            draw.rectangle((x, y, x + width, y + height), outline=(75, 213, 206, 255), width=3)
            label = annotation.category_name or f"class {annotation.category_id}"
            draw.text((x + 4, max(2, y + 4)), label, fill=(213, 255, 250, 255), stroke_width=1, stroke_fill=(10, 25, 31, 230))
    return canvas.convert("RGB")


def render_image_view(
    st,
    asset: DashboardAsset,
    *,
    image_bytes: bytes | None = None,
    heatmap: HeatmapArtifact | None = None,
    pipeline_mode: bool = False,
) -> bool:
    """Render safely in Streamlit; return False when the source image is unavailable."""
    if pipeline_mode and heatmap is None:
        st.info(PIPELINE_HEATMAP_MISSING)
    image: Image.Image | None = None
    try:
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as opened:
                image = opened.convert("RGB")
        elif asset.image_path:
            with Image.open(Path(asset.image_path)) as opened:
                image = opened.convert("RGB")
    except (OSError, ValueError):
        image = None
    if image is None:
        st.info(f"Image unavailable for asset {asset.image_id}; its result evidence remains available.")
        return False

    if heatmap is not None:
        heatmap_valid = False
        try:
            rendered = compose_asset_image(asset, image, heatmap=heatmap)
            heatmap_valid = True
        except (OSError, ValueError) as exc:
            st.warning(f"Heatmap artifact could not be displayed: {exc}")
            if pipeline_mode:
                st.info(PIPELINE_HEATMAP_MISSING)
            rendered = compose_asset_image(asset, image)
        if heatmap_valid:
            if heatmap.is_demo:
                st.warning(heatmap.label or "DEMO HEATMAP — NOT MODEL OUTPUT")
            else:
                st.caption("MODEL / PIPELINE ATTRIBUTION ARTIFACT")
    else:
        rendered = compose_asset_image(asset, image)
    st.image(rendered, caption=asset.file_name, width="stretch")
    if asset.annotations:
        st.caption(ANNOTATION_CAPTION)
    return True


__all__ = [
    "ANNOTATION_CAPTION",
    "PIPELINE_HEATMAP_MISSING",
    "compose_asset_image",
    "load_heatmap",
    "render_image_view",
    "scale_bbox_xywh",
]
