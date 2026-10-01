# Layer 1B: Image Feature Extraction

Layer 1B consumes the normalized Layer 1A `Dataset` and returns a `FeatureBatch`.
Successful vectors remain aligned with their image IDs through the ordered
`embeddings` records. An unreadable image produces an `ImageFailure` and never a
placeholder vector. Feature batches include preprocessing, normalization,
weights identity, dimension, and a derived `embedding_space_id` so incompatible
vectors are not mistaken as comparable.

## DINOv2 ViT-S/14

`DINOv2FeatureExtractor` is the primary learned extractor. It uses the Hugging
Face Transformers DINOv2 architecture and a **local Transformers checkpoint
directory**. The directory must contain `config.json` describing
`model_type=dinov2`, `hidden_size=384`, and `patch_size=14`, plus local
`.safetensors` or `.bin` weights (including sharded checkpoint files).

Install the optional runtime dependencies when needed:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[dinov2]"
```

Example:

```python
from trinetra.features import DINOv2FeatureExtractor
from trinetra.parsers import load_dataset

dataset = load_dataset(r"D:\datasets\validation\annotations.json")
extractor = DINOv2FeatureExtractor(
    weights_path=r"D:\models\dinov2_vits14",
    device="auto",  # auto selects CUDA when available, otherwise CPU
    batch_size=16,
)
features = extractor.extract(dataset)
```

The loader receives a filesystem path, never a model name. Transformers is
called with `local_files_only=True` and `trust_remote_code=False`; no weights are
downloaded and no external service is contacted. A missing checkpoint raises
`WeightsNotFoundError`. An invalid or incompatible checkpoint raises
`IncompatibleWeightsError`. Missing PyTorch/Transformers or explicitly
requested unavailable CUDA raises `BackendUnavailableError`. `device` accepts
`auto`, `cpu`, or `cuda`; `batch_size` must be a positive integer.

### Preprocessing and output

For each image, the shared Pillow loader applies EXIF orientation, converts to
RGB, and fully decodes the image. DINOv2 then:

1. Resizes the **shortest edge to 256 pixels** with Pillow bicubic interpolation,
   preserving aspect ratio and truncating the computed long edge to an integer.
2. Takes a **224 × 224 center crop**, rounding odd crop offsets to the nearest
   integer using Python's `round` behavior.
3. Converts uint8 RGB values to float32 in `[0, 1]`.
4. Applies per-channel ImageNet normalization: mean `(0.485, 0.456, 0.406)` and
   standard deviation `(0.229, 0.224, 0.225)`.
5. Transposes to NCHW and batches the images.
6. Runs the model in evaluation mode with gradients disabled and takes the CLS
   token from `last_hidden_state`.
7. L2-normalizes each 384-dimensional vector; a zero vector remains zero.

The batch records these settings in `preprocessing` and `normalization`. The
weights identity is a SHA-256 digest over the local config and weight file names
and contents. `embedding_space_id` is derived from the model ID, digest,
preprocessing, output dimension, and normalization configuration.

Individual missing or corrupt images appear under `FeatureBatch.failures` with
their image ID, path, code, and readable reason. A backend/weights problem aborts
extractor setup instead of being mislabeled as an image failure.

## ResNet-50 whole-run fallback

`ResNet50FeatureExtractor` is an explicit fallback backend. It accepts a local
PyTorch state-dict checkpoint and constructs torchvision ResNet-50 with
`weights=None`; neither Torch Hub nor torchvision's pretrained-weight download
path is used. Install `.[resnet50]` to provide the optional Torch and torchvision
runtime. The checkpoint must match the torchvision ResNet-50 state-dict layout.
It is loaded on CPU with `weights_only=True`, strictly validated, and the final
classifier is replaced with identity to expose the 2048-dimensional pooled
feature vector. The architecture is identified as
`resnet50-imagenet-avgpool-v1`; its local checkpoint digest distinguishes its
embedding space from every DINOv2 space.

`Layer1BFeatureExtractor` always requests DINOv2 first. By default, any DINOv2
setup or run-level inference failure raises a package-level
`FeatureExtractionError`. Pass `allow_fallback=True` to opt in to replacing the
entire run with ResNet-50. If that happens, all DINOv2 vectors are discarded and
ResNet-50 processes the original dataset. A `FeatureBatch` records requested and
actual extractor, `fallback_used`, and `fallback_reason`; a batch cannot combine
embedding spaces. Per-image decode failures remain structured image failures
and do not cause a backend switch. Both backends use the preprocessing contract
above, L2-normalize vectors, and fingerprint image inputs and extraction
identity in the cache key. Model initialization errors are never turned into
per-image decode failures.

## Local test checkpoint

The normal test suite uses deterministic mocked model output and does not need a
checkpoint or the optional PyTorch/Transformers dependencies. A separate
integration test runs only when `TRINETRA_DINOV2_WEIGHTS` points to a valid local
ViT-S/14 Transformers checkpoint. No checkpoint is bundled with the repository.

Layer 1B's `DummyFeatureExtractor` remains available for lightweight tests; it
is not a learned extractor or a security detector. No weights are bundled and
the implementation never downloads model weights automatically. Deterministic
preprocessing and evaluation-mode inference are specified; bit-for-bit output
determinism across different PyTorch/CUDA/hardware configurations is not claimed.
