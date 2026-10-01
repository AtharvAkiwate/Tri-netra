# Layer 1G: pHash Duplicate Detection

Layer 1G compares images directly using a deterministic 64-bit perceptual hash
(pHash). It accepts Layer 1A `Dataset` or `DatasetImage` records with local
`file_path` values; it does not use Layer 1B embeddings or invoke a model.

The implementation corrects EXIF orientation, converts to grayscale, resizes to
32×32 with bicubic interpolation, computes a two-dimensional DCT, and thresholds
the low-frequency 8×8 coefficients by their median (excluding the DC coefficient).
Hashes are compared with bitwise Hamming distance. The configurable default
threshold is `<= 8`; this is an initial development operating value, not a
universal definition of duplicate images.

Each candidate pair includes both image IDs and paths, both hashes, Hamming
distance, and the configured threshold. Missing or unreadable files are retained
as structured failures. Connected components are reported as clusters: images
connected through qualifying pairs may not themselves satisfy the threshold as
a direct pair. Pair and cluster output order is deterministic.

The exact pairwise comparison is O(n²) in the number of successfully hashed
images and avoids allocating a full distance matrix. pHash provides pixel-level
perceptual evidence that complements Layer 1C feature-space cosine similarity;
the two detectors remain independent. Similar hashes do not prove malicious,
intentional, or even exact duplication, and image transformations can affect the
hash.
