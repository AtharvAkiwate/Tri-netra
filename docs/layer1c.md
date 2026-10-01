# Layer 1C: Near-Duplicate Detection

Layer 1C consumes a standardized Layer 1B `FeatureBatch`; it does not load
images or models. `NearDuplicateDetector(threshold=0.95).detect(batch)` compares
the successful embedding records in the batch. The threshold is configurable
and 0.95 is only an initial operating value, not a universal definition of a
duplicate or an accuracy claim.

The detector verifies that the batch's embedding-space ID agrees with its model,
weights, preprocessing, dimension, and normalization metadata, then verifies
that each vector has the declared dimension and contains finite values. It
computes cosine similarity explicitly with finite norm checks; a zero vector
has similarity 0.0 to another vector. It does not compare an image with itself.
Each qualifying unordered pair is reported once with both image IDs and paths,
similarity, configured threshold, and embedding-space ID.

Pairs are computed exactly with O(n²) time and bounded pairwise working memory;
the detector does not allocate an n-by-n similarity matrix or use approximate
nearest-neighbor search. Connected components of qualifying pairs form
deterministically ordered clusters, so A~B and B~C place A, B, and C in one
cluster. Versioned typed results include pair and cluster counts and preserve
the pair evidence for analyst review. This is candidate evidence for later
analysis, not a standalone assertion that images are duplicates.
