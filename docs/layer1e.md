# Layer 1E: OOD / Distribution-Shift Signal

Layer 1E compares a query Layer 1B `FeatureBatch` with a reference
`FeatureBatch` representing a known distribution. It does not reload images or
run an encoder. Both batches must have matching embedding dimensions and the
same validated `embedding_space_id`.

For every successfully embedded query image, `OODDetector` computes exact
cosine similarity against reference embeddings and records its nearest
reference image, path, and similarity. The default operating threshold is
`0.80`: a nearest cosine similarity below the configured threshold is returned
as a `distribution_shift_candidate`. The anomaly score is
`max(0, threshold - nearest_similarity)`. This score is an evidence scale for
sorting/review, not a calibrated probability. The threshold is configurable and
0.80 is not a universal scientific definition of OOD.

Results are deterministic and typed/versioned. Equal-similarity reference ties
are resolved by stable image-ID ordering. A query image whose ID also exists in
the reference batch excludes that ID from its comparison candidates, avoiding
self-similarity. If no references remain (including an empty reference set),
the query is returned as unscored rather than being called anomalous. Layer 1B
query failures are preserved as unscored evidence. Zero vectors have cosine
similarity 0.0; non-finite vectors and incompatible spaces/dimensions are
rejected.

Exact comparison costs O(query × reference) time and does not allocate a
combined pairwise matrix. This initial implementation does not use approximate
nearest-neighbor search. A distant representation may reflect a legitimate
change in terrain, season, sensor, illumination, acquisition conditions, or
environment. Distribution shift alone is not evidence of malicious
manipulation; candidates recommend analyst review and do not assert an attack.
