# Layer 1D: Label Manipulation / Clean-Label Anomaly Signal

Layer 1D checks whether an image's assigned category is consistent with the
visual embedding neighborhood produced by Layer 1B. It consumes the Layer 1A
`Dataset` and the matching Layer 1B `FeatureBatch`; it does not reload images,
run a model, or contact a network service.

For each labeled image, the detector finds up to `k` other labeled images using
exact cosine similarity in the FeatureBatch embedding space. The default is
`k=5`. The `label_inconsistency_score` is the fraction of available top-k
neighbors that do not share any of the image's assigned categories. The default
score threshold is `0.8`; the default minimum same-class support is one neighbor.
These are configurable development operating values, not scientifically
validated cutoffs. A zero embedding has cosine similarity 0.0 and is handled
without producing NaN values.

Results are typed and versioned. They include assigned labels, ordered neighbor
IDs, paths, labels and similarities, neighbor class counts, same/different-class
counts, same-class and strongest-competing-class mean similarities, score,
threshold, embedding-space ID, and available contributor/source/batch metadata.
Images without usable annotations are listed as exclusions. Missing embeddings
for labeled images, mismatched embedding spaces, invalid vectors, and dimension
mismatches raise explicit errors. When contributor/source/batch metadata is not
present in the input objects, the result keeps those fields unavailable.

A `review` state recommends analyst review; it does not assert that an image is
poisoned or that its label is malicious. A high neighborhood disagreement can
arise for many reasons, and this detector makes no accuracy guarantee. Classes
without sufficient same-class neighbors are marked `insufficient_class_support`
instead of being automatically flagged. Exact search costs O(n²) time and does
not allocate an n-by-n matrix; approximate search is outside this initial slice.
