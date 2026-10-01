# Layer 1H: Silhouette / Clean-Label Cluster Validation

Layer 1H measures whether Layer 1B feature vectors form separated groups under
the labels already present in a Layer 1A `Dataset`. It performs no image loading
or feature inference. The default and currently supported metric is cosine
distance, `1 - cosine_similarity`.

For each scorable image, `a(i)` is its mean distance to the other images in its
assigned class. For each competing class, the detector computes the mean
distance to that class; `b(i)` is the smallest such mean. The score is
`(b(i) - a(i)) / max(a(i), b(i))`, with a zero result when the denominator is
zero. Scores are separation signals, not calibrated probabilities.

An image's assigned class is the complete sorted set of valid category IDs in
its annotations. Thus, images annotated with `{cat, dog}` form a class distinct
from images annotated with `{cat}`. Category names are carried through when
available; category IDs define identity. Images with no valid annotations or no
embedding are listed as unscorable. Zero/non-finite/wrong-dimension vectors are
reported as unscorable. The assigned class must have at least the configured
`minimum_class_size` (default 2) scorable images so that `a(i)` has peers. A
singleton class cannot be scored itself, but can serve as a competing class for
`b(i)`. A single-class dataset therefore has no scorable silhouettes.

Results include per-image distances, score, assigned and nearest competing
classes, neutral interpretation, per-class summaries, dataset mean/median/min/max,
and explicit unscorable reasons. Ties use category ID ordering and output uses
stable image ordering.

The exact method takes O(n²d) arithmetic for `n` usable images and embedding
dimension `d`, while computing distances as needed rather than storing a full
pairwise matrix. Silhouette analysis measures feature-space separation and is
not, by itself, evidence of malicious manipulation or poisoning. Low or negative
scores can reflect legitimate class overlap, difficult samples, poor embeddings,
insufficient reference data, distribution shift, or incorrect class assumptions.
