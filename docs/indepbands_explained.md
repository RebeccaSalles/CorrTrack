# Independent band bits, explained (2026-10-23)

## What the LSH search has to do

Each window of each series is summarised by a sketch: 64 numbers, normalised to unit length. The Pearson
correlation of two windows is (approximately) the cosine between their sketches, so finding correlated pairs means
finding sketch pairs whose cosine is at least a gate gamma. Comparing every pair is quadratic; the LSH index exists to
look only at pairs that are likely to pass.

## Sign bits: one random cut of the space

Take a hyperplane through the origin and record, for each sketch, which side of it the sketch falls on: one bit. Two
sketches at angle theta are separated by a random hyperplane with probability theta / pi, so their bits agree with
probability p = 1 - theta / pi. Similar sketches (small angle) agree on most bits; unrelated ones (theta near 90
degrees) agree on half.

At T = 0.8 a true pair has theta = arccos(0.8) = 37 degrees, so p = 0.795. An unrelated pair has p = 0.5.

## Bands: several bits glued into one key

A band glues K bits into one key, and each window is filed in the bucket of that key. Two sketches land in the same
bucket only if all K bits agree: probability p^K. With K = 10, a true pair (p = 0.795) collides with probability 0.10,
an unrelated pair (p = 0.5) with probability 0.001. One band is selective but misses most true pairs, so the index uses
b bands and a query looks in all of its b buckets: a true pair is found if it collides in at least one band,

    recall = 1 - (1 - p^K)^b    if the bands are independent.

K is set from the index size (K = log2(m L / occupancy), so each bucket holds about `occupancy` unrelated windows) and
b is then the smallest number of bands that reaches the recall target (0.95 plus a 0.028 safety margin).

## What CorrTrack does today: bits taken from one pool of 64

Today the hyperplanes are the coordinate axes themselves: bit i is the sign of sketch coordinate i. There are only 64
such bits, and every band picks its K bits from that same pool. With dozens or hundreds of bands of 6 to 11 bits, the
bands share bits heavily.

Sharing breaks the independence the formula needs. For one particular pair the 64 coordinate signs are fixed: say 50
agree and 14 disagree. A band succeeds only if all of its K bits fall among the 50 agreeing ones. Every band draws from
the same 50 out of 64, so a pair that happens to have a few more disagreeing coordinates than average loses in all
bands at once. It is like asking many witnesses who all copied their notes from the same notebook: asking more of them
helps less than it would if each had looked for themselves. The index corrects for this by using many more bands
(an overlap-corrected formula), and the correction grows with K, which grows with m. At m = 4000 and T = 0.7 the pooled
index needs 354 bands where independent bands would need 93. More bands means more buckets to probe and more repeat
collisions per query, which is why CorrTrack's measured exponent sat above textbook LSH.

## The new mode: a fresh random cut for every band bit

Instead of reusing the coordinate axes, the index draws its own hyperplanes in sketch space: one random 64-dimensional
Gaussian vector h for every bit of every band, b x K of them. A window's bit is the sign of h . x, the side of that
hyperplane its sketch falls on. Each hyperplane is a new, independent random cut, so for a given pair of sketches the
bits are independent coin flips with the same probability p = 1 - angle / pi. The bands are then truly independent,
the textbook formula holds exactly, and far fewer bands reach the same recall (46 instead of 132 at m = 4000, T = 0.8).

## How the projections are computed

- When the index is sized (once, when the number of series is known), it draws a matrix H with b x K rows and 64
  columns of standard Gaussian numbers, from a fixed seed, so runs are reproducible.
- At every step the m new windows arrive as an m x 64 matrix X of sketches. One matrix product P = X H^T (m rows,
  b x K columns) gives every projection of every new window at once; this is a single call to the BLAS library.
- For window i and band j, the key is the K signs P[i, jK], ..., P[i, jK + K - 1], packed into an integer (first bit
  most significant). The window is then filed in bucket (j, key), exactly as before.
- Negative correlation: sign(h . (-x)) = -sign(h . x), so the key of the negated sketch is the bitwise complement of
  the key. The query looks up the complement bucket as well, the same trick the pooled index already used.

## What does not change

The sketch, its normalisation, the Hamming prefilter (still on the 64 coordinate signs), the dot-product gate, the
exact Pearson validation, the recall target and margin, and the negative-correlation lookup. Only how bucket keys are
formed and how many bands are needed. Precision stays 1 because every reported pair is still validated exactly.

## Cost and gain

Extra work: b x K x 64 multiply-adds per inserted window (about 33,000 at m = 1000, T = 0.7), done in one BLAS product
per step: about 8% of the run on the cluster (it was 23% when computed window by window). Gain: the scan probes far
fewer bands and meets fewer repeat collisions, so it halves or better. Net, at m = 1000: 1.57x faster than pooled at
T = 0.8 and 1.81x at T = 0.7, with the same recall, and the exponent of m drops to near textbook LSH.

## How to present it in the paper (2026-10-24)

There is one sketching algorithm, not two. The incremental sketch is the only computation that touches the raw data;
the band keys are LSH hash codes computed from the sketch, the standard construction of LSH for cosine similarity
(Charikar, STOC 2002) applied to the sketch vectors. A hash is a function of a representation, not a second
representation: the pooled index already hashed the sketch (its keys are the signs of the sketch coordinates, i.e.
sign random projection with the coordinate axes as hyperplanes); the independent mode hashes the same sketch with
random hyperplanes instead. Formally, with s = R x the sketch of window x, a key bit is sign(h . s) = sign((hR) . x),
so it inherits the sketch's incremental, lag-reusable and normalisation-after-sketching properties for free, and it
costs one matrix product per step on the sketches already computed.

Suggested structure:
- Sketching phase (one pass over the stream, incremental): for every window, the sketch s_t.
- Candidate search (LSH): for each new window, its hash codes g_j(s_t) = sign(H_j s_t), j = 1..b, with H_j a K x k
  Gaussian matrix drawn once, computed for all new windows of a step in one matrix product; file the window in
  bucket (j, g_j(s_t)) for every band j; a query probes its own buckets and their complements (negative correlation).
- Why random hyperplanes rather than the sketch's own coordinate signs: k coordinate signs give only k distinct bits,
  so b bands of K bits reuse bits, their collisions become dependent, and the number of bands needed for a recall
  target grows faster than N^rho; independent hyperplanes restore the textbook guarantee
  recall = 1 - (1 - p^K)^b with p = 1 - arccos(cos)/pi, and the measured exponent drops accordingly (table in
  docs/indepbands_results_2026-10-20.md).
- Wording to avoid: "a second sketch" or "two sketching algorithms". Use "the sketch" and "its hash codes".
