# rsomics-ansari

Ansari-Bradley two-sample scale test — a value-exact Rust port of
`scipy.stats.ansari`.

The Ansari-Bradley test is a non-parametric test for equal scale (dispersion)
of the distributions behind two samples. It ranks the pooled data symmetrically
(`1, 2, …, ⌈N/2⌉, …, 2, 1`) and sums those scores over the first sample to form
the statistic `AB`. The p-value is exact (combinatorial) for small samples
without ties and a tie-corrected normal approximation otherwise.

## Usage

```sh
rsomics-ansari X.tsv Y.tsv [--alternative two-sided|less|greater]
```

Each input is a single-column file, one numeric value per line; `-` reads
stdin. Output is one line `AB<TAB>p`:

```sh
$ rsomics-ansari x.tsv y.tsv --alternative two-sided
27	0.5252525252525253
```

`--json` emits a structured envelope instead. `-t/--threads`, `-q/--quiet`
follow the shared rsomics CLI conventions.

## Value exactness

Verified against scipy 1.17.1 across 1200 random cases spanning the exact and
normal branches and all three alternatives:

- `AB` statistic: bit-exact.
- Exact-branch p-value: **bit-identical**. The exact null distribution is a
  faithful port of SciPy's `gscale` (Dinneen & Blakesley AS 93), reproducing
  its `f32` frequency accumulation and NumPy's pairwise summation of the cdf/sf,
  so every bit matches.
- Normal-approximation p-value: ≤ 1e-12 relative error (it flows through a
  Cephes `ndtr` port matching `scipy.special.ndtr`; the residual is last-ULP).

## Origin

This crate is an independent Rust reimplementation of `scipy.stats.ansari`.

- Method: Ansari, A. R. and Bradley, R. A. (1960), "Rank-sum tests for
  dispersions", *Annals of Mathematical Statistics*, 31, 1174-1189.
- Exact null distribution: Dinneen, L. C. and Blakesley, B. C. (1976),
  "Algorithm AS 93: A Generator for the Null Distribution of the
  Ansari-Bradley W Statistic", *Applied Statistics*, 25(1).
  [doi:10.2307/2346534](https://doi.org/10.2307/2346534). The `gscale`
  recurrence is ported from SciPy's `_ansari_swilk_statistics.pyx` (BSD-3),
  preserving its `f32` arithmetic so the exact p-value is bit-identical.
- Normal CDF: Cephes `ndtr` (Moshier), matching `scipy.special.ndtr`.
- Reference behaviour: SciPy `scipy.stats.ansari` (1.17.1, BSD-3-Clause).
  Golden expected values were produced once with SciPy and committed; the
  compatibility test runs without SciPy.

License: MIT OR Apache-2.0.
Upstream credit: [SciPy](https://github.com/scipy/scipy) (BSD-3-Clause).
