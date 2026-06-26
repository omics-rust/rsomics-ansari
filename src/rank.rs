//! Symmetric (Ansari-Bradley) ranking of the pooled samples, matching SciPy
//! `_rankdata(method='average')` followed by `minimum(rank, N - rank + 1)`.
//!
//! SciPy stably sorts the pooled values, gives each member of a tie group the
//! average ordinal rank (`first + (count-1)/2`, 1-based), folds it into the
//! symmetric score `min(rank, N-rank+1)`, sums the scores over sample `x` to get
//! `AB`, and uses `sum(symrank^2)` over the whole pool plus a "any ties" flag for
//! the tie-corrected variance.

/// Aggregates from the pooled symmetric ranking needed by the Ansari-Bradley test.
pub struct RankAggregates {
    /// AB = Σ of symmetric ranks over the members of sample `x`.
    pub ab: f64,
    /// Σ symrankⱼ² over the entire pool (the `fac` term in the tie variance).
    pub fac: f64,
    /// Whether any value appears more than once across the pooled samples.
    pub repeats: bool,
}

/// Rank the pooled samples and reduce to the Ansari-Bradley aggregates. `pooled`
/// pairs each value with `true` if it came from sample `x`. Sorting is on the
/// value (`total_cmp`) to match SciPy's stable argsort; tie membership, not the
/// within-tie order, is what matters.
#[must_use]
pub fn rank_aggregate(mut pooled: Vec<(f64, bool)>) -> RankAggregates {
    let n = pooled.len();
    let big_n = n as f64;
    pooled.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut ab = 0.0_f64;
    let mut fac = 0.0_f64;
    let mut repeats = false;

    let mut i = 0;
    while i < n {
        let v = pooled[i].0;
        let mut j = i + 1;
        while j < n && pooled[j].0 == v {
            j += 1;
        }
        let count = j - i;
        let avg_rank = (i + 1) as f64 + (count as f64 - 1.0) / 2.0;
        let symrank = avg_rank.min(big_n - avg_rank + 1.0);
        let sq = symrank * symrank;
        for &(_, in_x) in &pooled[i..j] {
            fac += sq;
            if in_x {
                ab += symrank;
            }
        }
        if count > 1 {
            repeats = true;
        }
        i = j;
    }

    RankAggregates { ab, fac, repeats }
}

#[cfg(test)]
mod tests {
    use super::rank_aggregate;

    fn pool(x: &[f64], y: &[f64]) -> Vec<(f64, bool)> {
        let mut p: Vec<(f64, bool)> = x.iter().map(|&v| (v, true)).collect();
        p.extend(y.iter().map(|&v| (v, false)));
        p
    }

    #[test]
    fn no_ties_symmetric_scores() {
        // N=4, ranks 1,2,3,4 -> symranks 1,2,2,1. x={1,4} (extremes) -> AB=1+1=2.
        let a = rank_aggregate(pool(&[1.0, 4.0], &[2.0, 3.0]));
        assert_eq!(a.ab, 2.0);
        // fac = 1+4+4+1 = 10
        assert_eq!(a.fac, 10.0);
        assert!(!a.repeats);
    }

    #[test]
    fn ties_average_then_fold() {
        // pooled 1,2,2,3: ranks 1,2.5,2.5,4 -> symranks 1, min(2.5,2.5)=2.5, 2.5, 1.
        let a = rank_aggregate(pool(&[1.0, 3.0], &[2.0, 2.0]));
        assert_eq!(a.ab, 2.0); // x={1,3} -> symranks 1 + 1
        // fac = 1 + 2.5^2 + 2.5^2 + 1 = 14.5
        assert!((a.fac - 14.5).abs() < 1e-12);
        assert!(a.repeats);
    }
}
