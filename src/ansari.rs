//! The Ansari-Bradley two-sample scale test (`scipy.stats.ansari`).
//!
//! AB is the sum of the symmetric ranks `min(rank, N-rank+1)` over the first
//! sample. The p-value uses the exact null distribution when both samples have
//! fewer than 55 observations and there are no ties; otherwise a normal
//! approximation with SciPy's even-N / odd-N (and tie-corrected) mean/variance.

use rsomics_common::{Result, RsomicsError};
use serde::Serialize;

use crate::exact::ExactDist;
use crate::ndtr::ndtr;
use crate::rank::rank_aggregate;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

/// Result of an Ansari-Bradley test: the AB statistic and its p-value.
#[derive(Debug, Clone, Serialize)]
pub struct AnsariResult {
    pub statistic: f64,
    pub pvalue: f64,
}

/// Run the Ansari-Bradley scale test of `x` against `y`.
pub fn ansari(x: &[f64], y: &[f64], alternative: Alternative) -> Result<AnsariResult> {
    let n = x.len();
    let m = y.len();
    if n < 1 {
        return Err(RsomicsError::InvalidInput(
            "x must have at least one observation".into(),
        ));
    }
    if m < 1 {
        return Err(RsomicsError::InvalidInput(
            "y must have at least one observation".into(),
        ));
    }

    let big_n = n + m;
    let mut pooled: Vec<(f64, bool)> = Vec::with_capacity(big_n);
    pooled.extend(x.iter().map(|&v| (v, true)));
    pooled.extend(y.iter().map(|&v| (v, false)));
    let agg = rank_aggregate(pooled);
    let ab = agg.ab;

    let exact = m < 55 && n < 55 && !agg.repeats;
    let pvalue = if exact {
        let dist = ExactDist::build(n, m);
        match alternative {
            Alternative::TwoSided => 2.0 * dist.cdf(ab).min(dist.sf(ab)),
            // AB is smaller when the x-scale is larger, so 'greater' maps to cdf.
            Alternative::Greater => dist.cdf(ab),
            Alternative::Less => dist.sf(ab),
        }
        .min(1.0)
    } else {
        normal_pvalue(ab, n, m, agg.fac, agg.repeats, alternative)
    };

    Ok(AnsariResult {
        statistic: ab,
        pvalue,
    })
}

fn normal_pvalue(
    ab: f64,
    n: usize,
    m: usize,
    fac: f64,
    repeats: bool,
    alternative: Alternative,
) -> f64 {
    let nf = n as f64;
    let mf = m as f64;
    let big_n = (n + m) as f64;
    let odd = (n + m) % 2 == 1;

    let mn_ab = if odd {
        nf * (big_n + 1.0).powi(2) / 4.0 / big_n
    } else {
        nf * (big_n + 2.0) / 4.0
    };

    let var_ab = if repeats {
        if odd {
            mf * nf * (16.0 * big_n * fac - (big_n + 1.0).powi(4))
                / (16.0 * big_n.powi(2) * (big_n - 1.0))
        } else {
            mf * nf * (16.0 * fac - big_n * (big_n + 2.0).powi(2)) / (16.0 * big_n * (big_n - 1.0))
        }
    } else if odd {
        nf * mf * (big_n + 1.0) * (3.0 + big_n.powi(2)) / (48.0 * big_n.powi(2))
    } else {
        mf * nf * (big_n + 2.0) * (big_n - 2.0) / 48.0 / (big_n - 1.0)
    };

    let z = (mn_ab - ab) / var_ab.sqrt();
    match alternative {
        Alternative::Less => ndtr(z),
        Alternative::Greater => ndtr(-z),
        Alternative::TwoSided => 2.0 * ndtr(-z.abs()),
    }
}

#[cfg(test)]
mod tests {
    use super::{Alternative, ansari};

    #[test]
    fn exact_two_sided_small() {
        // scipy.stats.ansari([1..7],[8..12]) -> AB=27.0 (exact, no ties).
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = [8.0, 9.0, 10.0, 11.0, 12.0];
        let r = ansari(&x, &y, Alternative::TwoSided).unwrap();
        assert!((r.statistic - 27.0).abs() < 1e-12);
        assert!((r.pvalue - 0.5252525252525253).abs() < 1e-12);
    }

    #[test]
    fn normal_branch_with_ties() {
        // Ties force the normal approximation regardless of size.
        let x = [1.0, 1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 2.0, 4.0];
        let r = ansari(&x, &y, Alternative::TwoSided).unwrap();
        assert!(r.pvalue > 0.0 && r.pvalue <= 1.0);
    }

    #[test]
    fn rejects_empty() {
        assert!(ansari(&[], &[1.0], Alternative::TwoSided).is_err());
        assert!(ansari(&[1.0], &[], Alternative::TwoSided).is_err());
    }
}
