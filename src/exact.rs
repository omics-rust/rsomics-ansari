//! Exact null distribution of the Ansari-Bradley W statistic.
//!
//! Faithful port of SciPy's `gscale`, itself a translation of Dinneen &
//! Blakesley's AS 93 generator (Applied Statistics 25(1), 1976). The algorithm
//! accumulates the frequency table in `f32` (the original uses `REAL*4`); SciPy's
//! exact p-value is built from that `f32` table, so the port must use `f32`
//! arithmetic and the same iteration order to reproduce SciPy bit-for-bit. The
//! table is then read back as `f64` for the cdf/sf sums, mirroring `_ABW`'s
//! `freqs.astype(np.float64)`.
//!
//! `gscale(test, other)` is called as `gscale(n, m)` where `n = len(x)`,
//! `m = len(y)`. `astart = ((test+1)/2) * (1 + test/2)` is the minimum statistic
//! value; the table length is `LL = test*other/2 + 1`.

/// The exact null distribution: the minimum statistic value and the `f64` view of
/// the `f32` frequency table.
pub struct ExactDist {
    astart: u32,
    freqs: Vec<f64>,
    total: f64,
}

/// NumPy's pairwise summation (`numpy/_core/src/umath/loops_utils.h.src`), so the
/// cdf/sf partial sums match `_ABW`'s `freqs[...].sum()` to the bit. Below 128
/// elements NumPy uses an 8-accumulator unrolled loop; above, it splits in half
/// at a multiple of 8.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    if n < 8 {
        let mut s = 0.0;
        for &x in a {
            s += x;
        }
        return s;
    }
    if n <= 128 {
        let mut acc = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i + 8 <= n {
            for (j, slot) in acc.iter_mut().enumerate() {
                *slot += a[i + j];
            }
            i += 8;
        }
        let mut s =
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        while i < n {
            s += a[i];
            i += 1;
        }
        return s;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

fn start1(a: &mut [f32], n: usize) {
    let lout = 1 + n / 2;
    a[..lout].fill(2.0);
    if n.is_multiple_of(2) {
        a[lout - 1] = 1.0;
    }
}

fn start2(a: &mut [f32], n: usize) {
    let odd = n % 2;
    let mut av: f32 = 1.0;
    let mut bv: f32 = 3.0;
    let cv: f32 = if odd == 1 { 2.0 } else { 0.0 };
    let ndo = (n + 2 + odd) / 2 - odd;

    for slot in a.iter_mut().take(ndo) {
        *slot = av;
        av += bv;
        bv = 4.0 - bv;
    }

    av = 1.0;
    bv = 3.0;
    let mut ind = n - odd;
    while ind >= ndo {
        a[ind] = av + cv;
        av += bv;
        bv = 4.0 - bv;
        if ind == 0 {
            break;
        }
        ind -= 1;
    }

    if odd == 1 {
        a[ndo * 2 - 1] = 2.0;
    }
}

fn frqadd(a: &mut [f32], b: &[f32], lenb: usize, offset: usize) -> usize {
    for ind in 0..lenb {
        a[offset + ind] += 2.0 * b[ind];
    }
    lenb + offset
}

// `i2` walks negative→positive and gates the `b[i2]` read, so it is part of the
// AS 93 recurrence rather than a plain index counter.
#[allow(clippy::explicit_counter_loop)]
fn imply(a: &mut [f32], curlen: usize, reslen: usize, b: &mut [f32], offset: usize) -> usize {
    let mut i2 = -(offset as isize);
    let mut j2 = (reslen - offset) as isize;
    let j2min = (j2 + 1) / 2 - 1;
    let nextlenb = j2 as usize;
    let mut j1 = reslen as isize - 1;
    j2 -= 1;

    for i1 in 0..reslen.div_ceil(2) {
        let summ = if i2 < 0 {
            a[i1]
        } else {
            let s = a[i1] + b[i2 as usize];
            a[i1] = s;
            s
        };
        i2 += 1;
        if j2 >= j2min {
            let diff = if j1 > curlen as isize - 1 {
                summ
            } else {
                summ - a[j1 as usize]
            };
            b[i1] = diff;
            b[j2 as usize] = diff;
            j2 -= 1;
        }
        a[j1 as usize] = summ;
        j1 -= 1;
    }
    nextlenb
}

fn gscale(test: usize, other: usize) -> (u32, Vec<f32>) {
    let m = test.min(other);
    let n = test.max(other);
    let astart = test.div_ceil(2) * (1 + test / 2);
    let ll = test * other / 2 + 1;
    let symm = (m + n).is_multiple_of(2);
    let odd = n % 2;

    let mut a1 = vec![0.0_f32; ll];
    let mut a2 = vec![0.0_f32; ll];
    let mut a3 = vec![0.0_f32; ll];

    if m == 0 {
        a1[0] = 1.0;
        return (astart as u32, a1);
    }
    if m == 1 {
        start1(&mut a1, n);
        if !(symm || other > test) {
            a1[0] = 1.0;
            a1[ll - 1] = 2.0;
        }
        return (astart as u32, a1);
    }
    if m == 2 {
        start2(&mut a1, n);
        if !(symm || other > test) {
            a1[..ll].reverse();
        }
        return (astart as u32, a1);
    }

    let mut loop_m = 3;
    let mut part_no;
    let (mut len2, mut len3, mut n2b1, mut n2b2);
    if odd == 1 {
        start1(&mut a1, n);
        start2(&mut a2, n - 1);
        len2 = n;
        len3 = 0;
        n2b1 = 1;
        n2b2 = 2;
        part_no = 0;
    } else {
        start2(&mut a1, n);
        start1(&mut a2, n - 1);
        start2(&mut a3, n - 2);
        len2 = n / 2;
        len3 = n - 1;
        n2b1 = 2;
        n2b2 = 1;
        part_no = 1;
    }
    let mut len1 = if odd == 1 { 1 + n / 2 } else { n + 1 };

    while loop_m <= m {
        if part_no == 0 {
            let l1out = frqadd(&mut a1, &a2, len2, n2b1);
            len1 += n;
            len3 = imply(&mut a1, l1out, len1, &mut a3, loop_m);
            n2b1 += 1;
            loop_m += 1;
            part_no = 1;
        } else {
            let l2out = frqadd(&mut a2, &a3, len3, n2b2);
            len2 += n - 1;
            imply(&mut a2, l2out, len2, &mut a3, loop_m);
            n2b2 += 1;
            loop_m += 1;
            part_no = 0;
        }
    }

    if !symm {
        let ks = (m + 3) / 2 - 1;
        for ind in 0..len2 {
            a1[ks + ind] += a2[ind];
        }
    }
    if other > test {
        a1[..ll].reverse();
    }

    (astart as u32, a1)
}

impl ExactDist {
    /// Build the distribution for sample sizes `n` (length of `x`) and `m`
    /// (length of `y`), via `gscale(n, m)`.
    #[must_use]
    pub fn build(n: usize, m: usize) -> ExactDist {
        let (astart, a1) = gscale(n, m);
        let freqs: Vec<f64> = a1.iter().map(|&v| v as f64).collect();
        let total: f64 = pairwise_sum(&freqs);
        ExactDist {
            astart,
            freqs,
            total,
        }
    }

    /// Cumulative distribution `P(W ≤ k)`. Matches `_ABW.cdf`: index by
    /// `ceil(k - astart)`.
    #[must_use]
    pub fn cdf(&self, k: f64) -> f64 {
        let ind = (k - self.astart as f64).ceil() as isize;
        let upper = (ind + 1).clamp(0, self.freqs.len() as isize) as usize;
        pairwise_sum(&self.freqs[..upper]) / self.total
    }

    /// Survival `P(W ≥ k)`. Matches `_ABW.sf`: index by `floor(k - astart)`.
    #[must_use]
    pub fn sf(&self, k: f64) -> f64 {
        let ind = (k - self.astart as f64).floor() as isize;
        let lower = ind.clamp(0, self.freqs.len() as isize) as usize;
        pairwise_sum(&self.freqs[lower..]) / self.total
    }
}

#[cfg(test)]
mod tests {
    use super::ExactDist;

    #[test]
    fn matches_gscale_small_symmetric() {
        // gscale(5, 5): astart=9, freqs sum to C(10,5)=252, length 13.
        let d = ExactDist::build(5, 5);
        assert_eq!(d.astart, 9);
        assert_eq!(d.freqs.len(), 13);
        assert!((d.total - 252.0).abs() < 1e-9);
        assert_eq!(d.freqs[0], 2.0);
        assert_eq!(*d.freqs.last().unwrap(), 2.0);
    }

    #[test]
    fn matches_gscale_asymmetric() {
        // gscale(3, 2): astart=4, freqs=[2,3,4,1], total=10.
        let d = ExactDist::build(3, 2);
        assert_eq!(d.astart, 4);
        assert_eq!(d.freqs, vec![2.0, 3.0, 4.0, 1.0]);
    }

    #[test]
    fn cdf_sf_partition() {
        let d = ExactDist::build(5, 5);
        // cdf at minimum value equals its mass / total.
        assert!((d.cdf(9.0) - 2.0 / 252.0).abs() < 1e-12);
        assert!((d.sf(21.0) - 2.0 / 252.0).abs() < 1e-12);
        assert!((d.cdf(15.0) + d.sf(16.0) - 1.0).abs() < 1e-12);
    }
}
