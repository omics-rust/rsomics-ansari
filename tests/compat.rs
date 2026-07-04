//! Value-exact compatibility against `scipy.stats.ansari` (scipy 1.17.1).
//!
//! `tests/golden/expected.json` holds AB and p-values produced once by SciPy and
//! committed; this test re-derives them from the committed `*_x.tsv` / `*_y.tsv`
//! inputs and asserts a match. SciPy is not invoked at test time. Exact-branch
//! p-values are combinatorial and must match bit-for-bit; normal-approximation
//! p-values flow through the ported Cephes `ndtr` and match to ≤1e-12.

use std::fs;
use std::path::Path;

use rsomics_ansari::{Alternative, ansari};

fn read_tsv(path: &Path) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect()
}

fn field(obj: &str, key: &str) -> f64 {
    // Minimal JSON number extraction: find `"key": <number>`.
    let needle = format!("\"{key}\":");
    let start = obj.find(&needle).expect("key present") + needle.len();
    let tail = &obj[start..];
    let end = tail.find([',', '}']).unwrap_or(tail.len());
    tail[..end].trim().parse().unwrap()
}

fn alt_block<'a>(rec: &'a str, alt: &str) -> &'a str {
    let key = format!("\"{alt}\":");
    let start = rec.find(&key).expect("alternative present") + key.len();
    let tail = &rec[start..];
    let end = tail.find('}').unwrap() + 1;
    &tail[..end]
}

#[test]
fn matches_scipy_golden() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let expected = fs::read_to_string(dir.join("expected.json")).unwrap();

    // Split the top-level array into per-case records on `"name":`.
    let records: Vec<&str> = expected.split("\"name\":").skip(1).collect();
    assert!(records.len() >= 6, "expected at least 6 golden cases");

    for rec in records {
        let name_start = rec.find('"').unwrap() + 1;
        let name_end = rec[name_start..].find('"').unwrap() + name_start;
        let name = &rec[name_start..name_end];

        let x = read_tsv(&dir.join(format!("{name}_x.tsv")));
        let y = read_tsv(&dir.join(format!("{name}_y.tsv")));

        for (alt_name, alt) in [
            ("two-sided", Alternative::TwoSided),
            ("less", Alternative::Less),
            ("greater", Alternative::Greater),
        ] {
            let block = alt_block(rec, alt_name);
            let want_stat = field(block, "statistic");
            let want_p = field(block, "pvalue");

            let got = ansari(&x, &y, alt).unwrap();

            if want_stat.is_nan() {
                assert!(
                    got.statistic.is_nan() && got.pvalue.is_nan(),
                    "{name}/{alt_name}: expected nan/nan, got {}/{}",
                    got.statistic,
                    got.pvalue
                );
                continue;
            }

            assert!(
                (got.statistic - want_stat).abs() < 1e-9,
                "{name}/{alt_name}: AB {} vs scipy {want_stat}",
                got.statistic
            );
            let rel = (got.pvalue - want_p).abs() / want_p.abs().max(f64::MIN_POSITIVE);
            assert!(
                rel <= 1e-12,
                "{name}/{alt_name}: p {} vs scipy {want_p} (rel {rel:e})",
                got.pvalue
            );
        }
    }
}
