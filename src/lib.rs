//! Ansari-Bradley two-sample scale test — `scipy.stats.ansari` equivalent.
//!
//! Two single-column inputs (one value per line) give the samples `x` and `y`.
//! The test reports the AB statistic and the p-value for the chosen alternative,
//! using the exact null distribution for small no-tie samples and SciPy's normal
//! approximation (with even/odd-N and tie-corrected variance) otherwise.

mod ansari;
mod exact;
mod ndtr;
mod rank;

use std::io::BufRead;

use rsomics_common::{Result, RsomicsError};

pub use ansari::{Alternative, AnsariResult, ansari};

/// Parse a single-column TSV of numeric values, one per line. Blank lines are
/// skipped; a non-numeric value is a hard error (fail-loud).
pub fn parse_values<R: BufRead>(reader: R) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(RsomicsError::Io)?;
        let field = line.trim();
        if field.is_empty() {
            continue;
        }
        let value: f64 = field.parse().map_err(|_| {
            RsomicsError::InvalidInput(format!(
                "line {}: value '{field}' is not a number",
                lineno + 1
            ))
        })?;
        values.push(value);
    }
    if values.is_empty() {
        return Err(RsomicsError::InvalidInput("no values in input".into()));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::parse_values;

    #[test]
    fn parse_skips_blank_lines() {
        let v = parse_values("1\n\n2.5\n3\n".as_bytes()).unwrap();
        assert_eq!(v, vec![1.0, 2.5, 3.0]);
    }

    #[test]
    fn rejects_non_numeric() {
        assert!(parse_values("1\nfoo\n".as_bytes()).is_err());
    }

    #[test]
    fn rejects_empty() {
        assert!(parse_values("\n\n".as_bytes()).is_err());
    }
}
