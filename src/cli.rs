use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use rsomics_common::{CommonFlags, RsomicsError, ToolMeta, run};

use rsomics_ansari::{Alternative, ansari, parse_values};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum AltArg {
    #[value(name = "two-sided")]
    TwoSided,
    Less,
    Greater,
}

impl From<AltArg> for Alternative {
    fn from(a: AltArg) -> Self {
        match a {
            AltArg::TwoSided => Alternative::TwoSided,
            AltArg::Less => Alternative::Less,
            AltArg::Greater => Alternative::Greater,
        }
    }
}

/// Ansari-Bradley two-sample scale test (`scipy.stats.ansari`).
///
/// Each input is a single-column file (one value per line); `-` reads stdin (at
/// most one input may be stdin). Output is a single line `AB<TAB>p`, where `AB`
/// is the statistic for the first sample.
#[derive(Parser, Debug)]
#[command(name = "rsomics-ansari", version, about, long_about = None)]
pub struct Cli {
    /// First sample (`x`): one value per line.
    #[arg(value_name = "X")]
    pub x: PathBuf,

    /// Second sample (`y`): one value per line.
    #[arg(value_name = "Y")]
    pub y: PathBuf,

    /// Alternative hypothesis.
    #[arg(long, value_enum, default_value = "two-sided")]
    pub alternative: AltArg,

    #[command(flatten)]
    pub common: CommonFlags,
}

fn read_sample(path: &PathBuf) -> rsomics_common::Result<Vec<f64>> {
    if path.as_os_str() == "-" {
        parse_values(std::io::stdin().lock())
    } else {
        let f = File::open(path).map_err(RsomicsError::Io)?;
        parse_values(BufReader::new(f))
    }
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let xs = read_sample(&self.x)?;
            let ys = read_sample(&self.y)?;
            let result = ansari(&xs, &ys, self.alternative.into())?;
            if !common.json {
                println!("{}\t{}", result.statistic, result.pvalue);
            }
            Ok(result)
        })
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
