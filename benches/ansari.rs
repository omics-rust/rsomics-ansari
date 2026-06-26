use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_ansari::{Alternative, ansari};
use std::hint::black_box;

// Deterministic pseudo-random samples large enough to fall in the normal-
// approximation branch (the hot path for non-trivial inputs).
fn sample(seed: u64, n: usize, scale: f64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            (u - 0.5) * scale
        })
        .collect()
}

fn bench_ansari(c: &mut Criterion) {
    let x = sample(1, 100_000, 2.0);
    let y = sample(2, 100_000, 1.0);
    c.bench_function("ansari_normal_200k", |b| {
        b.iter(|| ansari(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap());
    });
}

criterion_group!(benches, bench_ansari);
criterion_main!(benches);
