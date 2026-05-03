// benches/qsp_bench.rs
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ex3_rs::target::TargetPoly;
use ndarray::Array1;
use num_complex::Complex64;
use rand::RngExt;
use rand::{SeedableRng, rngs::StdRng};
use std::f64::consts::PI;

use ex3_rs::compute::ComputeBackend;
use ex3_rs::compute::cpu::CpuComputeBackend;
use ex3_rs::compute::cpu::c2x2::C2x2;
use ex3_rs::compute::cpu::qsp::{signal_operator, z_rotation};

// ---------- Test fixtures ----------

fn make_target(i_count: usize) -> TargetPoly {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let xs: Array1<f64> = (0..i_count)
        .map(|k| ((k as f64 + 0.5) / i_count as f64 * PI).cos())
        .collect();
    let ys: Array1<Complex64> = (0..i_count)
        .map(|_| {
            if rng.random::<bool>() {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();
    TargetPoly::from_parts(xs, ys)
}

fn make_phases(n: usize) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    (0..n).map(|_| rng.random_range(0.0..2.0 * PI)).collect()
}

// ---------- Microbenchmarks ----------

fn bench_signal_operator(c: &mut Criterion) {
    let xs: Vec<f64> = (0..1024)
        .map(|k| ((k as f64 + 0.5) / 1024.0 * PI).cos())
        .collect();
    c.bench_function("signal_operator/1024", |b| {
        b.iter(|| {
            let mut acc = C2x2::empty();
            for &x in &xs {
                acc = acc + signal_operator(black_box(x));
            }
            black_box(acc)
        })
    });
}

fn bench_z_rotation(c: &mut Criterion) {
    let phis: Vec<f64> = (0..1024).map(|k| (k as f64) * 0.01).collect();
    c.bench_function("z_rotation/1024", |b| {
        b.iter(|| {
            let mut acc = C2x2::empty();
            for &p in &phis {
                acc = acc + z_rotation(black_box(p));
            }
            black_box(acc)
        })
    });
}

fn bench_c2x2_mul(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xABCD);
    let mats: Vec<C2x2> = (0..1024)
        .map(|_| {
            C2x2::new([
                [
                    Complex64::new(rng.random(), rng.random()),
                    Complex64::new(rng.random(), rng.random()),
                ],
                [
                    Complex64::new(rng.random(), rng.random()),
                    Complex64::new(rng.random(), rng.random()),
                ],
            ])
        })
        .collect();
    c.bench_function("c2x2_mul/1023_chained", |b| {
        b.iter(|| {
            let mut acc = mats[0];
            for m in &mats[1..] {
                acc = acc * black_box(*m);
            }
            black_box(acc)
        })
    });
}

// ---------- evaluate_both at multiple sizes ----------

fn bench_evaluate_both(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_both");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(15));
    for &(i_count, n) in &[(50_usize, 50_usize), (500, 2000)] {
        let target = make_target(i_count);
        let backend =
            CpuComputeBackend::new(target, ex3_rs::compute::cpu::BackendMode::MultiThread);
        let phases = make_phases(n);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("I={},N={}", i_count, n)),
            &(backend, phases),
            |b, (backend, phases)| {
                b.iter(|| {
                    let (loss, grad) = backend.evaluate_f_grad(black_box(phases));
                    black_box((loss, grad))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_signal_operator,
    bench_z_rotation,
    bench_c2x2_mul,
    bench_evaluate_both
);
criterion_main!(benches);
