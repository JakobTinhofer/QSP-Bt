use ndarray::Array1;
use num_complex::Complex64;

use crate::compute::cpu::c2x2::C2x2;

pub fn signal_operator(x: f64) -> C2x2 {
    let t = x.acos();
    return C2x2::new([
        [t.cos().into(), Complex64::new(0., t.sin())],
        [Complex64::new(0., t.sin()), t.cos().into()],
    ]);
}

pub fn z_rotation(phi: f64) -> C2x2 {
    let arg = Complex64::new(0., phi / 2.);
    return C2x2::new([[arg.exp(), (0.).into()], [(0.).into(), (-arg).exp()]]);
}

pub fn qsp_unitary(phases: &[f64], x: f64) -> C2x2 {
    assert!(x >= -1. && x <= 1., "x may only be in [-1,1]! Got {}", x);
    assert!(phases.len() > 1, "need at least 2 phases!");
    let mut u = z_rotation(phases[0]);
    for p in &phases[1..] {
        u = u * signal_operator(x) * z_rotation(*p);
    }
    u
}

pub fn qsp_poly(phases: &[f64], xs: &[f64]) -> Array1<Complex64> {
    xs.iter()
        .map(|x| qsp_unitary(phases, *x).get(0, 0))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use rand::distr::{Distribution, Uniform};

    use super::*;

    #[test]
    fn check_unitary() {
        let degree = 20;
        let nr_tests = 100;
        let tol = 1e-12;

        let x_dist = Uniform::new(-1., 1.).expect("Failed to create random distribution!");
        let t_dist = Uniform::new(0., 2. * PI).expect("Failed to create random distribution!");
        let mut rng = rand::rng();
        let i = C2x2::eye();
        let max_dev = (0..nr_tests)
            .map(|_| {
                let x = x_dist.sample(&mut rng);
                let phis = (0..degree)
                    .map(|_| t_dist.sample(&mut rng))
                    .collect::<Vec<f64>>();
                let u = qsp_unitary(&phis, x);

                let res = (u * u.dagger() - i).l1_norm();
                println!(
                    "Running w/ x={}, phis={:?} and u={:?}. Got {}",
                    x,
                    phis,
                    u * u.dagger(),
                    res
                );
                res
            })
            .inspect(|&dev| assert!(dev <= tol, "deviation {} exceeds tolerance {}", dev, tol))
            .reduce(f64::max)
            .expect("Could not find a maximum deviation!");

        println!("Max deviation: {}", max_dev);
    }
}
