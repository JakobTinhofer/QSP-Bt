use ndarray::{Array1, Array2, ArrayView1, Axis, concatenate};

use crate::compute::ComputeBackend;

pub struct RidgeRegularizedBackend<T>
where
    T: ComputeBackend,
{
    backend: T,
    lambda: f64,
}

impl<T: ComputeBackend> RidgeRegularizedBackend<T> {
    pub fn new(b: T, l: f64) -> Self {
        Self {
            backend: b,
            lambda: l,
        }
    }
}

fn phase_sum(phases: &ArrayView1<f64>) -> f64 {
    phases.dot(phases)
}

fn phase_grad(phases: &ArrayView1<f64>) -> Array1<f64> {
    phases.clone().map(|p| 2.0 * p)
}

impl<T: ComputeBackend> ComputeBackend for RidgeRegularizedBackend<T> {
    fn evaluate_f_grad(
        &self,
        phases: &ndarray::prelude::ArrayView1<f64>,
    ) -> (f64, ndarray::prelude::Array1<f64>) {
        let (f, grad) = self.backend.evaluate_f_grad(phases);
        let f_reg = f + self.lambda * phase_sum(phases);
        let grad_reg = grad + self.lambda * phase_grad(phases);
        (f_reg, grad_reg)
    }

    fn evaluate_res_jac(
        &self,
        phases: &ndarray::prelude::ArrayView1<f64>,
    ) -> (ndarray::prelude::Array1<f64>, ndarray::prelude::Array2<f64>) {
        let (res, jac) = self.backend.evaluate_res_jac(phases);
        let n = phases.len();
        let s = self.lambda.sqrt();

        let reg_res = phases.map(|p| s * p);
        let res_aug = concatenate(Axis(0), &[res.view(), reg_res.view()])
            .expect("Failed to concatenate gradient in regularization!");

        let reg_jac = Array2::<f64>::eye(n) * s;
        let jac_aug = concatenate(Axis(0), &[jac.view(), reg_jac.view()]).unwrap();

        (res_aug, jac_aug)
    }

    fn evaluate_f(&self, phases: &ndarray::prelude::ArrayView1<f64>) -> f64 {
        self.backend.evaluate_f(phases) + self.lambda * phase_sum(phases)
    }

    fn evaluate_poly(
        phases: &ndarray::prelude::ArrayView1<f64>,
        xs: &ndarray::prelude::ArrayView1<f64>,
    ) -> ndarray::prelude::Array1<num_complex::Complex64> {
        T::evaluate_poly(phases, xs)
    }

    fn get_target(&self) -> &crate::target::TargetPoly {
        self.backend.get_target()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::ArrayView1;
    use ndarray::{Array1, Array2, array};

    // A fake backend with a hand-computable residual/Jacobian, so we know
    // the right answers in advance. Model: r(θ) = θ - c, Jacobian = I.
    // => f = Σ(θ_i - c_i)²  and  grad = 2(θ - c)  — the "f = Σr², grad = 2Jᵀr"
    // convention the wrapper assumes.
    struct MockBackend {
        c: Array1<f64>,
    }

    impl ComputeBackend for MockBackend {
        fn evaluate_res_jac(&self, phases: &ArrayView1<f64>) -> (Array1<f64>, Array2<f64>) {
            let res = phases.to_owned() - &self.c;
            let jac = Array2::<f64>::eye(phases.len());
            (res, jac)
        }
        fn evaluate_f(&self, phases: &ArrayView1<f64>) -> f64 {
            let (res, _) = self.evaluate_res_jac(phases);
            res.dot(&res)
        }
        fn evaluate_f_grad(&self, phases: &ArrayView1<f64>) -> (f64, Array1<f64>) {
            let (res, jac) = self.evaluate_res_jac(phases);
            (res.dot(&res), jac.t().dot(&res) * 2.0) // (f, 2 Jᵀr)
        }
        // The regularization paths never call these two, so a panicking
        // placeholder is fine — the tests never reach it.
        fn evaluate_poly(
            _phases: &ArrayView1<f64>,
            _xs: &ArrayView1<f64>,
        ) -> Array1<num_complex::Complex64> {
            unimplemented!("not needed for these tests")
        }
        fn get_target(&self) -> &crate::target::TargetPoly {
            unimplemented!("not needed for these tests")
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }
    fn arrays_close(a: &Array1<f64>, b: &Array1<f64>) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn lambda_zero_matches_backend() {
        let c = array![1.0, -2.0, 0.5];
        let theta = array![0.3, 0.7, -1.1];
        let plain = MockBackend { c: c.clone() };
        let ridge = RidgeRegularizedBackend::new(MockBackend { c }, 0.0);

        let (f_plain, g_plain) = plain.evaluate_f_grad(&theta.view());
        let (f_ridge, g_ridge) = ridge.evaluate_f_grad(&theta.view());
        assert!(close(f_plain, f_ridge));
        assert!(arrays_close(&g_plain, &g_ridge));
    }

    #[test]
    fn f_includes_penalty() {
        let theta = array![2.0, 3.0];
        let ridge = RidgeRegularizedBackend::new(
            MockBackend {
                c: array![1.0, 0.0],
            },
            0.5,
        );
        // backend f = (2-1)² + 3² = 10 ; penalty = 0.5*(4+9) = 6.5 ; total = 16.5
        assert!(close(ridge.evaluate_f(&theta.view()), 16.5));
    }

    #[test]
    fn grad_includes_penalty() {
        let theta = array![2.0, 3.0];
        let ridge = RidgeRegularizedBackend::new(
            MockBackend {
                c: array![1.0, 0.0],
            },
            0.5,
        );
        // backend grad = [2,6] ; penalty 2λθ = [2,3] ; total = [4,9]
        let (_f, grad) = ridge.evaluate_f_grad(&theta.view());
        assert!(arrays_close(&grad, &array![4.0, 9.0]));
    }

    // The keystone: the BFGS path (evaluate_f_grad) and the LM path
    // (evaluate_res_jac) must describe the SAME objective.
    #[test]
    fn f_grad_paths_agree() {
        let theta = array![0.3, 0.7, -1.1];
        let ridge = RidgeRegularizedBackend::new(
            MockBackend {
                c: array![1.0, -2.0, 0.5],
            },
            0.7,
        );

        let (f, grad) = ridge.evaluate_f_grad(&theta.view());
        let (res, jac) = ridge.evaluate_res_jac(&theta.view());

        assert!(close(f, res.dot(&res))); // f == Σr² of the augmented system
        let grad_from_res = jac.t().dot(&res) * 2.0; // 2 Jᵀr of the augmented system
        assert!(arrays_close(&grad, &grad_from_res));
    }

    #[test]
    fn augmented_shapes() {
        let theta = array![0.3, 0.7, -1.1]; // n = 3
        let ridge = RidgeRegularizedBackend::new(
            MockBackend {
                c: array![1.0, -2.0, 0.5],
            },
            0.4,
        );
        let (res, jac) = ridge.evaluate_res_jac(&theta.view());
        assert_eq!(res.len(), 6); // 3 residuals + 3 regularization rows
        assert_eq!(jac.shape(), &[6, 3]);
    }
}
