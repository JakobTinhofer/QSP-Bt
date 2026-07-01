use std::ops::{Add, Mul, Sub};

use num_complex::{Complex64, ComplexFloat};

pub trait C2x2: Copy {
    fn get(&self, i: usize, j: usize) -> Complex64;
    fn get_direct<const I: usize, const J: usize>(&self) -> Complex64;

    #[inline(always)]
    fn entries(&self) -> [[Complex64; 2]; 2] {
        [
            [self.get(0, 0), self.get(0, 1)],
            [self.get(1, 0), self.get(1, 1)],
        ]
    }

    fn eye() -> Self;
    fn dagger(&self) -> Self;

    /// General 2×2 product against ANY other C2x2. Output is always DenseC2x2 —
    /// the product of two arbitrary 2×2 matrices has no preserved structure.
    /// NOTE: calling this for `Su2 × Su2` gives the slow 8-mult dense path. For
    /// that case use the `*` operator, which dispatches to the fast SU(2) kernel.
    #[inline(always)]
    fn matmul<R: C2x2>(&self, rhs: &R) -> DenseC2x2 {
        let l00 = self.get_direct::<0, 0>();
        let l01 = self.get_direct::<0, 1>();
        let l10 = self.get_direct::<1, 0>();
        let l11 = self.get_direct::<1, 1>();
        let r00 = rhs.get_direct::<0, 0>();
        let r01 = rhs.get_direct::<0, 1>();
        let r10 = rhs.get_direct::<1, 0>();
        let r11 = rhs.get_direct::<1, 1>();
        DenseC2x2::new([
            [l00 * r00 + l01 * r10, l00 * r01 + l01 * r11],
            [l10 * r00 + l11 * r10, l10 * r01 + l11 * r11],
        ])
    }

    /// Element-wise sum against any C2x2. Always DenseC2x2 (Su2 is not closed
    /// under addition: det(A + B) ≠ 1 in general).
    #[inline(always)]
    fn matadd<R: C2x2>(&self, rhs: &R) -> DenseC2x2 {
        DenseC2x2::new([
            [
                self.get_direct::<0, 0>() + rhs.get_direct::<0, 0>(),
                self.get_direct::<0, 1>() + rhs.get_direct::<0, 1>(),
            ],
            [
                self.get_direct::<1, 0>() + rhs.get_direct::<1, 0>(),
                self.get_direct::<1, 1>() + rhs.get_direct::<1, 1>(),
            ],
        ])
    }

    /// Element-wise difference against any C2x2. Always DenseC2x2.
    #[inline(always)]
    fn matsub<R: C2x2>(&self, rhs: &R) -> DenseC2x2 {
        DenseC2x2::new([
            [
                self.get_direct::<0, 0>() - rhs.get_direct::<0, 0>(),
                self.get_direct::<0, 1>() - rhs.get_direct::<0, 1>(),
            ],
            [
                self.get_direct::<1, 0>() - rhs.get_direct::<1, 0>(),
                self.get_direct::<1, 1>() - rhs.get_direct::<1, 1>(),
            ],
        ])
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DenseC2x2 {
    inner: [[Complex64; 2]; 2],
}
impl DenseC2x2 {
    pub fn new(a: [[Complex64; 2]; 2]) -> Self {
        DenseC2x2 { inner: a }
    }

    pub fn empty() -> Self {
        DenseC2x2 {
            inner: [[(0.).into(); 2]; 2],
        }
    }

    #[inline(always)]
    pub fn at(&mut self, i: usize, j: usize) -> &mut Complex64 {
        &mut self.inner[i][j]
    }

    pub fn transpose(&self) -> Self {
        DenseC2x2::new([
            [self.get(0, 0), self.get(1, 0)],
            [self.get(0, 1), self.get(1, 1)],
        ])
    }

    pub fn conj_self(&mut self) {
        for i in 0..2 {
            for j in 0..2 {
                *self.at(i, j) = self.get(i, j).conj();
            }
        }
    }

    pub fn conj(&self) -> DenseC2x2 {
        let mut c = self.clone();
        c.conj_self();
        c
    }

    pub fn l1_norm(&self) -> f64 {
        let mut r = 0.;
        for i in 0..2 {
            for j in 0..2 {
                r += self.get(i, j).abs();
            }
        }
        r
    }
}
impl C2x2 for DenseC2x2 {
    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> Complex64 {
        self.inner[i][j]
    }

    #[inline(always)]
    fn get_direct<const I: usize, const J: usize>(&self) -> Complex64 {
        const { assert!(I < 2 && J < 2, "C2x2 index out of range") };
        match (I, J) {
            (0, 0) => self.inner[0][0],
            (0, 1) => self.inner[0][1],
            (1, 0) => self.inner[1][0],
            (1, 1) => self.inner[1][1],
            _ => panic!("C2x2 index out of range"),
        }
    }

    fn eye() -> Self {
        DenseC2x2 {
            inner: [[(1.).into(), (0.).into()], [(0.).into(), (1.).into()]],
        }
    }

    fn dagger(&self) -> Self {
        let mut c = self.transpose();
        c.conj_self();
        c
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Su2 {
    a: Complex64,
    b: Complex64,
}
impl Su2 {
    #[inline(always)]
    pub fn from_dense(m: &DenseC2x2) -> Self {
        let a = m.get_direct::<0, 0>();
        let b = m.get_direct::<0, 1>();
        debug_assert!(
            (m.get_direct::<1, 0>() + b.conj()).norm() < 1e-9
                && (m.get_direct::<1, 1>() - a.conj()).norm() < 1e-9,
            "from_dense called on a non-SU(2) matrix",
        );
        Self { a, b }
    }

    #[inline(always)]
    pub fn from_ab(a: Complex64, b: Complex64) -> Self {
        Self { a, b }
    }

    #[inline(always)]
    pub fn z_rotation(phi: f64) -> Su2 {
        let (s, c) = phi.sin_cos();
        Su2::from_ab(Complex64::new(c, s), (0.).into())
    }

    #[inline(always)]
    pub fn x_rotation(phi: f64) -> Su2 {
        let (s, c) = phi.sin_cos();
        Su2::from_ab(Complex64::new(c, 0.), Complex64::new(0., s))
    }

    #[inline(always)]
    pub fn c00(&self) -> Complex64 {
        self.a
    }
    #[inline(always)]
    pub fn c01(&self) -> Complex64 {
        self.b
    }
    #[inline(always)]
    pub fn c10(&self) -> Complex64 {
        -self.b.conj()
    }
    #[inline(always)]
    pub fn c11(&self) -> Complex64 {
        self.a.conj()
    }
}
impl C2x2 for Su2 {
    fn get(&self, i: usize, j: usize) -> Complex64 {
        match (i, j) {
            (0, 0) => self.c00(),
            (0, 1) => self.c01(),
            (1, 0) => self.c10(),
            (1, 1) => self.c11(),
            _ => unreachable!("C2x2 index out of range: ({i}, {j})"),
        }
    }

    fn get_direct<const I: usize, const J: usize>(&self) -> Complex64 {
        const { assert!(I < 2 && J < 2, "C2x2 index out of range") };
        match (I, J) {
            (0, 0) => self.a,
            (0, 1) => self.b,
            (1, 0) => -self.b.conj(),
            (1, 1) => self.a.conj(),
            _ => unreachable!("C2x2 index out of range: ({I}, {J})"),
        }
    }

    fn eye() -> Self {
        Su2 {
            a: Complex64::new(1.0, 0.0),
            b: Complex64::new(0.0, 0.0),
        }
    }

    fn dagger(&self) -> Self {
        Su2 {
            a: self.a.conj(),
            b: -self.b,
        }
    }
}

impl Mul<DenseC2x2> for DenseC2x2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn mul(self, rhs: DenseC2x2) -> DenseC2x2 {
        self.matmul(&rhs)
    }
}
impl Mul<Su2> for Su2 {
    type Output = Su2;

    fn mul(self, rhs: Su2) -> Self::Output {
        Su2 {
            a: self.a * rhs.a - self.b * rhs.b.conj(),
            b: self.a * rhs.b + self.b * rhs.a.conj(),
        }
    }
}
impl Mul<DenseC2x2> for Su2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn mul(self, rhs: DenseC2x2) -> DenseC2x2 {
        self.matmul(&rhs)
    }
}
impl Mul<Su2> for DenseC2x2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn mul(self, rhs: Su2) -> DenseC2x2 {
        self.matmul(&rhs)
    }
}

impl Add<Su2> for Su2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn add(self, rhs: Su2) -> DenseC2x2 {
        self.matadd(&rhs)
    }
}
impl Add<DenseC2x2> for Su2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn add(self, rhs: DenseC2x2) -> DenseC2x2 {
        self.matadd(&rhs)
    }
}
impl Add<Su2> for DenseC2x2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn add(self, rhs: Su2) -> DenseC2x2 {
        self.matadd(&rhs)
    }
}
impl Add<DenseC2x2> for DenseC2x2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn add(self, rhs: DenseC2x2) -> DenseC2x2 {
        self.matadd(&rhs)
    }
}

impl Sub<Su2> for Su2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn sub(self, rhs: Su2) -> DenseC2x2 {
        self.matsub(&rhs)
    }
}
impl Sub<DenseC2x2> for Su2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn sub(self, rhs: DenseC2x2) -> DenseC2x2 {
        self.matsub(&rhs)
    }
}
impl Sub<Su2> for DenseC2x2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn sub(self, rhs: Su2) -> DenseC2x2 {
        self.matsub(&rhs)
    }
}
impl Sub<DenseC2x2> for DenseC2x2 {
    type Output = DenseC2x2;
    #[inline(always)]
    fn sub(self, rhs: DenseC2x2) -> DenseC2x2 {
        self.matsub(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

    // ── helpers ────────────────────────────────────────────────────────────

    const TOL: f64 = 1e-13;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// Assert two complex numbers agree to TOL (absolute, on each component).
    fn assert_c(got: Complex64, want: Complex64, ctx: &str) {
        let d = (got - want).norm();
        assert!(d <= TOL, "{ctx}: got {got:?}, want {want:?}, |Δ|={d:.3e}",);
    }

    /// Assert two C2x2 values are entrywise equal to TOL.
    fn assert_eq_mat<L: C2x2, R: C2x2>(got: &L, want: &R, ctx: &str) {
        for i in 0..2 {
            for j in 0..2 {
                let d = (got.get(i, j) - want.get(i, j)).norm();
                assert!(
                    d <= TOL,
                    "{ctx} at ({i},{j}): got {:?}, want {:?}, |Δ|={d:.3e}",
                    got.get(i, j),
                    want.get(i, j),
                );
            }
        }
    }

    /// A random SU(2) element, returned in BOTH representations so tests can
    /// cross-check. Construction: a, b random complex, normalized so |a|²+|b|²=1.
    /// Dense form is the literal [[a, b], [-b̄, ā]] so it's a true SU(2) matrix.
    fn random_su2_pair(rng: &mut StdRng) -> (Su2, DenseC2x2) {
        let a = c(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
        let b = c(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
        let norm = (a.norm_sqr() + b.norm_sqr()).sqrt();
        let a = a / norm;
        let b = b / norm;
        let su2 = Su2::from_ab(a, b); // assumes a constructor; see note below
        let dense = DenseC2x2::new([[a, b], [-b.conj(), a.conj()]]);
        (su2, dense)
    }

    /// A fully arbitrary (non-SU(2)) dense matrix, for testing the general
    /// product/add/sub paths where no structure is assumed.
    fn random_dense(rng: &mut StdRng) -> DenseC2x2 {
        let mut g = || c(rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0));
        DenseC2x2::new([[g(), g()], [g(), g()]])
    }

    fn rng() -> StdRng {
        StdRng::seed_from_u64(0xC0FFEE)
    }

    // ── 1. Su2 accessor / reconstruction correctness ───────────────────────

    #[test]
    fn su2_accessors_match_dense_entries() {
        let mut r = rng();
        for _ in 0..200 {
            let (s, d) = random_su2_pair(&mut r);
            assert_c(s.c00(), d.get(0, 0), "c00");
            assert_c(s.c01(), d.get(0, 1), "c01");
            assert_c(s.c10(), d.get(1, 0), "c10"); // -b̄
            assert_c(s.c11(), d.get(1, 1), "c11"); // ā
        }
    }

    #[test]
    fn su2_get_matches_get_direct_and_accessors() {
        let mut r = rng();
        let (s, _) = random_su2_pair(&mut r);
        // runtime get vs compile-time get_direct vs named accessor: all three agree
        assert_c(s.get(0, 0), s.get_direct::<0, 0>(), "get vs direct 00");
        assert_c(s.get(0, 1), s.get_direct::<0, 1>(), "get vs direct 01");
        assert_c(s.get(1, 0), s.get_direct::<1, 0>(), "get vs direct 10");
        assert_c(s.get(1, 1), s.get_direct::<1, 1>(), "get vs direct 11");
        assert_c(s.get_direct::<0, 0>(), s.c00(), "direct vs accessor 00");
        assert_c(s.get_direct::<1, 0>(), s.c10(), "direct vs accessor 10");
    }

    #[test]
    fn dense_get_matches_get_direct() {
        let mut r = rng();
        let d = random_dense(&mut r);
        assert_c(d.get(0, 0), d.get_direct::<0, 0>(), "00");
        assert_c(d.get(0, 1), d.get_direct::<0, 1>(), "01");
        assert_c(d.get(1, 0), d.get_direct::<1, 0>(), "10");
        assert_c(d.get(1, 1), d.get_direct::<1, 1>(), "11");
    }

    #[test]
    fn entries_default_method_matches_get() {
        let mut r = rng();
        let (s, d) = random_su2_pair(&mut r);
        let es = s.entries();
        let ed = d.entries();
        for i in 0..2 {
            for j in 0..2 {
                assert_c(es[i][j], s.get(i, j), "su2 entries");
                assert_c(ed[i][j], d.get(i, j), "dense entries");
            }
        }
    }

    // ── 2. eye / dagger ────────────────────────────────────────────────────

    #[test]
    fn eye_is_identity_both_reprs() {
        let se = <Su2 as C2x2>::eye();
        let de = <DenseC2x2 as C2x2>::eye();
        assert_eq_mat(&se, &de, "eye su2 vs dense");
        assert_c(se.get(0, 0), c(1.0, 0.0), "eye 00");
        assert_c(se.get(0, 1), c(0.0, 0.0), "eye 01");
        assert_c(se.get(1, 0), c(0.0, 0.0), "eye 10");
        assert_c(se.get(1, 1), c(1.0, 0.0), "eye 11");
    }

    #[test]
    fn su2_eye_is_mul_identity() {
        let mut r = rng();
        let e = <Su2 as C2x2>::eye();
        for _ in 0..100 {
            let (s, _) = random_su2_pair(&mut r);
            assert_eq_mat(&(e * s), &s, "eye * s");
            assert_eq_mat(&(s * e), &s, "s * eye");
        }
    }

    #[test]
    fn su2_dagger_matches_dense_dagger() {
        let mut r = rng();
        for _ in 0..200 {
            let (s, d) = random_su2_pair(&mut r);
            assert_eq_mat(&s.dagger(), &d.dagger(), "dagger agreement");
        }
    }

    #[test]
    fn su2_dagger_is_inverse() {
        let mut r = rng();
        let e = <Su2 as C2x2>::eye();
        for _ in 0..200 {
            let (s, _) = random_su2_pair(&mut r);
            // U U† = U† U = I  (the defining SU(2) property)
            assert_eq_mat(&(s * s.dagger()), &e, "U U†");
            assert_eq_mat(&(s.dagger() * s), &e, "U† U");
        }
    }

    #[test]
    fn dagger_involution() {
        let mut r = rng();
        for _ in 0..100 {
            let (s, d) = random_su2_pair(&mut r);
            assert_eq_mat(&s.dagger().dagger(), &s, "su2 (U†)† = U");
            assert_eq_mat(&d.dagger().dagger(), &d, "dense (U†)† = U");
        }
    }

    // ── 3. Su2 fast Mul matches the dense oracle ───────────────────────────

    #[test]
    fn su2_mul_matches_dense_mul() {
        let mut r = rng();
        for _ in 0..500 {
            let (sa, da) = random_su2_pair(&mut r);
            let (sb, db) = random_su2_pair(&mut r);
            // fast 4-mult SU(2) kernel vs general 8-mult dense product
            assert_eq_mat(&(sa * sb), &(da * db), "su2*su2 vs dense*dense");
        }
    }

    #[test]
    fn su2_product_stays_su2() {
        // Closure: product of two SU(2) is SU(2), i.e. still satisfies U U† = I.
        let mut r = rng();
        let e = <Su2 as C2x2>::eye();
        for _ in 0..200 {
            let (sa, _) = random_su2_pair(&mut r);
            let (sb, _) = random_su2_pair(&mut r);
            let p = sa * sb;
            assert_eq_mat(&(p * p.dagger()), &e, "product is unitary");
        }
    }

    #[test]
    fn su2_mul_associative() {
        let mut r = rng();
        for _ in 0..200 {
            let (a, _) = random_su2_pair(&mut r);
            let (b, _) = random_su2_pair(&mut r);
            let (cc, _) = random_su2_pair(&mut r);
            assert_eq_mat(&((a * b) * cc), &(a * (b * cc)), "associativity");
        }
    }

    // ── 4. Generalized matmul/matadd/matsub default methods ────────────────

    #[test]
    fn matmul_method_matches_operator_dense() {
        let mut r = rng();
        for _ in 0..200 {
            let x = random_dense(&mut r);
            let y = random_dense(&mut r);
            assert_eq_mat(&x.matmul(&y), &(x * y), "matmul == * for dense");
        }
    }

    #[test]
    fn matmul_method_su2_matches_fast_kernel() {
        // The footgun case: su2.matmul(&su2) takes the slow dense path but must
        // still produce the SAME value as the fast operator.
        let mut r = rng();
        for _ in 0..200 {
            let (sa, _) = random_su2_pair(&mut r);
            let (sb, _) = random_su2_pair(&mut r);
            assert_eq_mat(&sa.matmul(&sb), &(sa * sb), "matmul vs fast *");
        }
    }

    #[test]
    fn matadd_matsub_methods_match_manual() {
        let mut r = rng();
        for _ in 0..200 {
            let x = random_dense(&mut r);
            let y = random_dense(&mut r);
            let sum = x.matadd(&y);
            let dif = x.matsub(&y);
            for i in 0..2 {
                for j in 0..2 {
                    assert_c(sum.get(i, j), x.get(i, j) + y.get(i, j), "matadd");
                    assert_c(dif.get(i, j), x.get(i, j) - y.get(i, j), "matsub");
                }
            }
        }
    }

    // ── 5. Every operator pair × {Mul, Add, Sub} ───────────────────────────
    //
    // For each combination, the result must equal the dense-oracle computation.
    // We compare against DenseC2x2 built from the same entries, so this checks
    // both correctness AND that the Output type carries the right values.

    #[test]
    fn mul_all_pairs() {
        let mut r = rng();
        for _ in 0..300 {
            let (sa, da) = random_su2_pair(&mut r);
            let (sb, db) = random_su2_pair(&mut r);
            let xa = random_dense(&mut r);
            let xb = random_dense(&mut r);

            // Su2 × Su2  (Output = Su2, fast path)
            assert_eq_mat(&(sa * sb), &(da * db), "Su2×Su2");
            // Su2 × Dense (Output = Dense)
            assert_eq_mat(&(sa * xb), &(da * xb), "Su2×Dense");
            // Dense × Su2 (Output = Dense)
            assert_eq_mat(&(xa * sb), &(xa * db), "Dense×Su2");
            // Dense × Dense
            assert_eq_mat(&(xa * xb), &xa.matmul(&xb), "Dense×Dense");
        }
    }

    #[test]
    fn add_all_pairs() {
        let mut r = rng();
        for _ in 0..300 {
            let (sa, da) = random_su2_pair(&mut r);
            let (sb, db) = random_su2_pair(&mut r);
            let xa = random_dense(&mut r);
            let xb = random_dense(&mut r);

            assert_eq_mat(&(sa + sb), &da.matadd(&db), "Su2+Su2");
            assert_eq_mat(&(sa + xb), &da.matadd(&xb), "Su2+Dense");
            assert_eq_mat(&(xa + sb), &xa.matadd(&db), "Dense+Su2");
            assert_eq_mat(&(xa + xb), &xa.matadd(&xb), "Dense+Dense");
        }
    }

    #[test]
    fn sub_all_pairs() {
        let mut r = rng();
        for _ in 0..300 {
            let (sa, da) = random_su2_pair(&mut r);
            let (sb, db) = random_su2_pair(&mut r);
            let xa = random_dense(&mut r);
            let xb = random_dense(&mut r);

            assert_eq_mat(&(sa - sb), &da.matsub(&db), "Su2-Su2");
            assert_eq_mat(&(sa - xb), &da.matsub(&xb), "Su2-Dense");
            assert_eq_mat(&(xa - sb), &xa.matsub(&db), "Dense-Su2");
            assert_eq_mat(&(xa - xb), &xa.matsub(&xb), "Dense-Dense");
        }
    }

    // ── 6. Cross-representation consistency on a mixed expression ───────────

    #[test]
    fn mixed_expression_su2_vs_dense() {
        // Evaluate the same nontrivial expression two ways: once keeping SU(2)
        // factors compact, once all-dense. A · B · A† + (B - A), say.
        let mut r = rng();
        for _ in 0..200 {
            let (a, da) = random_su2_pair(&mut r);
            let (b, db) = random_su2_pair(&mut r);

            // compact: a*b stays Su2, *a.dagger() stays Su2, then + widens to Dense
            let lhs = (a * b * a.dagger()) + (b.matsub(&a));
            // oracle: everything dense
            let rhs = (da * db * da.dagger()).matadd(&db.matsub(&da));

            assert_eq_mat(&lhs, &rhs, "mixed expr");
        }
    }

    // ── 7. get_direct compile-time bound (doc/compile_fail) ─────────────────
    //
    // get_direct::<2, 0>() must NOT compile (const assert fires at build time).
    // Expressed as a doctest on the trait method; shown here for completeness.
    //
    // ```compile_fail
    // use crate::compute::c2x2::{C2x2, Su2};
    // let s = Su2::eye();
    // let _ = s.get_direct::<2, 0>(); // const { assert!(I < 2 && J < 2) } fails
    // ```
}
