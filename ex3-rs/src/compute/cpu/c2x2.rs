use std::ops::{Add, Mul, Sub};

use num_complex::{Complex64, ComplexFloat};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct C2x2 {
    inner: [[Complex64; 2]; 2],
}

impl C2x2 {
    pub fn new(a: [[Complex64; 2]; 2]) -> Self {
        C2x2 { inner: a }
    }

    pub fn eye() -> Self {
        C2x2 {
            inner: [[(1.).into(), (0.).into()], [(0.).into(), (1.).into()]],
        }
    }

    pub fn empty() -> Self {
        C2x2 {
            inner: [[(0.).into(); 2]; 2],
        }
    }

    pub fn at(&mut self, i: usize, j: usize) -> &mut Complex64 {
        assert!(i < 2 && j < 2, "Invalid index!");
        &mut self.inner[i][j]
    }

    pub fn get(&self, i: usize, j: usize) -> Complex64 {
        assert!(i < 2 && j < 2, "Invalid index!");
        self.inner[i][j]
    }

    pub fn transpose(&self) -> Self {
        C2x2::new([
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

    pub fn conj(&self) -> C2x2 {
        let mut c = self.clone();
        c.conj_self();
        c
    }

    pub fn dagger(&self) -> C2x2 {
        let mut c = self.transpose();
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

impl Add<C2x2> for C2x2 {
    type Output = C2x2;

    fn add(self, rhs: C2x2) -> Self::Output {
        let mut c: C2x2 = C2x2::empty();
        for i in 0..2 {
            for j in 0..2 {
                *c.at(i, j) = self.get(i, j) + rhs.get(i, j);
            }
        }
        c
    }
}

impl Sub<C2x2> for C2x2 {
    type Output = C2x2;

    fn sub(self, rhs: C2x2) -> Self::Output {
        let mut c: C2x2 = C2x2::empty();
        for i in 0..2 {
            for j in 0..2 {
                *c.at(i, j) = self.get(i, j) - rhs.get(i, j);
            }
        }
        c
    }
}

impl Mul<C2x2> for C2x2 {
    type Output = C2x2;

    fn mul(self, rhs: C2x2) -> Self::Output {
        let mut c: C2x2 = C2x2::empty();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    *c.at(i, j) = c.get(i, j) + self.get(i, k) * rhs.get(k, j);
                }
            }
        }
        c
    }
}
