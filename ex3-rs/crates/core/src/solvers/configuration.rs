use anyhow::Result;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, concatenate};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhaseGenerator {
    Random { magnitude: f64, seed: Option<u64> },
    Fixed(Array1<f64>),
    Zeros,
}

impl PhaseGenerator {
    pub fn get(&self, n: usize) -> Array1<f64> {
        match self {
            PhaseGenerator::Random { magnitude, seed } => {
                let mut rng = StdRng::seed_from_u64(if let Some(s) = seed {
                    *s
                } else {
                    rand::random::<u64>()
                });
                (0..n).map(|_| rng.random_range(0.0..*magnitude)).collect()
            }
            PhaseGenerator::Fixed(a) => (0..n).map(|i| a[i % a.len()]).collect(),
            PhaseGenerator::Zeros => Array1::zeros(n),
        }
    }

    pub fn resize(&self, phases: &mut Array1<f64>, new_len: usize) {
        if new_len < phases.len() {
            *phases = phases.iter().take(new_len).map(|f| *f).collect();
        } else if new_len > phases.len() {
            *phases = concatenate![
                Axis(0),
                *phases,
                match self {
                    PhaseGenerator::Fixed(_) => PhaseGenerator::Zeros.get(new_len - phases.len()),
                    _ => self.get(new_len - phases.len()),
                }
            ];
        }
    }
}

impl FromStr for PhaseGenerator {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        let trimmed = s.trim();

        let parts: Vec<&str> = trimmed.split(",").collect();
        match parts.as_slice() {
            ["rand" | "r" | "random", f] => Ok(PhaseGenerator::Random {
                magnitude: f64::from_str(*f)?,
                seed: None,
            }),
            ["rand" | "r" | "random", f, s] => Ok(PhaseGenerator::Random {
                magnitude: f64::from_str(*f)?,
                seed: Some(u64::from_str(*s)?),
            }),
            ["zeros" | "zero" | "0" | "empty" | "z"] => Ok(PhaseGenerator::Zeros),
            [first, ..]
                if matches!(
                    *first,
                    "rand" | "r" | "random" | "zeros" | "zero" | "0" | "empty" | "z"
                ) =>
            {
                anyhow::bail!(format!("Wrong number of arguments for '{}'", first))
            }
            _ => Ok(PhaseGenerator::Fixed(
                trimmed
                    .split(',')
                    .map(|part| {
                        let cleaned = part.replace(char::is_whitespace, "");
                        cleaned.parse::<f64>()
                    })
                    .collect::<Result<_, _>>()?,
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PhaseMap {
    /// pass on the phases as-is
    None,
    /// Mirrors the phases around the middle phase,
    /// with the first phase receiving a pi/4 kick
    /// as described in ref[1]. This will effectively
    /// double the number of phases
    Mirror,
    MirrorIfPossible,
}

impl FromStr for PhaseMap {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "none" | "n" => Ok(Self::None),
            "mirror" | "m" => Ok(Self::Mirror),
            "auto" | "mirror-if-possible" | "mirror_if_possible" | "o" | "a" => {
                Ok(Self::MirrorIfPossible)
            }
            _ => anyhow::bail!(format!("Could not convert {s} into PhaseMap type!")),
        }
    }
}

use ndarray::{Data, Dimension, IntoDimension};
use std::{f64::consts::PI, ops::AddAssign, str::FromStr};

use crate::{
    target::{Parity, TargetPoly},
    utils::parse_usize_gt_0,
};

fn fold<A, S, D>(a: &ArrayBase<S, D>) -> Array<A, D>
where
    A: Clone + Default + AddAssign,
    S: Data<Elem = A>,
    D: Dimension,
{
    let n = a.raw_dim();

    let mut half = n.clone();
    for (h, &len) in half.slice_mut().iter_mut().zip(n.slice()) {
        *h = (len + 1) / 2;
    }
    let mut out = Array::<A, D>::from_elem(half, A::default());

    for (idx, val) in a.indexed_iter() {
        let mut o = idx.into_dimension();
        for (c, &len) in o.slice_mut().iter_mut().zip(n.slice()) {
            *c = (*c).min(len - 1 - *c);
        }
        out[o] += val.clone();
    }
    out
}

fn fold_axis<A, S, D>(a: &ArrayBase<S, D>, axis: Axis) -> Array<A, D>
where
    A: Clone + Default + AddAssign,
    S: Data<Elem = A>,
    D: Dimension,
{
    let n = a.raw_dim();
    let ax = axis.index();
    let len = n.slice()[ax];
    let mut half = n.clone();
    half.slice_mut()[ax] = (len + 1) / 2;
    let mut out = Array::<A, D>::from_elem(half, A::default());
    for (idx, val) in a.indexed_iter() {
        let mut o = idx.into_dimension();
        let c = o.slice()[ax];
        o.slice_mut()[ax] = c.min(len - 1 - c);
        out[o] += val.clone();
    }
    out
}

impl PhaseMap {
    fn does_mirror(&self, t: &TargetPoly) -> Result<bool> {
        match (self, t.all_real()) {
            (PhaseMap::None, _) | (PhaseMap::MirrorIfPossible, false) => Ok(false),
            (PhaseMap::Mirror, true) | (PhaseMap::MirrorIfPossible, true) => Ok(true),
            (PhaseMap::Mirror, false) => {
                anyhow::bail!(format!("Cannot do mirror if target isn't real!"))
            }
        }
    }

    pub fn apply(&self, phase_in: &mut Array1<f64>, t: &TargetPoly) -> Result<()> {
        match self.does_mirror(t)? {
            false => Ok(()),
            true => {
                // idk if there is a nice way to do without double copy
                let n = phase_in.len();
                let copy = Array1::from_iter(
                    phase_in
                        .iter()
                        // keep the parity of the phase array
                        .take(match (t.get_parity(), n % 2 == 0) {
                            (Some(Parity::Odd), _) | (None, true) => n,
                            (Some(Parity::Even), _) | (None, false) => n - 1,
                        })
                        .rev()
                        .map(|p| *p),
                );
                phase_in.append(Axis(0), copy.view())?;
                phase_in[0] += PI / 2.;
                Ok(())
            }
        }
    }

    pub fn fold<D: Dimension>(&self, a: &mut Array<f64, D>, t: &TargetPoly) -> Result<()> {
        match self.does_mirror(t)? {
            false => Ok(()),
            true => {
                *a = fold(a);
                Ok(())
            }
        }
    }

    pub fn fold_jacobian(&self, a: &mut Array2<f64>, t: &TargetPoly) -> Result<()> {
        match self.does_mirror(t)? {
            false => Ok(()),
            true => {
                *a = fold_axis(a, Axis(1));
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SolveMode {
    /// Direct solve at the given degree.
    Simple(usize),
    /// Solve at the first degree, then warm-start a solve at the second.
    /// Constraint: first < second.
    Hotstart(usize, usize),
    /// N cascading solves, gradually increasing degree from a small initial value
    /// up to the final degree, each warm-starting from the previous.
    Cascade(usize, usize),
}

impl FromStr for SolveMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        let trimmed = s.trim();

        let parts: Vec<&str> = trimmed.split(",").collect();
        match parts.as_slice() {
            ["simple" | "s", n] => Ok(SolveMode::Simple(parse_usize_gt_0(n, "simple")?)),
            ["hotstart" | "h", f, s] => Ok(SolveMode::Hotstart(
                parse_usize_gt_0(f, "hotstart")?,
                parse_usize_gt_0(s, "hotstart")?,
            )),
            ["cascade" | "c", s, f] => Ok(SolveMode::Cascade(
                parse_usize_gt_0(f, "cascade")?,
                parse_usize_gt_0(s, "cascade")?,
            )),
            _ => anyhow::bail!(format!("Could not parse solve mode: '{s}'")),
        }
    }
}
impl SolveMode {
    pub fn rescale(self, d: usize) -> Self {
        match self {
            SolveMode::Simple(_) => SolveMode::Simple(d),
            SolveMode::Hotstart(d1, d2) => {
                SolveMode::Hotstart(((d1 as f64) / (d2 as f64) * (d as f64)) as usize, d)
            }
            SolveMode::Cascade(s, f) => {
                SolveMode::Cascade(((s as f64) / (f as f64) * (d as f64)) as usize, d)
            }
        }
    }
}
