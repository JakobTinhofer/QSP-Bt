use clap::Parser;
use ndarray::Array1;
use num_complex::Complex64;
use rand::distr::{Distribution, Uniform};
use std::f64::consts::TAU;
use std::path::PathBuf;

use crate::solver::Parity;

#[derive(Parser)]
#[command(about = "Fit a QSP polynomial to the given sequence of target points.")]
pub struct Args {
    /// complex numbers with mod <= 1 seperated by commas (eg "0.5+0.3i, -0.2-0.7i, 0.1i") or
    /// "rand,n": initializes with n points that are either 1 or 0 (which are then mirrored, see parity)
    /// "rand-phase,n" initializes with n points that are like exp(iφ) with random φ ∈ [0, 2π)
    #[arg(value_parser = parse_target)]
    pub target_y: Array1<Complex64>,

    /// degree (>0)
    #[arg(short = 'd', long, value_parser = parse_positive, default_value_t=60)]
    pub degree: usize,

    /// hotstart-degree (>0)
    #[arg(short = 's', long, value_parser = parse_positive, default_value_t=20)]
    pub hotstart: usize,

    /// parity ("even" or "odd")
    #[arg(short = 'p', long, value_enum, default_value_t = Parity::Even)]
    pub parity: Parity,

    /// Path for the solution output
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Path for outputing the data formated to be drawn in gnuplot
    #[arg(short = 'D', long)]
    pub drawable: Option<PathBuf>,
    /*
    /// Will retry until error func is below this value. This might not happen for some polynomials. By default it will just run once and always succeed.
    #[arg(short = 't', long)]
    pub tolerance: Option<f64>,

    /// Will reseed for up to maxiter times as long as tolerance is not reached.
    #[arg(short = 'i', long, default_value_t = 10)]
    pub maxiter: usize,
    */
}

pub fn parse_positive(s: &str) -> Result<usize, String> {
    let n: usize = s
        .parse()
        .map_err(|_| format!("'{s}' ist keine gültige Ganzzahl"))?;
    if n == 0 {
        Err("Wert muss > 0 sein".into())
    } else {
        Ok(n)
    }
}

pub fn parse_count(s: &str, mode: &str) -> Result<usize, String> {
    let n: usize = s
        .trim()
        .parse()
        .map_err(|_| format!("{mode}: '{s}' ist keine gültige Anzahl"))?;
    if n == 0 {
        Err(format!("{mode}: Anzahl muss > 0 sein"))
    } else {
        Ok(n)
    }
}

pub fn parse_target(s: &str) -> Result<Array1<Complex64>, String> {
    let trimmed = s.trim();
    let mut rng = rand::rng();

    if let Some(rest) = trimmed.strip_prefix("rand-phase,") {
        let n = parse_count(rest, "rand-phase")?;
        let p_dist = Uniform::new(0., TAU).unwrap();
        let numbers: Array1<Complex64> = (0..n)
            .map(|_| {
                let phi: f64 = p_dist.sample(&mut rng);
                Complex64::from_polar(1.0, phi)
            })
            .collect();
        return Ok(numbers);
    }

    if let Some(rest) = trimmed.strip_prefix("rand,") {
        let n = parse_count(rest, "rand")?;
        let bool_dist = rand::distr::Bernoulli::new(0.5).unwrap();
        let numbers: Array1<Complex64> = (0..n)
            .map(|_| { if bool_dist.sample(&mut rng) { 1. } else { 0. } }.into())
            .collect();
        return Ok(numbers);
    }

    let numbers: Array1<Complex64> = trimmed
        .split(',')
        .map(|part| {
            let cleaned = part.replace(char::is_whitespace, "");
            cleaned
                .parse::<Complex64>()
                .map_err(|e| format!("'{}': {}", part.trim(), e))
        })
        .collect::<Result<_, _>>()?;

    for (i, z) in numbers.iter().enumerate() {
        if z.norm() > 1.0 {
            return Err(format!("Zahl {i} ({z}) hat Betrag {:.4} > 1", z.norm()));
        }
    }

    Ok(numbers)
}

pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const DIM: &str = "\x1b[2m";
pub const YELLOW: &str = "\x1b[33m";
pub const BLUE: &str = "\x1b[34m";
pub const CYAN: &str = "\x1b[36m";
pub const BOLD: &str = "\x1b[1m";
pub const RESET: &str = "\x1b[0m";

pub fn trim_zeros(value: f64, precision: usize) -> String {
    let s = format!("{:.prec$}", value, prec = precision);
    if s.contains('.') {
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        trimmed.to_string()
    } else {
        s
    }
}

pub fn format_complex_polar(z: &Complex64, precision: usize) -> String {
    let r = z.norm();
    let phi = z.arg();
    let i = format!("{YELLOW}{BOLD}i{RESET}");

    let eps = 10f64.powi(-(precision as i32));
    let r_str = trim_zeros(r, precision);

    if phi.abs() < eps {
        // φ ≈ 0 → kein Exponentialteil
        r_str
    } else {
        format!("{r_str} · exp({i}·{})", trim_zeros(phi, precision),)
    }
}

/// Formatiert ein ganzes Array, mehrzeilig oder einzeilig je nach Länge
pub fn format_array(arr: &Array1<Complex64>, precision: usize) -> String {
    let formatted: Vec<String> = arr
        .iter()
        .map(|z| format_complex_polar(z, precision))
        .collect();

    if arr.len() <= 6 {
        format!("[ {} ]", formatted.join(", "))
    } else {
        let mut s = String::from("[\n");
        for (i, item) in formatted.iter().enumerate() {
            s.push_str(&format!("  {DIM}{i:3}:{RESET} {item}\n"));
        }
        s.push(']');
        s
    }
}

fn format_real_compact(value: f64, precision: usize) -> String {
    if value >= 0.0 {
        format!(" {}", trim_zeros(value, precision))
    } else {
        format!(
            "{GREEN}{BOLD}-{RESET}{}",
            trim_zeros(value.abs(), precision)
        )
    }
}

/// Formatiert ein ganzes Array, mehrzeilig oder einzeilig je nach Länge
pub fn format_array_real(arr: &Array1<f64>, precision: usize) -> String {
    let formatted: Vec<String> = arr
        .iter()
        .map(|x| format_real_compact(*x, precision))
        .collect();

    if arr.len() <= 8 {
        format!("[{} ]", formatted.join(","))
    } else {
        let mut s = String::from("[\n");
        for (i, item) in formatted.iter().enumerate() {
            s.push_str(&format!("  {DIM}{i:3}:{RESET} {item}\n"));
        }
        s.push(']');
        s
    }
}
