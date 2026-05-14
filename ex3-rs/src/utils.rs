use anyhow::Result;
use std::time::Duration;
pub fn parse_usize_gt_0(s: &str, m: &str) -> Result<usize> {
    let n: usize = s.trim().parse()?;
    if n == 0 {
        anyhow::bail!(format!("{m}: size must be > 0"))
    } else {
        Ok(n)
    }
}

pub fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();

    // Sub-second: show as ms or smaller
    if total_secs == 0 {
        let ms = d.as_secs_f64() * 1000.0;
        if ms >= 1.0 {
            // Trim trailing zeros: 1.500ms → 1.5ms, 60.000ms → 60ms
            return format!("{}ms", trim_float(ms, 3));
        }
        let us = d.as_secs_f64() * 1_000_000.0;
        if us >= 1.0 {
            return format!("{}µs", trim_float(us, 1));
        }
        return format!("{}ns", d.as_nanos());
    }

    // 1s to <60s: show with millisecond precision
    if total_secs < 60 {
        let secs = d.as_secs_f64();
        return format!("{}s", trim_float(secs, 3));
    }

    // ≥1 minute: integer seconds, structured units
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = (total_secs / 3600) % 24;
    let days = total_secs / 86400;

    if days > 0 {
        format!("{days}d{hours:02}h{mins:02}m{secs:02}s")
    } else if hours > 0 {
        format!("{hours}h{mins:02}m{secs:02}s")
    } else {
        format!("{mins}m{secs:02}s")
    }
}

/// Format a float with up to `decimals` places, trimming trailing zeros.
/// 1.500 (3) → "1.5", 60.000 (3) → "60", 1.234 (3) → "1.234"
fn trim_float(value: f64, decimals: usize) -> String {
    let s = format!("{value:.*}", decimals);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}
