use anyhow::Result;
pub fn parse_usize_gt_0(s: &str, m: &str) -> Result<usize> {
    let n: usize = s.trim().parse()?;
    if n == 0 {
        anyhow::bail!(format!("{m}: size must be > 0"))
    } else {
        Ok(n)
    }
}
