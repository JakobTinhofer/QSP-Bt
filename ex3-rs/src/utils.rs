pub fn parse_usize_gt_0(s: &str, m: &str) -> Result<usize, String> {
    let n: usize = s
        .trim()
        .parse()
        .map_err(|_| format!("{m}: '{s}' ist an invalid usize!"))?;
    if n == 0 {
        Err(format!("{m}: size must be > 0"))
    } else {
        Ok(n)
    }
}
