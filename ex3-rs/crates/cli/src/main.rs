use crate::{cli::Args, tasks::TaskTrait};
use anyhow::Result;
use clap::Parser;
mod cli;
mod data;
mod observe;
mod tasks;

fn main() -> Result<()> {
    let args = Args::parse();
    args.task.execute(args.config)
}
