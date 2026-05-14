use clap::Parser;
use ex3_rs::{cli::Args, tasks::TaskTrait};

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    args.task.execute(args.config)
}
