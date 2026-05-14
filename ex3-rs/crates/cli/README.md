# qsp-rs-cli

Command-line interface for [qsp-rs](../..).

## Build

```bash
cargo build --release -p qsp-rs-cli
# binary at target/release/qsp-rs-cli
```

Or run directly from the workspace root:

```bash
cargo run --release -p qsp-rs-cli -- <subcommand> [args]
```

## Subcommands

| Command            | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `solve-poly`       | Solve for phases that produce a given target polynomial.   |
| `plot-runtimes`    | Benchmark solver performance across problem sizes.         |
| `get-least-pulses` | Find the minimum degree at which the solver fits a target. |

Run `qsp-rs-cli <subcommand> --help` for full flag listings.

## Examples

Solve a random real target, mirrored to length 60 from 30 half-points:

```bash
qsp-rs solve-poly -n 30 -t rand -p even
```

Use Levenberg–Marquardt instead of the default L-BFGS, on a generalized-parity target:

```bash
qsp-rs --solver lm solve-poly -n 40 -t "gp,2,3" -p odd
```

Cascade up to degree 80, write phases + plot data to a file:

```bash
qsp-rs --mode cascade,4,80 solve-poly -n 40 -o result.dat
```

## Usage

```
Fit a QSP polynomial to the given sequence of target points.

Usage: qsp-rs [OPTIONS] <COMMAND>

Commands:
  solve-poly
  plot-runtimes
  get-least-pulses
  help              Print this message or the help of the given subcommand(s)

Options:
  -m, --backend-mode <BACKEND_MODE>  Enable/disable multithreading for gradient, lossfunction evaluation. Auto: will do single threading for small d & short sequences. (both <= 100) [default: auto] [possible values: auto, single-thread, multi-thread]
  -S, --solver <KIND>                Which optimizer to use [default: bfgs] [possible values: bfgs, lm]
  -h, --help                         Print help (see more with '--help')

L-BFGS Options:
      --bfgs-max-iters <MAX_ITERS>  [default: 500000]
      --bfgs-mem <MEM>              [default: 10]
      --bfgs-tol-grad <TOL_GRAD>    [default: 0.00000001]

Levenberg-Marquardt Options:
      --lm-max-iters <lm_max_iters>            [default: 500]
      --lm-initial-lambda <lm_initial_lambda>  [default: 1e-4]
      --lm-tol <lm_tol>                        [default: 1e-10]

General solver options:
  -M, --mode <MODE>
          Solve mode: "simple,D" — direct solve at degree D "hotstart,S,D" — solve at degree S, then continue at degree D "cascade,N,D" — N cascading steps up to degree D if running PlotRuntimes task, this will be interpreted as a ratio and scaled accordingly [default: hotstart,20,60]
  -P, --phase-map <PHASE_MAP>
          Maps the phases before the QSP unitary is constructed. Use mirror for polynomial targets that are real. Mirror will double the effective parameter count, the parameter count given by --mode is the number of parameters used by the optimizer, but (if mirrored) only half the amount of phases used to construct the qsp unitary [default: mirror-if-possible] [possible values: none, mirror, mirror-if-possible]
  -i, --init-perturb-mag <INIT_PERTURB_MAG>
          The max. magnitude of the first guess for the random phases to initialize the solver. Choose 0 for mirrored phases to get faster convergance and lower total phase values [default: .4]

Target configuration:
  -t, --target-pattern <TARGET_PATTERN>
          complex numbers with mod <= 1 seperated by commas (eg "0.5+0.3i, -0.2-0.7i, 0.1i") that will be repeated up to target length "rand": random points that are either 1 or 0 (which are then mirrored, see parity) "rand-phase" random points that are like exp(iφ) with random φ ∈ [0, 2π) "gp,r,k" generalized parity target with r,k [default: rand]
  -p, --parity <PARITY>
          parity ("even" or "odd") [default: even] [possible values: even, odd]
  -l, --label <LABEL>
```
