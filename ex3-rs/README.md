TODO:

- get plot data for k plot
- more flexible polynome input

runs in <1s for poly degree of 60 using hotstart

```
Fit a QSP polynomial to the given sequence of target points.

Usage: ex3-rs [OPTIONS] <TARGET_Y>

Arguments:
  <TARGET_Y>  complex numbers with mod <= 1 seperated by commas (eg "0.5+0.3i, -0.2-0.7i, 0.1i") or "rand,n": initializes with n points that are either 1 or 0 (which are then mirrored, see parity) "rand-phase,n" initializes with n points that are like exp(iφ) with random φ ∈ [0, 2π)

Options:
  -d, --degree <DEGREE>      degree (>0) [default: 60]
  -s, --hotstart <HOTSTART>  hotstart-degree (>0) [default: 20]
  -p, --parity <PARITY>      parity ("even" or "odd") [default: even] [possible values: even, odd]
  -o, --output <OUTPUT>      Path for the solution output
  -D, --drawable <DRAWABLE>  Path for outputing the data formated to be drawn in gnuplot
  -h, --help                 Print help
```
