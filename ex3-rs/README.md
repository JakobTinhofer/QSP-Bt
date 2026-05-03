# The goal

push larger and larger polys. Current largest fit: rand,100 with hotstart,80,400 in ~800s

runs in <1s for poly degree of 60 using hotstart

### Usage

```
Fit a QSP polynomial to the given sequence of target points.

Usage: ex3-rs [OPTIONS] <COMMAND>

Commands:
  solve-poly
  plot-runtimes
  help           Print this message or the help of the given subcommand(s)

Options:
  -m, --backend-mode <BACKEND_MODE>  Enable/disable multithreading for gradient, lossfunction evaluation. Auto: will do single threading for small d & short sequences. (both <= 100) [default: auto] [possible values: single-thread, multi-thread, auto]
  -M, --mode <MODE>                  Solve mode: "simple,D" — direct solve at degree D "hotstart,S,D" — solve at degree S, then continue at degree D "cascade,N,D" — N cascading steps up to degree D if running PlotRuntimes task, this will be interpreted as a ratio and scaled accordingly [default: hotstart,20,60]
  -s, --solver <KIND>                Which optimizer to use [default: bfgs] [possible values: bfgs, lm]
  -h, --help                         Print help (see more with '--help')

L-BFGS Options:
      --bfgs-max-iters <bfgs_max_iters>  [default: 500000]
      --bfgs-mem <bfgs_mem>              [default: 10]
      --bfgs-tol-grad <bfgs_tol_grad>    [default: 1e-8]

Levenberg-Marquardt Options:
      --lm-max-iters <lm_max_iters>            [default: 500]
      --lm-initial-lambda <lm_initial_lambda>  [default: 1e-4]
      --lm-tol <lm_tol>                        [default: 1e-10]
```

### Example output

```
ex3-rs "rand,10" -D test.dat
[i] Running with degree=60 and hotstart_degree=20. Target:
x:
[
    0:  0
    1: -0.09
    2: -0.184
    3: -0.284
    4: -0.389
    5: -0.5
    6: -0.616
    7: -0.738
    8: -0.866
    9: -1
   10:  1
   11:  0.866
   12:  0.738
   13:  0.616
   14:  0.5
   15:  0.389
   16:  0.284
   17:  0.184
   18:  0.09
   19:  0
]
y:
[
    0: 1
    1: 0
    2: 0
    3: 1
    4: 1
    5: 0
    6: 0
    7: 0
    8: 0
    9: 0
   10: 0
   11: 0
   12: 0
   13: 0
   14: 0
   15: 1
   16: 1
   17: 0
   18: 0
   19: 1
]
[+] Finished solving! Elapsed: 217.140254ms ms. Result:
[
    0:  1.99235
    1: -12.30864
    2:  5.87124
    3: -31.07559
    4:  87.80426
    5: -81.57304
    6:  43.83502
    7: -94.29692
    8:  31.52906
    9: -81.54348
   10:  37.61189
   11: -93.94086
   12:  68.51819
   13: -68.99755
   14: -5.9434
   15:  12.10736
   16: -5.3424
   17:  25.02614
   18: -0.33147
   19:  18.19677
   20:  5.33617
   21:  0.88458
   22:  1.86445
   23:  3.2063
   24:  2.04917
   25:  1.55721
   26:  5.9394
   27:  2.29069
   28:  0.12507
   29:  2.73139
   30:  5.86748
   31:  1.0883
   32:  4.0973
   33:  4.20156
   34:  3.25944
   35:  5.92565
   36:  6.27739
   37: -0.19827
   38:  0.17269
   39:  2.71934
   40:  5.56757
   41:  2.51009
   42:  3.76638
   43:  4.20126
   44:  5.25548
   45:  3.56253
   46:  2.74394
   47:  5.0699
   48:  0.2794
   49:  4.67792
   50:  3.47817
   51:  4.65378
   52:  4.61106
   53:  0.70079
   54:  4.69396
   55:  0.29727
   56:  5.23873
   57:  6.67593
   58:  2.40277
   59:  0.29762
   60:  1.22765
]
Wrote drawing data to 'test.dat'
```
