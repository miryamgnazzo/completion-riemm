# Riemannian Tucker tensor completion for parametric Markov models

Code and numerical results for

> **Enhanced evaluation of parametric Markov models via Riemannian Tucker tensor completion**
> Miryam Gnazzo, Leonardo Robol, Silvano Chiaradonna, Felicita Di Giandomenico

Dependability measures depending on time and on several parameters are
represented as a low-rank Tucker tensor, reconstructed by Riemannian completion
from a few fibres sampled at Chebyshev nodes, and compared against Adaptive
Cross Approximation (ACA).

## Requirements

MATLAB R2022b or later, plus three external dependencies:

| Dependency | Provides |
|---|---|
| [Manopt](https://www.manopt.org/) >= 7.0 | Riemannian optimisation |
| [Tensor Toolbox](https://www.tensortoolbox.org/) >= 2.6 | Tucker tensor arithmetic |
| [MarkovCrossApproximation](https://github.com/106ohm/MarkovCrossApproximation) | ACA, the comparison method |

```matlab
addpath(genpath('/path/to/manopt'));
addpath(genpath('/path/to/tensor_toolbox'));
addpath('/path/to/MarkovCrossApproximation');
```

The published results use MarkovCrossApproximation at commit `5fa8b5d`. No
other toolbox is required. To check the installation:

```bash
matlab -batch "smoke_check"
```

## Layout

```
src/completion/   Chebyshev interpolation + Riemannian completion
src/models/       generators of the two case studies, Kolmogorov solvers
experiments/      one folder per experiment, with its .mat results
```

`setup_paths.m` is called by every script, so the repository needs no manual
path setup.

## Experiments

| | Section of the paper | Setting | Runs | Time |
|---|---|---|---|---|
| **EX1** | Comparison with ACA | Case 1, reliability, `d = 3,4,5` | 50 seeds | ~1 h |
| **EX1-under** | Comparison with ACA | Case 1, under repair, `d = 6` | 50 seeds | ~4 h |
| **EX2** | Increasing dimensionality | Case 1, reliability, `d = 6..9` | 1 | ~10 min |
| **EX3** | Scalability in state-space size | Case 2 (IPS), reliability, `d = 5,6,7` × `ne = 20,30,40` | 50 seeds | ~10 h |
| **EX3-under** | Scalability in state-space size | Case 2 (IPS), under repair, `d = 5` | 1 | ~5 min per `ne` |

```bash
matlab -batch "cd experiments/EX1_case1; Script_case1_robustness_seed50"
```

```bash
matlab -batch "cd experiments/EX1_case1; Script_case1_underrepair_d6ext_r3_seed50"
```

```bash
matlab -batch "cd experiments/EX2_dsweep; Script_case1new_cheb_dsweep_ordEX1"
```

```bash
matlab -batch "cd experiments/EX3_case2_reliability; run_ex3_all_ne"
```

```bash
matlab -batch "cd experiments/EX3_case2_underrepair; run_ex3_underrepair_paper"
```

Every script saves after each unit of work and supports resuming: re-launch the
same command to continue an interrupted run.

## Results and figures

The `.mat` files of the published runs are included. Everything derived from
them — `.tex` fragments, diagnostic `.png`, logs — is git-ignored and rebuilt on
demand, in seconds and with base MATLAB only:

```matlab
cd experiments/EX1_case1;             gen_boxplot_tex_EX1
cd experiments/EX3_case2_reliability; gen_boxplot_tex_50
```

Quartiles and Tukey whiskers are computed in `log10` scale, so the spread read
off the plots is multiplicative.

## Notes

- Case study 1 activates parameters in the order of the parameter table of the
  paper, `par_order = [1 5 8 4 3 2 7 6]`, not in the argument order of
  `evalQ_extended`.
- Case study 2 reliability uses `par_order = [3 4 5 7 8 9 10 11 12 6 1 2]`.
- The test set is fixed and shared by both methods (`sample_seed = 0`); the run
  seed varies the observed fibres and the ACA pivots identically for both.
