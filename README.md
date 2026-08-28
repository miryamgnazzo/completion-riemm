# Riemannian Tucker tensor completion for parametric Markov models

Code and numerical results for

> **Enhanced evaluation of parametric Markov models via Riemannian Tucker tensor completion**
> Miryam Gnazzo, Leonardo Robol, Silvano Chiaradonna, Felicita Di Giandomenico

Dependability measures depending on time and on several parameters are
represented as a low-rank Tucker tensor, reconstructed by Riemannian completion
from a few fibres sampled at Chebyshev nodes, and compared against Adaptive
Cross Approximation (ACA).

## Requirements

MATLAB R2022b or later, with three external dependencies:

```matlab
addpath(genpath('/path/to/manopt'));
addpath(genpath('/path/to/tensor_toolbox'));
addpath('/path/to/MarkovCrossApproximation');
```

```
src/completion/   Chebyshev interpolation + Riemannian completion
src/models/       generators of the two case studies, Kolmogorov solvers
experiments/      one folder per experiment, with its .mat results
```

## Experiments

| | Section of the paper | Setting |
|---|---|---|
| **EX1** | Comparison with ACA | Case 1, reliability, `d = 3,4,5` | 
| **EX1-under** | Comparison with ACA | Case 1, under repair, `d = 6` | 
| **EX2** | Increasing dimensionality | Case 1, reliability, `d = 6..9` | 
| **EX3** | Scalability in state-space size | Case 2 (IPS), reliability, `d = 5,6,7` × `ne = 20,30,40` | 
| **EX3-under** | Scalability in state-space size | Case 2 (IPS), under repair, `d = 5` | 
