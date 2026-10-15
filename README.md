# n-invariants

[![DOI](https://zenodo.org/badge/1347439527.svg)](https://doi.org/10.5281/zenodo.23210358)

Companion code for the article

> **Distributions of metric $n$-point invariants and applications to molecular evolution**  
> Michael Bleher, Mario Gómez, Facundo Mémoli, Maximilian Neumann and Andreas Ott  
> Preprint forthcoming

Data can often be viewed as samples from a metric measure space $(X,d_X,\mu_X)$, where $d_X$ measures distances between possible observations and $\mu_X$ is the distribution by which they occur.
The geometry of $X$ then carries information about the data-generating process, much of which is accessible through *metric $n$-point invariants*, quantities that depend only on the distances within an $n$-tuple.
Examples are the pairwise distance, the four-point hyperbolicity deficit, and the Vietoris–Rips persistence diagram.
Their distributions are determined by Gromov's $n$-th curvature measure $\mu_n$, which records the full '$n$-point geometry' of $X$, i.e. how often each configuration of distances between $n$ points occurs.

We bound the expected $p$-Wasserstein distance between $\mu_n$ and the empirical curvature measure of $N$ sampled $n$-tuples, and determine sample sizes sufficient for a target accuracy and confidence.
The bounds carry over to the distribution of any Lipschitz metric $n$-point invariant, and they do not depend on the number of points of $X$, so the distributions can be estimated even when the distance matrix of $X$ is too large to compute.
We apply this framework to the Jukes–Cantor and Halpern–Bruno substitution models and to five million SARS-CoV-2 genomes, studying the distributions of pairwise distance, hyperbolicity deficit, and Vietoris–Rips persistence diagrams.

This repository provides the Python package `n_invariants`.
It computes these sample sizes for arbitrary $n$, Wasserstein orders $p$ and Lipschitz metric $n$-point invariants, and estimates the distributions by sampling from data under a metric of your choice.
The directory `manuscript/` contains the code and precomputed results that reproduce the article's figures and analyses.

## Contents

```
src/n_invariants/       Python package: sample sizes, sampling, invariants, CLI
tests/                  tests of the package
```

```
manuscript/                         Article material, a separate uv project
  manuscript/figures/               scripts used to generate the article's figures
  manuscript/sars_cov2/             SARS-CoV-2 pipeline and its derived results
  manuscript/substitution_models/   Jukes–Cantor and Halpern–Bruno simulation and its results
  manuscript/tests/                 tests of the article code
```

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/subthaumic/n-invariants.git
cd n-invariants
uv sync
```

This installs the package `n_invariants` and the command `n-invariants`.

## Usage

### 1. Sample size

The number of $n$-tuples to sample so that, with probability at least `alpha`, the empirical curvature measure is within `epsilon * diam(X)` of the true one in $W_p$:

```bash
# the curvature measure mu_n (Cor. 6.3)
uv run n-invariants bound -n 4 -p 1 --epsilon 0.05

# one Lipschitz n-invariant with Lipschitz constant 2 and range constant 1 (Cor. 6.5)
uv run n-invariants bound -n 4 -p 1 --epsilon 0.05 --lipschitz 2 --lambda 1

# one real-valued n-invariant with range constant 1, whatever n is (Prop. 6.6)
uv run n-invariants bound-real-valued -p 1 --epsilon 0.05
```

```python
from n_invariants import sample_size, sample_size_real_valued

sample_size(epsilon=0.05, alpha=0.999, n=4, p=1)                   # 3,385,553,635
sample_size(epsilon=0.1, alpha=0.999, n=4, p=1, lip=2, lam=1)
sample_size_real_valued(epsilon=0.001, alpha=0.999, p=1, lam=1)    # 18,255,749
```

| Parameter | CLI flag | CLI default | Description |
|---|---|---|---|
| `epsilon` | `-e`, `--epsilon` | 0.1 | accuracy, in units of $\mathrm{diam}(X)$ |
| `alpha` | `-a`, `--alpha` | 0.999 | confidence |
| `n` | `-n` | 4 | points per sampled tuple (`bound` only) |
| `p` | `-p` | 2 | order of the Wasserstein distance |
| `lip` | `-L`, `--lipschitz` | 1 | Lipschitz constant of the $n$-invariant, w.r.t. the sup norm on distance matrices (`bound` only) |
| `lam` | `--lambda` | `lip`, or 1 for `bound-real-valued` | range constant: $\mathrm{diam}\,F(K_n(X)) \le \lambda\,\mathrm{diam}(X)$ |
| `method` | `--method` | `WB` | concentration inequality: `WB` (as in the article), `MD` or `Markov` (`bound` only) |

### 2. Sampling

```python
import numpy as np
from n_invariants import (HYPERBOLICITY_DEFICIT, PAIRWISE_DISTANCE, VR_PERSISTENCE_4_1,
                          empirical_curvature_measure, empirical_distributions, sample_size,
                          sample_tuples)

X = np.random.default_rng(0).normal(size=(10_000, 3))   # 10,000 points in R^3
N = sample_size(epsilon=0.2, alpha=0.999, n=4, p=1)      # 874,241 quadruples

# all three n-invariants from the same quadruples, as histograms
hists = empirical_distributions(X, [PAIRWISE_DISTANCE, HYPERBOLICITY_DEFICIT, VR_PERSISTENCE_4_1],
                                4, N, metric="euclidean", seed=0, decimals=3)
values, counts = hists["hyperbolicity_deficit"]

# the same quadruples, lazily, in batches of point indices and the distances among them
for indices, distances in sample_tuples(X, 4, N, metric="euclidean", seed=0):
    ...

# the same quadruples as one array of distances, one row per quadruple
K = empirical_curvature_measure(X, 4, N, metric="euclidean", seed=0)
```

- `sample_tuples(X, n, n_samples, ...)` yields the sampled tuples batch by batch.
- `empirical_curvature_measure(X, n, n_samples, ...)` returns the sampled distance matrices, an array of shape `(n_samples, n(n-1)/2)`.
  Above `max_bytes` (4 GiB by default) it refuses to allocate the array in memory; pass `out="file.npy"` to write it to disk instead.
- `empirical_distributions(X, invariants, n, n_samples, ...)` computes the distance matrices of each batch and evaluates every $n$-invariant on them.
  It returns `{name: (values, counts)}` and keeps only the histograms, so its memory use does not grow with `n_samples`.
  An $n$-invariant on $k < n$ points is evaluated on the first $k$ points of each tuple, which gives the same distribution as sampling $k$-tuples.
  `empirical_distribution` does the same for one $n$-invariant.

**The sample is fixed by the seed.**
The seed, the points with their weights, and `n` fix one sequence of i.i.d. tuples, and `n_samples` only says how many of them to use.
All functions draw the same tuples for the same seed, so a sample can be revisited later for other $n$-invariants, and the first $M$ tuples of a sample of size $N$ are the sample of size $M$.
`batch_size` and `n_workers` change the speed and the memory in use, not the tuples.

| Parameter | Default | Description |
|---|---|---|
| `metric` | required | `"euclidean"`, `"hamming"` (number of differing coordinates), `"precomputed"` (`X` is a distance matrix), or a function `metric(A, B)` returning the distances between corresponding rows of `A` and `B` |
| `weights` | uniform | the measure on the points of `X`, up to normalization |
| `seed` | `None` | fixes the sample |
| `batch_size` | 100,000 | tuples per batch; bounds the memory in use |
| `n_workers` | 1 | parallel processes, all available CPUs if 0 or negative |
| `progress` | `True` | print the percentage of batches done to stderr, in steps of 5% |
| `decimals` | `None` | `empirical_distributions` only: round values before counting, useful for real-valued metrics |
| `out`, `max_bytes` | `None`, 4 GiB | `empirical_curvature_measure` only: write to a `.npy` file, or the largest array to allocate in memory |

### 3. Guaranteed accuracy

For each $n$-invariant, `guaranteed_accuracy` returns the sharpest accuracy that $N$ sampled tuples guarantee: through the curvature measure (Cor. 6.3, or Cor. 6.5 when the range constant is below the Lipschitz constant), or for a real-valued $n$-invariant by Prop. 6.6.

```python
from n_invariants import guaranteed_accuracy

guarantees = guaranteed_accuracy(N, [PAIRWISE_DISTANCE, HYPERBOLICITY_DEFICIT, VR_PERSISTENCE_4_1], p=1)
for g in guarantees.values():
    print(g.invariant, g.accuracy, g.statement)
```

```bash
uv run n-invariants accuracy -N 53433249 -p 1 --alpha 0.996667
```

Each accuracy holds with probability at least `alpha` on its own.
For $m$ of them to hold together with probability $\beta$, choose `alpha = 1 - (1 - beta) / m` (union bound).
The bounds depend only mildly on `alpha`, so the default 0.999 costs little: it covers 10 statements at joint confidence 0.99, or 50 at 0.95.

### 4. $n$-invariants

An `Invariant` carries a function of the distance matrices together with the Lipschitz constant, the range constant, and the number of points.

| `Invariant` | Function | `n` | `lip` | `lam` | values |
|---|---|---|---|---|---|
| `PAIRWISE_DISTANCE` | `pairwise_distance` | 2 | 1 | 1 | real |
| `HYPERBOLICITY_DEFICIT` | `hyperbolicity_deficit` | 4 | 2 | 1 | real |
| `VR_PERSISTENCE_4_1` | `vr_persistence` | 4 | 1 | 1 | persistence diagrams |

Your own $n$-invariant works too:

```python
from n_invariants import Invariant

def perimeter(distance_matrices): # (size, n(n-1)/2) -> (size,)
    return distance_matrices.sum(axis=1)

PERIMETER = Invariant(perimeter, n=3, lip=3, lam=3, real_valued=True)
```

Here `distance_matrices` is of shape `(size, n(n-1)/2)`, where `size` is the number of tuples in a batch.
Row $i$ is the distance matrix of the $i$-th tuple $(x_0, \dots, x_{n-1})$ in flattened form: the distances $d(x_a, x_b)$ for $a < b$, in the order $(0,1), (0,2), \dots, (n-2,n-1)$, as in `scipy.spatial.distance.squareform`.
`Invariant` returns an array of shape `(size,)`, or `(size, k)` for values with $k$ components, such as the (birth, death) pairs of `vr_persistence`.

## Reproducing the article

The directory `manuscript/` contains all code for reproducing the computational results and figures of the article; see [`manuscript/README.md`](manuscript/README.md).
It simulates the Jukes–Cantor and Halpern–Bruno models from scratch in about 40 minutes.
The SARS-CoV-2 pipeline needs the GISAID sequences, which require registration and cannot be redistributed.
The results of both are stored in the repository, so every figure can be redrawn in seconds without running the simulations or data analysis pipeline.

## Citation

The article reference will be added when the preprint appears.
To cite the software, use [`CITATION.cff`](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
