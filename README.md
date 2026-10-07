# n-invariants

Companion code for the article

> **Distributions of metric $n$-point invariants and applications to molecular evolution**  
> Michael Bleher, Mario Gómez, Facundo Mémoli, Maximilian Neumann and Andreas Ott  
> Preprint forthcoming

Data can often be viewed as samples from a metric measure space $(X,d_X,\mu_X)$, where $d_X$
measures distances between possible observations and $\mu_X$ is the distribution by which they
occur. The geometry of $X$ then carries information about the data-generating process, much of
which is accessible through *metric $n$-point invariants*, quantities that depend only on the
distances within an $n$-tuple. Examples are the pairwise distance, the four-point hyperbolicity
deficit, and the Vietoris–Rips persistence diagram. Their distributions are determined by
Gromov's $n$-th curvature measure $\mu_n$, which records the full '$n$-point geometry' of $X$,
i.e. how often each configuration of distances between $n$ points occurs.

We bound the expected $p$-Wasserstein distance between $\mu_n$ and the empirical curvature
measure of $N$ sampled $n$-tuples, and determine sample sizes sufficient for a target accuracy
and confidence. The bounds carry over to the distribution of any Lipschitz metric $n$-point
invariant, and they do not depend on the number of points of $X$, so the distributions can be
estimated even when the distance matrix of $X$ is too large to compute. We apply this framework
to the Jukes–Cantor and Halpern–Bruno substitution models and to five million SARS-CoV-2
genomes, studying the distributions of pairwise distance, hyperbolicity deficit, and
Vietoris–Rips persistence diagrams.

This repository provides the Python package `n_invariants`, which computes these sample sizes
for any $n$, any Wasserstein order $p$ and any Lipschitz metric $n$-point invariant.
It also contains the material used to reproduce the article's material in the application to
molecular evolution.

## Contents

```
src/n_invariants/       Python package: sample sizes, sampling, invariants, CLI
tests/                  tests of the package
```

```
manuscript/                         Article material
  manuscript/figures/               scripts used to generate the article's figures
  manuscript/sars_cov2/             SARS-CoV-2 pipeline and its derived results
  manuscript/substitution_models/   Jukes–Cantor and Halpern–Bruno models, data for Figure 6
  manuscript/tests/                 tests of the article code
```

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/subthaumic/n-invariants.git
cd n-invariants
uv sync                 # uv sync --extra sars for the SARS-CoV-2 pipeline
```

This installs the package `n_invariants` and the command `n-invariants`. `uv.lock` pins the
library versions the article's figures were made with.

## Usage

### Sample size

The number of $n$-tuples to sample so that, with probability at least `alpha`, the empirical
distribution is within `epsilon * diam(X)` of the true one in $W_p$:

```bash
# the curvature measure mu_n, and with it every Lipschitz invariant (Cor. 6.3)
uv run n-invariants bound -n 4 -p 1 --epsilon 0.1 --alpha 0.95

# a single invariant with Lipschitz constant 2 and range constant 1 (Cor. 6.5)
uv run n-invariants bound -n 4 -p 1 --epsilon 0.1 --alpha 0.95 --lipschitz 2 --lambda 1
```

```python
from n_invariants import sample_size

sample_size(epsilon=0.1, alpha=0.95, n=4, p=1)
sample_size(epsilon=0.1, alpha=0.95, n=4, p=1, lip=2, lam=1)
```

| Parameter | CLI flag | CLI default | Description |
|---|---|---|---|
| `epsilon` | `-e`, `--epsilon` | 0.1 | accuracy, in units of $\mathrm{diam}(X)$ |
| `alpha` | `-a`, `--alpha` | 0.95 | confidence |
| `n` | `-n` | 4 | points per sampled tuple |
| `p` | `-p` | 2 | order of the Wasserstein distance |
| `lip` | `-L`, `--lipschitz` | 1 | Lipschitz constant of the invariant, w.r.t. the sup norm on distance matrices |
| `lam` | `--lambda` | `lip` | range constant: $\mathrm{diam}\,F(K_n(X)) \le \lambda\,\mathrm{diam}(X)$ |
| `method` | `--method` | `WB` | concentration inequality: `WB` (as in the article), `MD` or `Markov` |

### Sampling

```python
import numpy as np
from n_invariants import sample_size, empirical_curvature_measure, empirical_distribution
from n_invariants.invariants import hyperbolicity_deficit

X = np.random.default_rng(0).normal(size=(10_000, 3))   # 10,000 points in R^3

# the empirical curvature measure: one row of six distances per sampled quadruple
N = sample_size(epsilon=0.2, alpha=0.95, n=4, p=1)
K = empirical_curvature_measure(X, 4, N, metric="euclidean", seed=0)

# the empirical distribution of the hyperbolicity deficit, as a histogram
N_F = sample_size(epsilon=0.2, alpha=0.95, n=4, p=1, lip=2, lam=1)
values, counts = empirical_distribution(X, hyperbolicity_deficit, 4, N_F,
                                        metric="euclidean", seed=0, decimals=3)
```

- `empirical_curvature_measure(X, n, n_samples, ...)` returns an array of shape
  `(n_samples, n(n-1)/2)`: the distances within each sampled tuple. It holds all samples
  in memory.
- `empirical_distribution(X, invariant, n, n_samples, ...)` returns `(values, counts)`. It
  keeps only the histogram, so its memory use does not grow with `n_samples`.
- Only the distances within the sampled tuples are computed, never the full distance matrix of the data set.
- With the same `seed`, both functions draw the same tuples.

| Parameter | Default | Description |
|---|---|---|
| `metric` | required | `"euclidean"`, `"hamming"` (number of differing coordinates), `"precomputed"` (`X` is a distance matrix), or a function `metric(A, B)` returning the distances between corresponding rows of `A` and `B` |
| `weights` | uniform | the measure on the points of `X`, up to normalization |
| `seed` | `None` | with `n_samples` and `batch_size`, determines the result |
| `batch_size` | 100,000 | tuples per batch; bounds the memory in use |
| `decimals` | `None` | `empirical_distribution` only: round values before counting, useful for real-valued metrics |
| `n_workers` | 1 | parallel processes; changes the speed, not the result |

### Invariants

The invariants of the article, in `n_invariants.invariants`, with their constants for
`sample_size`:

| Invariant | Function | `n` | `lip` | `lam` |
|---|---|---|---|---|
| pairwise distance | `pairwise_distance` | 2 | 1 | 1 |
| four-point hyperbolicity deficit | `hyperbolicity_deficit` | 4 | 2 | 1 |
| $(4,1)$-persistence | `vr_persistence` | 4 | 1 | 1 |

Your own invariant works too:

- It maps distance rows of shape `(size, n(n-1)/2)` to one value per row, an array of shape
  `(size,)` or `(size, k)`.
- Pass it to `empirical_distribution`, and its Lipschitz and range constants to
  `sample_size`.

## Reproducing the article

To reproduce all figures in the article, run

```bash
for script in manuscript/figures/figure_*.py; do uv run python "$script"; done
```

The figures are written to `manuscript/figures/output/`. Figures 6 and 7 read precomputed
results, the invariant distributions and MDS embeddings of the two applications:

| Results | Used for | To recompute |
|---|---|---|
| `manuscript/substitution_models/results/` | Figure 6 | one script, ~40 min |
| `manuscript/sars_cov2/results/` | Figure 7, Section 8.4 statistics | GISAID registration, sequence download, four-step pipeline |

All commands: [`manuscript/README.md`](manuscript/README.md).

## Tests

```bash
uv run pytest                    # package
uv run pytest manuscript/tests   # article code; the pipeline test requires the sars extra
```

## Citation

The article reference will be added when the preprint appears. To cite the software, use
[`CITATION.cff`](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
