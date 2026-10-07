# Article material

The code and derived data behind the figures and numbers of the article, with all settings
fixed to the values used there. This directory is a separate uv project, pinned to release
v1.1.0 of `n_invariants`. Commands are run from here:

```bash
cd manuscript
uv sync                 # uv sync --extra sars for the SARS-CoV-2 pipeline
```

## Figures and statistics

| Article | Command | Runtime | Reads |
|---|---|---|---|
| Figure 2 (§6) | `uv run python figures/figure_2_sample_size_nomogram.py` | ~10 s | |
| Figure 5 (§8.2) | `uv run python figures/figure_5_hamming_hyperbolicity_exact.py` | ~5 s | |
| Figure 6 (§8.3, App. A) | `uv run python figures/figure_6_substitution_models.py` | ~10 s | `substitution_models/results/` |
| Figure 7 (§8.4, App. B) | `uv run python figures/figure_7_sars_geometry.py` | ~10 s | `sars_cov2/results/` |
| Statistics (§8.4) | `uv run python sars_cov2/summary_stats.py` | <1 s | `sars_cov2/results/` |

- The figures are written to `figures/output/`, which holds them as they appear in the
  article.
- Figures 2 and 5 are computed exactly within their scripts. Figures 6 and 7 and the
  statistics read precomputed results.

## Results

| Results | Computed by | Runtime |
|---|---|---|
| `substitution_models/results/` | `uv run python substitution_models/compute_invariants.py` | ~40 min |
| `sars_cov2/results/` | steps 0 to 3 of the [SARS-CoV-2 pipeline](sars_cov2/README.md) | see its README |

- `compute_invariants.py` overwrites the precomputed file; `--output` writes elsewhere.
- The SARS-CoV-2 pipeline needs the GISAID sequences (registration required, not
  redistributable). `tests/test_sars_pipeline.py` runs it on a synthetic alignment.

## Layout

```
figures/                 one script per figure, and plotting.py, the style shared by Figures 6 and 7
figures/output/          the figures as they appear in the article
substitution_models/     the Jukes–Cantor and Halpern–Bruno models (models.py), their sampler
                         (sampling.py), and compute_invariants.py
sars_cov2/               the SARS-CoV-2 pipeline and its sampler (sampling.py), see its README
tests/                   tests of the code above, run with uv run --extra sars pytest
pyproject.toml, uv.lock  the environment of this directory
```

Both applications evaluate the invariants of `n_invariants.invariants`, each with its own
sampler.

## Reproducibility

- `uv.lock` pins release v1.1.0 of the package and the library versions used for the
  article. With them, every figure script reproduces its PDF in `figures/output/` byte for
  byte.
- All random steps are seeded, so `compute_invariants.py` reproduces
  `substitution_models/results/invariants.npz`.
