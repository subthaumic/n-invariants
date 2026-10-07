"""Sample sizes and sampling for distributions of metric n-point invariants.

Companion code for *Distributions of metric n-point invariants and applications to
molecular evolution*.

:mod:`n_invariants.bounds`
    :func:`~n_invariants.bounds.sample_size` gives the number of n-tuples to sample so
    that the empirical curvature measure, or the empirical distribution of a Lipschitz
    n-point invariant, is within a given Wasserstein accuracy of the true one with a given
    confidence. The number does not depend on the number of points.
:mod:`n_invariants.sampling`
    :func:`~n_invariants.sampling.empirical_curvature_measure` and
    :func:`~n_invariants.sampling.empirical_distribution` draw these samples from a point
    set with a chosen metric and measure.
:mod:`n_invariants.invariants`
    The invariants of the article: pairwise distance, four-point hyperbolicity deficit and
    Vietoris-Rips persistence.

The ``manuscript/`` directory at the repository root reproduces the article: its figures,
the Jukes-Cantor and Halpern-Bruno substitution models, and the SARS-CoV-2 pipeline.
"""

from .bounds import expected_wasserstein_bound, fournier_constant, sample_size, wasserstein_bound
from .sampling import empirical_curvature_measure, empirical_distribution

__version__ = "1.0.0"

__all__ = [
    "sample_size",
    "wasserstein_bound",
    "expected_wasserstein_bound",
    "fournier_constant",
    "empirical_curvature_measure",
    "empirical_distribution",
    "__version__",
]
