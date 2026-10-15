"""Sample sizes and sampling for distributions of metric n-point invariants.

Companion code for *Distributions of metric n-point invariants and applications to
molecular evolution*.

:mod:`n_invariants.bounds`
    :func:`~n_invariants.bounds.sample_size` gives the number of n-tuples to sample so
    that the empirical curvature measure, or the empirical distribution of a Lipschitz
    n-point invariant, is within a given Wasserstein accuracy of the true one with a given
    confidence; :func:`~n_invariants.bounds.sample_size_real_valued` does the same for a
    real-valued n-point invariant. The numbers do not depend on the number of points.
    :func:`~n_invariants.bounds.guaranteed_accuracy` gives the accuracy a sample guarantees for each
    of several n-point invariants.
:mod:`n_invariants.sampling`
    :func:`~n_invariants.sampling.sample_tuples`,
    :func:`~n_invariants.sampling.empirical_curvature_measure` and
    :func:`~n_invariants.sampling.empirical_distributions` draw these samples from a point
    set with a chosen metric and measure. A seed fixes the sample, so it can be revisited.
:mod:`n_invariants.invariants`
    The n-point invariants of the article, pairwise distance, four-point hyperbolicity
    deficit and Vietoris-Rips persistence, and :class:`~n_invariants.invariants.Invariant`,
    which carries the constants their bounds need.

The ``manuscript/`` directory at the repository root reproduces the article: its figures,
the Jukes-Cantor and Halpern-Bruno substitution models, and the SARS-CoV-2 pipeline.
"""

from .bounds import (
    Guarantee,
    fournier_constant,
    guaranteed_accuracy,
    sample_size,
    sample_size_real_valued,
    wasserstein_bound,
    wasserstein_bound_real_valued,
    wasserstein_moment_bound,
    wasserstein_moment_bound_real_valued,
)
from .invariants import HYPERBOLICITY_DEFICIT, PAIRWISE_DISTANCE, VR_PERSISTENCE_4_1, Invariant
from .sampling import (
    empirical_curvature_measure,
    empirical_distribution,
    empirical_distributions,
    sample_tuples,
)

__version__ = "1.2.0"

__all__ = [
    "sample_size",
    "wasserstein_bound",
    "wasserstein_moment_bound",
    "sample_size_real_valued",
    "wasserstein_bound_real_valued",
    "wasserstein_moment_bound_real_valued",
    "fournier_constant",
    "guaranteed_accuracy",
    "Guarantee",
    "Invariant",
    "PAIRWISE_DISTANCE",
    "HYPERBOLICITY_DEFICIT",
    "VR_PERSISTENCE_4_1",
    "sample_tuples",
    "empirical_curvature_measure",
    "empirical_distributions",
    "empirical_distribution",
    "__version__",
]
