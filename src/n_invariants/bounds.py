"""Sample-size bounds for estimating the distribution of a metric n-point invariant.

The distribution of an n-point invariant ``F`` of a metric measure space ``(X, d, mu)`` is
the pushforward ``nu_F = F_#(mu_n)`` of the *curvature measure* ``mu_n``, the law of the
``n(n-1)/2`` pairwise distances among n points drawn i.i.d. from ``mu``. Estimating it by
Monte Carlo means replacing ``mu_n`` by the empirical measure ``mu_{N,n}`` of N sampled
n-tuples, and ``nu_F`` by its pushforward ``nu_{F,N}``.

This module answers the practical question that makes such an estimate rigorous:

    How many n-tuples N must be drawn so that ``W_p(nu_{F,N}, nu_F) <= epsilon diam(X)``
    with probability at least ``alpha``?

The article answers it in three statements, each implemented here. Statement numbers
refer to the preprint of the article.

=================  ================================================  ==================================
Statement          Bounds                                            Function
=================  ================================================  ==================================
Thm 6.1, Cor 6.2   ``E[W_p^p]`` for ``mu_n`` and for a Lipschitz F   :func:`expected_wasserstein_bound`
Cor 6.3            N for ``mu_n``, and so for every Lipschitz F      :func:`sample_size`
Cor 6.5            N for one Lipschitz F of known range              :func:`sample_size`
=================  ================================================  ==================================

:func:`wasserstein_bound` is the inverse of :func:`sample_size`: the high-probability bound
on ``W_p`` that a given N achieves.

Accuracies are in units of ``diam(X)``, so the diameter never enters a sample count. An
invariant ``F : K_n -> Z`` is described by two constants, which the caller supplies:

``lip``
    Its Lipschitz constant with respect to the supremum norm on distance matrices
    (Def 4.1). It scales the approximation term.
``lam``
    A range constant with ``diam F(K_n(X)) <= lam * diam(X)``. It scales the
    concentration term. ``lam = lip`` is always valid, and is the default.

``lip = lam = 1`` describes the curvature measure ``mu_n`` itself (Cor 6.3). Its guarantee
transfers to every Lipschitz F at once, with error ``epsilon * Lip(F) * diam(X)``, so one
sample serves any number of invariants. A specific ``lip`` gives the guarantee for that one
F at error ``epsilon * diam(X)``; with ``lam = lip`` this is Cor 6.3 at accuracy
``epsilon / lip``, and a smaller valid ``lam`` refines it (Cor 6.5). The article's
invariants (Prop 4.8 and Section 8.4), implemented in :mod:`n_invariants.invariants`:

=================================  =========================  ===  =====  =====
Invariant                          Function                   n    lip    lam
=================================  =========================  ===  =====  =====
pairwise distance                  ``pairwise_distance``      2    1      1
four-point hyperbolicity deficit   ``hyperbolicity_deficit``  4    2      1
(4,1)-persistence                  ``vr_persistence``         4    1      1
=================================  =========================  ===  =====  =====

No sample count depends on the number of points of X, so the bounds apply equally to point
sets too large for a pairwise distance matrix, such as the SARS-CoV-2 haplotypes of the
article.

The bounds combine two ingredients:

* an upper bound on ``E[W_p^p(mu_{N,n}, mu_n)]`` in the form ``C * N^{-e}``, obtained by
  specializing the explicit non-asymptotic bounds of Fournier [1]_ to ``K_n(X)`` with the
  ``L^inf`` metric, and
* a concentration inequality turning that expectation bound into a high-probability
  statement (Lemma 3.12). The default is the bounded-differences inequality of Weed and
  Bach [2]_ (``method="WB"``); Markov and McDiarmid variants are provided for comparison
  and give markedly worse constants.

Because ``K_n(X)`` is compact, ``mu_n`` has finite moments of every order, so the ``q``-th
moment appearing in the general theory can be taken at ``q = inf``, where it equals the
minimum enclosing radius ``diam(X)/2``. This collapses Fournier's constant to ``kappa``
alone and is what makes the specialized forms below so much simpler than the general
ones.

References
----------
.. [1] N. Fournier, "Convergence of the empirical measure in expected Wasserstein
       distance: non-asymptotic explicit bounds in R^d", ESAIM: PS 27 (2023) 749-775.
       DOI: 10.1051/ps/2023011
.. [2] J. Weed and F. Bach, "Sharp asymptotic and finite-sample rates of convergence of
       empirical measures in Wasserstein distance", Bernoulli 25 (2019) 2620-2648.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import fsolve, minimize_scalar

__all__ = [
    "sample_size",
    "wasserstein_bound",
    "expected_wasserstein_bound",
    "fournier_constant",
    "N_cover",
    "K_cover",
    "kappa_1",
    "kappa_2",
    "kappa_3",
]


# ---------------------------------------------------------------------------
# Covering numbers (needed only for the finite-m branches of kappa_*)
# ---------------------------------------------------------------------------
def N_cover(x, d, m):
    """``x``-covering number of the unit ball in ``R^d`` under the ``L^m`` metric.

    Requires ``0 < x <= 1`` and ``1 <= m <= inf``. Defined in [1]_, Section 2.2.

    For finite ``m`` this returns the ``m = inf`` value, which is an upper bound rather
    than a tight estimate; a warning is issued. The bounds on ``K_n(X)`` in this module
    always evaluate at ``m = inf``, so they are unaffected.
    """
    if m < np.inf:
        warnings.warn(
            "N_cover is only an upper bound for finite m; no strict bound is implemented.",
            stacklevel=2,
        )

    return np.ceil(1 / x) ** d


def K_cover(d, m):
    """Maximum of ``x**d * N_cover(x, d, m)`` over ``0 < x <= 1``.

    Defined in [1]_, Section 2.2, equation (2.3).
    """

    def objective(x):
        # The maximization range is (0, 1], but scipy optimizes over the closed [0, 1]
        # and N_cover is undefined at 0; substitute machine epsilon if we land there.
        if x == 0:
            x = np.finfo(type(x)).eps

        return -(x**d) * N_cover(x, d, m)

    # Minimize the negative to maximize the function.
    res = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    return -res.fun


# ---------------------------------------------------------------------------
# The constant kappa in the three cases of Fournier's Theorem 2.1
# ---------------------------------------------------------------------------
def kappa_1(d, p, m):
    """Case (i) of [1]_, Theorem 2.1. Requires ``p > d/2``, ``q > 2p``, ``m >= 1``."""
    if m == np.inf:
        return 2 ** (d / 2 - 1) / (1 - 2 ** (d / 2 - p))

    K = K_cover(d, m)

    def objective(r):
        top = 2 ** (p - 1 - d / 2) * r ** (p + d / 2)
        bot = (r - 1) ** p * (1 - r ** (d / 2 - p))
        return np.sqrt(K) * top / bot

    res = minimize_scalar(objective, bounds=(2, None), method="bounded")
    return res.fun


def kappa_2(N, d, p, m):
    """Case (ii) of [1]_, Theorem 2.1. Requires ``p == d/2``, ``q > 2p``, ``m >= 1``.

    Unlike the other two cases this is not a constant: it carries a ``log(sqrt(N))``
    factor, so it depends on the sample count it is used to bound.
    """
    if m == np.inf:
        log_term = np.log((2 ** (1 - p) - 2 ** (1 - 2 * p)) * np.sqrt(N))
        log_term = np.max([log_term, 0])

        return 2 ** (p - 1) / (p * np.log(2)) * log_term + 2 ** (p - 1) / (1 - 2 ** (-p))

    K = K_cover(d, m)

    def objective(r):
        log_term = np.log(2 ** (p + 1) * (r ** (-p) - r ** (-2 * p)) * np.sqrt(N / K))
        log_term = np.max([log_term, 0])

        term_1 = r ** (2 * p) / ((r - 1) ** p * p * np.log(r)) * log_term
        term_2 = r ** (3 * p) / ((r - 1) ** p * (r**p - 1))
        return np.sqrt(K) / 2 * (term_1 + term_2)

    res = minimize_scalar(objective, bounds=(2, None), method="bounded")
    return res.fun


def kappa_3(d, p, m):
    """Case (iii) of [1]_, Theorem 2.1. Requires ``0 < p < d/2``, ``q > dp/(d-p)``."""
    if m == np.inf:
        top = 2 ** (p - 2 * p / d) * (1 - 2 ** (-d / 2)) ** (1 - 2 * p / d)
        bot = 1 - 2 ** (p - d / 2)
        return top / bot

    K = K_cover(d, m)

    def objective(r):
        top = r ** (2 * p) * (1 - r ** (-d / 2)) ** (1 - 2 * p / d)
        bot = (r - 1) ** p * (1 - r ** (p - d / 2))
        return (K / 4) ** (p / d) * top / bot

    res = minimize_scalar(objective, bounds=(2, None), method="bounded")
    return res.fun


# ---------------------------------------------------------------------------
# Theorem 6.1 and Corollary 6.2: the expectation bound on K_n(X)
# ---------------------------------------------------------------------------
def _exponent(d, p):
    """The rate ``e_{d,p}`` of Thm 6.1: ``1/2`` for ``p >= d/2``, ``p/d`` below.

    At ``p == d/2`` the rate is also ``1/2``, with the logarithmic term inside the constant.
    """
    return p / d if p < d / 2 else 0.5


def _concentration(alpha):
    """The Weed--Bach deviation coefficient ``sqrt(-ln(1-alpha)/2)`` of Lemma 3.12."""
    return np.sqrt(-0.5 * np.log(1 - alpha))


def fournier_constant(n, p, N=None):
    """The constant ``C(d,p)`` of Thm 6.1, with ``d = n(n-1)/2``.

    Compiles the three cases of [1]_, Theorem 2.1, specialized to ``K_n(X)``.

    The general bound reads ``2^p * kappa * M^p * theta``. Since ``K_n(X)`` is compact we
    may take ``q = inf``, where ``theta -> 1`` and ``M = D/2``; the whole expression
    collapses to ``kappa * D^p``, so this function returns ``kappa`` alone and the
    diameter is applied by the callers below.

    Parameters
    ----------
    n : int
        Index of the curvature set (the tuple size).
    p : float
        Order of the Wasserstein distance.
    N : int, optional
        Sample count. Needed only in the boundary case ``p == d/2``, i.e.
        ``p == n(n-1)/4``, which the article excludes and where the constant carries a
        ``log(sqrt(N))`` factor.

    Notes
    -----
    Unit-free: assumes diameter 1. ``K_n(X)`` carries the ``L^inf`` metric.
    """
    d = n * (n - 1) / 2  # dimension of the sample space K_n(X)

    if p > d / 2:
        return kappa_1(d, p, m=np.inf)
    if p == d / 2:
        if N is None:
            raise ValueError(
                f"p == n(n-1)/4 == {p:g} is the boundary case of Theorem 2.1, where the "
                "constant carries a log(sqrt(N)) factor and is therefore not constant. "
                "Pass an explicit N (the sample count you intend to use, or an estimate "
                "of it) to evaluate it."
            )
        return kappa_2(N, d, p, m=np.inf)
    return kappa_3(d, p, m=np.inf)


def expected_wasserstein_bound(N, n, p, D=1, lip=1):
    """Upper bound for ``E[W_p^p]`` from N sampled n-tuples (Thm 6.1, Cor 6.2).

    The bound is ``C(d,p) (lip D)^p N^{-e_{d,p}}``. With ``lip = 1`` it bounds
    ``E[W_p^p(mu_{N,n}, mu_n)]`` (Thm 6.1); with the Lipschitz constant of an invariant F
    it bounds ``E[W_p^p(nu_{F,N}, nu_F)]`` (Cor 6.2).

    Parameters
    ----------
    N : int
        Number of sampled n-tuples.
    n : int
        Index of the curvature set (the tuple size).
    p : float
        Order of the Wasserstein distance.
    D : float, default 1
        Diameter of X. The default gives the bound in units of ``diam(X)^p``.
    lip : float, default 1
        Lipschitz constant of F with respect to the supremum norm on ``K_n``.
    """
    d = n * (n - 1) / 2
    C = fournier_constant(n, p, N=N)

    # For p == d/2 the logarithmic term already sits inside C.
    return lip**p * C * D**p * N ** (-_exponent(d, p))


# ---------------------------------------------------------------------------
# Corollaries 6.3 and 6.5: high-probability bounds and sample counts through K_n(X)
# ---------------------------------------------------------------------------
def wasserstein_bound(N, alpha, n, p, D=1, lip=1, lam=None):
    """High-probability bound on ``W_p`` from N sampled n-tuples (Cor 6.3, Cor 6.5).

    With probability at least ``alpha``,
    ``W_p(nu_{F,N}, nu_F) < (lip^p C N^{-e} + lam^p s N^{-1/2})^{1/p} D``, where
    ``s = sqrt(-ln(1-alpha)/2)``. With ``lip = lam = 1`` this bounds
    ``W_p(mu_{N,n}, mu_n)``. It is the inverse of :func:`sample_size`: at
    ``N = sample_size(epsilon, alpha, n, p, lip, lam)`` it equals ``epsilon D``.

    Uses the concentration of ``W_p^p`` around its expectation, after Weed and Bach [2]_,
    which is the sharper of the two routes and the one the article uses.

    Parameters are as in :func:`sample_size`, plus the diameter ``D`` of X (default 1, which
    gives the bound in units of ``diam(X)``).
    """
    if lam is None:
        lam = lip
    d = n * (n - 1) / 2
    C = fournier_constant(n, p, N=N)

    base_1 = lip**p * C * N ** (-_exponent(d, p))
    base_2 = lam**p * _concentration(alpha) * N ** (-1 / 2)

    return (base_1 + base_2) ** (1 / p) * D


def _wasserstein_bound_md(N, alpha, n, p, D=1, lip=1, lam=None):
    """High-probability bound on ``W_p`` via McDiarmid's inequality.

    Bounds the deviation of ``W_p`` from its expectation. Gives markedly worse constants
    than :func:`wasserstein_bound`; provided for comparison.
    """
    if lam is None:
        lam = lip
    d = n * (n - 1) / 2
    C = fournier_constant(n, p, N=N)

    e = _exponent(d, p)

    base_1 = lip * C ** (1 / p) * N ** (-e / p)
    base_2 = lam * _concentration(alpha) * N ** (1 / p - 1 / 2)

    return (base_1 + base_2) * D


def _solve_sample_count(f, initial, method):
    """Solve ``f(N) = 0`` for the sample count and reject a non-solution.

    ``fsolve`` returns its last iterate whether or not it converged, so a bound whose
    defining equation has no root would otherwise come back as a plausible-looking
    integer. Check the residual and the sign of the result before returning.
    """
    solution, info, status, message = fsolve(f, initial, full_output=True)
    n_samples = solution[0]
    if status != 1 or not np.isfinite(n_samples) or n_samples <= 0:
        raise ValueError(
            f"The {method} sample-size equation has no usable solution for these "
            f"parameters (solver said: {message.strip()}). "
            "This happens when the deviation term does not decay in N, which for McDiarmid "
            "is the case for every p <= 2. Use method='WB'."
        )
    return np.ceil(n_samples).astype(int)


def _sample_size_markov(epsilon, alpha, n, p, lip=1, N=None):
    """Minimum sample count via Markov's inequality.

    Closed form, but the bounds are impractically large. Use ``method="WB"``.
    Markov's inequality does not use the range of F, so there is no ``lam``.
    """
    d = n * (n - 1) / 2
    C = fournier_constant(n, p, N=N)

    e = d if p < d / 2 else 2 * p

    solution = (C / (1 - alpha)) ** (e / p) * (epsilon / lip) ** (-e)
    return np.ceil(solution).astype(int)


def _sample_size_md(epsilon, alpha, n, p, lip=1, lam=None, N=None):
    """Minimum sample count via McDiarmid's inequality.

    Provided for comparison; ``method="WB"`` is sharper and has no domain restriction.

    McDiarmid bounds the deviation of ``W_p`` rather than ``W_p^p``, which leaves a term
    scaling as ``N^{1/p - 1/2}``. That term only decays for ``p > 2``; at ``p <= 2`` it is
    constant or growing, the defining equation has no root, and this raises ``ValueError``.
    """
    if lam is None:
        lam = lip
    d = n * (n - 1) / 2
    C = fournier_constant(n, p, N=N)

    e = _exponent(d, p)
    s = _concentration(alpha)

    def f(x):
        return lip * C ** (1 / p) * x ** (-e / p) + lam * s * x ** (1 / p - 1 / 2) - epsilon

    initial = lip ** (p / e) * C ** (1 / e) * epsilon ** (-p / e)
    return _solve_sample_count(f, initial, "McDiarmid")


def _sample_size_wb(epsilon, alpha, n, p, lip=1, lam=None, N=None):
    """Minimum sample count via the Weed and Bach [2]_ concentration inequality.

    Solves ``lip^p C N^{-e} + lam^p sqrt(-ln(1-alpha)/2) N^{-1/2} = epsilon^p`` for N,
    the defining equation of Cor 6.5 (and of Cor 6.3 when ``lip = lam = 1``).
    """
    if lam is None:
        lam = lip
    d = n * (n - 1) / 2
    C = fournier_constant(n, p, N=N)

    e = _exponent(d, p)
    s = _concentration(alpha)

    def f(x):
        return lip**p * C * x ** (-e) + lam**p * s * x ** (-1 / 2) - epsilon**p

    initial = lip ** (p / e) * C ** (1 / e) * epsilon ** (-p / e)
    return _solve_sample_count(f, initial, "Weed-Bach")


def sample_size(epsilon, alpha, n, p, lip=1, lam=None, N=None, method="WB"):
    """Number of n-tuples sufficient to estimate ``mu_n``, or one invariant, to a given accuracy.

    Returns the smallest N for which the bounds of Cor 6.3 and Cor 6.5 guarantee that, with
    probability at least ``alpha``, the empirical measure of N sampled n-tuples is within
    ``epsilon * diam(X)`` of its target in ``W_p``. Fewer samples may suffice, but the
    bounds cannot certify them. The result depends neither on the number of points of X
    nor on its diameter.

    * ``lip = lam = 1`` (the default): the target is the curvature measure ``mu_n``
      (Cor 6.3). The same sample then estimates every Lipschitz invariant F at once, with
      error ``epsilon * Lip(F) * diam(X)``.
    * ``lip = Lip(F)``: the target is the distribution ``nu_F`` of the one invariant F,
      with error ``epsilon * diam(X)``. With ``lam`` at its default ``lip`` this is
      Cor 6.3 at accuracy ``epsilon / lip``; a smaller valid ``lam`` needs fewer samples
      (Cor 6.5).

    Parameters
    ----------
    epsilon : float
        Target accuracy, in units of ``diam(X)``. For an absolute tolerance ``t``, pass
        ``t / diam(X)``.
    alpha : float
        Confidence: the probability with which the accuracy is achieved.
    n : int
        Index of the curvature set, i.e. the number of points per sampled tuple.
    p : float
        Order of the Wasserstein distance.
    lip : float, default 1
        Lipschitz constant of the invariant ``F : K_n -> Z``, with respect to the supremum
        norm on distance matrices (Def 4.1). The pairwise distance and the
        (4,1)-persistence are 1-Lipschitz, the four-point hyperbolicity deficit is
        2-Lipschitz (Prop 4.8).
    lam : float, optional
        Range constant: any ``lam`` with ``diam F(K_n(X)) <= lam * diam(X)``. Defaults to
        ``lip``, which is always valid. All three invariants above admit ``lam = 1``.
    N : int, optional
        Initial guess for the sample count. Needed as the ``log(sqrt(N))`` argument in
        the boundary case ``p == n(n-1)/4``.
    method : {"WB", "MD", "Markov"}, default "WB"
        Concentration inequality to invert. "WB" (Weed and Bach) is sharpest and is what
        the article uses; the others are provided for comparison.

    Returns
    -------
    int
        The required number of sampled n-tuples.

    Notes
    -----
    ``K_n(X)`` carries the ``L^inf`` metric. Since finite spaces are compact, so is
    ``K_n(X)``, hence ``mu_n`` has finite moments of all orders and its moment of order
    infinity is the minimum enclosing radius ``diam(X)/2``.

    Examples
    --------
    The sample count of the substitution-model persistence distributions in the article
    (Section 8.3: four points, ``p = 2``, per-distribution confidence 0.995, which a union
    bound turns into 0.95 across the ten model-time pairs):

    >>> int(sample_size(epsilon=0.1, alpha=0.995, n=4, p=2))
    117207232

    The hyperbolicity deficit in the SARS-CoV-2 analysis (Section 8.4), which is
    2-Lipschitz with range constant 1:

    >>> int(sample_size(epsilon=0.1, alpha=0.95, n=4, p=1, lip=2, lam=1))
    3376863048
    """
    if method == "WB":
        return _sample_size_wb(epsilon, alpha, n, p, lip=lip, lam=lam, N=N)
    if method == "MD":
        warnings.warn(
            "method='MD' is provided for completeness, but gives unrealistically bad bounds.",
            stacklevel=2,
        )
        return _sample_size_md(epsilon, alpha, n, p, lip=lip, lam=lam, N=N)
    if method == "Markov":
        warnings.warn(
            "Markov's inequality gives impractical bounds. Use method='WB' instead.",
            stacklevel=2,
        )
        return _sample_size_markov(epsilon, alpha, n, p, lip=lip, N=N)

    raise ValueError(
        f"Unrecognized method {method!r}. Use one of 'WB' (preferred), 'MD', or 'Markov'."
    )

