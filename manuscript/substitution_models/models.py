"""Single-nucleotide substitution models and the exact laws of their n-point invariants.

Two homogeneous product models over an alphabet of size ``k`` and sequence length ``L``,
started from the point mass ``delta_0^{(x)L}``:

* **JC** (Jukes-Cantor): a symmetric generator with total exit rate ``beta`` from every
  symbol. Its stationary law is uniform.
* **HB** (Halpern-Bruno): the same JC baseline reweighted by a per-symbol fitness profile
  ``phi``, so that mutations toward fitter symbols are favoured. Its stationary law is
  proportional to ``exp(phi)``.

Because both models are products over sites, the one-site law ``rho_t`` determines the
distributions of all n-point invariants:

* the **pairwise distance** is ``Bin(L, 1 - c_2(rho_t))``, where ``c_2(rho) = sum rho^2``
  is the match probability (see :func:`binomial_pmf_with_coefficients`);
* the **four-point hyperbolicity deficit** follows from the one-site law of the three
  opposite pair sums, convolved L times by FFT and pushed forward along the gap between
  the largest and median sum (see :func:`hyperbolicity_pmf_from_rho`);
* the **(4,1)-persistence** has no comparable closed form and is estimated by the
  sampler in ``sampling.py``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "jc_single_site_distribution",
    "hb_factor",
    "build_hb_generator",
    "matrix_exponential_eig",
    "hb_single_site_distribution",
    "hb_stationary_distribution",
    "sample_sequences",
    "jc_distance_probability",
    "distance_probability_from_site_distribution",
    "log_binomial_coefficients",
    "binomial_pmf_with_coefficients",
    "single_site_pair_sum_law",
    "build_delta_tick_grid",
    "pair_sum_law_from_single_site_law",
    "hyperbolicity_pmf_from_rho",
]


# ---------------------------------------------------------------------------
# Single-site substitution laws
# ---------------------------------------------------------------------------
def jc_single_site_distribution(
    t: float, alphabet_size: int, beta: float, start_symbol: int = 0
) -> np.ndarray:
    """One-site law ``rho_t`` of the JC model started at ``start_symbol``."""
    decay = math.exp(-alphabet_size * beta * t / (alphabet_size - 1))
    rho = np.full(alphabet_size, (1.0 - decay) / alphabet_size, dtype=float)
    rho[start_symbol] += decay
    return rho


def hb_factor(delta_phi: float) -> float:
    """Stable Halpern-Bruno factor ``x / (1 - exp(-x))``, with limit 1 at 0."""
    x = float(delta_phi)
    if abs(x) < 1.0e-10:
        return 1.0
    return x / (-math.expm1(-x))


def build_hb_generator(alphabet_size: int, beta: float, phi: np.ndarray) -> np.ndarray:
    """Build the HB single-site generator from a symmetric JC baseline."""
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least 2.")
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if phi.shape != (alphabet_size,):
        raise ValueError(f"Expected {alphabet_size} HB fitness values, got {phi.shape[0]}.")

    q = np.zeros((alphabet_size, alphabet_size), dtype=float)
    baseline_rate = beta / (alphabet_size - 1)
    for sigma in range(alphabet_size):
        for tau in range(alphabet_size):
            if sigma == tau:
                continue
            q[sigma, tau] = baseline_rate * hb_factor(phi[tau] - phi[sigma])
        q[sigma, sigma] = -float(np.sum(q[sigma, :]))
    return q


def matrix_exponential_eig(matrix: np.ndarray) -> np.ndarray:
    """Small-matrix exponential via eigendecomposition."""
    values, vectors = np.linalg.eig(matrix)
    inverse_vectors = np.linalg.inv(vectors)
    exp_matrix = vectors @ np.diag(np.exp(values)) @ inverse_vectors
    return np.asarray(np.real_if_close(exp_matrix, tol=1000), dtype=float)


def hb_single_site_distribution(
    t: float, alphabet_size: int, beta: float, phi: np.ndarray, start_symbol: int = 0
) -> np.ndarray:
    """One-site law ``rho_t = delta_0 exp(tQ)`` of the HB model, by matrix exponentiation."""
    q = build_hb_generator(alphabet_size, beta, phi)
    rho_0 = np.zeros(alphabet_size, dtype=float)
    rho_0[start_symbol] = 1.0
    rho_t = rho_0 @ matrix_exponential_eig(q * t)
    rho_t = np.maximum(np.real_if_close(rho_t, tol=1000).astype(float), 0.0)
    total = float(np.sum(rho_t))
    if total <= 0.0:
        raise ValueError("HB single-site distribution has nonpositive total mass.")
    return rho_t / total


def hb_stationary_distribution(phi: np.ndarray) -> np.ndarray:
    """Stationary law for HB rates with a symmetric JC baseline: proportional to ``exp(phi)``."""
    shifted = phi - float(np.max(phi))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def sample_sequences(
    rho: np.ndarray,
    length: int,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``samples`` sequences drawn i.i.d. from the product measure ``rho^{(x)length}``.

    Symbols are ``0, ..., len(rho) - 1``, as ``uint8``. This is the law of a sequence at the
    model time whose one-site law is ``rho``.
    """
    alphabet = np.arange(rho.size, dtype=np.uint8)
    return rng.choice(alphabet, size=(samples, length), replace=True, p=rho).astype(
        np.uint8,
        copy=False,
    )


# ---------------------------------------------------------------------------
# Pairwise distance law: Bin(L, 1 - c_2(rho))
# ---------------------------------------------------------------------------
def jc_distance_probability(t: float, alphabet_size: int, beta: float) -> float:
    """Per-site mismatch probability between two JC sequences from a common start."""
    return (
        (alphabet_size - 1)
        / alphabet_size
        * (1.0 - math.exp(-2.0 * alphabet_size * beta * t / (alphabet_size - 1)))
    )


def distance_probability_from_site_distribution(rho: np.ndarray) -> float:
    """Per-site mismatch probability ``1 - c_2(rho)`` for two points drawn from ``rho``."""
    return 1.0 - float(np.sum(rho * rho))


def log_binomial_coefficients(length: int) -> np.ndarray:
    """``log binom(length, k)`` for ``k = 0..length``, computed once and reused."""
    return np.array(
        [
            math.lgamma(length + 1) - math.lgamma(k + 1) - math.lgamma(length - k + 1)
            for k in range(length + 1)
        ],
        dtype=float,
    )


def binomial_pmf_with_coefficients(length: int, p: float, log_binom: np.ndarray) -> np.ndarray:
    """``Bin(length, p)`` in log space, reusing precomputed log binomial coefficients.

    Working in log space and subtracting the maximum before exponentiating keeps the pmf
    accurate in the far tails, which the log-scale figure axes make visible.
    """
    p = min(max(float(p), 0.0), 1.0)
    if p == 0.0:
        pmf = np.zeros(length + 1, dtype=float)
        pmf[0] = 1.0
        return pmf
    if p == 1.0:
        pmf = np.zeros(length + 1, dtype=float)
        pmf[-1] = 1.0
        return pmf

    k = np.arange(length + 1, dtype=float)
    log_p = math.log(p)
    log_q = math.log1p(-p)
    log_values = log_binom + k * log_p + (length - k) * log_q
    log_values -= float(np.max(log_values))
    pmf = np.exp(log_values)
    return pmf / float(np.sum(pmf))


# ---------------------------------------------------------------------------
# Four-point hyperbolicity law: one-site pair sums, convolved L times by FFT
# ---------------------------------------------------------------------------
def single_site_pair_sum_law(rho: np.ndarray) -> np.ndarray:
    """One-site joint law of the three opposite pair sums, as a ``(3, 3, 3)`` array.

    Enumerates all ``k^4`` symbol assignments to a quadruple at one site and accumulates
    the probability of each triple ``(d12+d34, d13+d24, d14+d23)``, each of which is 0, 1
    or 2 at a single site.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1 or rho.size < 2:
        raise ValueError("rho must be a one-dimensional probability vector.")
    total = float(np.sum(rho))
    if total <= 0.0:
        raise ValueError("rho must have positive total mass.")
    rho = rho / total

    law = np.zeros((3, 3, 3), dtype=float)
    alphabet_size = rho.size
    for x1 in range(alphabet_size):
        for x2 in range(alphabet_size):
            for x3 in range(alphabet_size):
                for x4 in range(alphabet_size):
                    probability = rho[x1] * rho[x2] * rho[x3] * rho[x4]
                    p12_34 = int(x1 != x2) + int(x3 != x4)
                    p13_24 = int(x1 != x3) + int(x2 != x4)
                    p14_23 = int(x1 != x4) + int(x2 != x3)
                    law[p12_34, p13_24, p14_23] += float(probability)
    law /= float(np.sum(law))
    return law


def build_delta_tick_grid(length: int) -> np.ndarray:
    """Precompute ``2*delta`` for every possible length-L pair-sum triple.

    The deficit is half the gap between the largest and the *median* of the three sums,
    so ``2*delta = max - median`` is an integer, the "tick" by which the pmf is indexed.
    """
    max_pair_sum = 2 * length
    coords = np.arange(max_pair_sum + 1, dtype=np.int16)
    p2, p3 = np.meshgrid(coords, coords, indexing="ij")
    ticks = np.empty((max_pair_sum + 1, max_pair_sum + 1, max_pair_sum + 1), dtype=np.int16)
    for p1 in range(max_pair_sum + 1):
        max_values = np.maximum(p1, np.maximum(p2, p3))
        min_values = np.minimum(p1, np.minimum(p2, p3))
        median_values = p1 + p2 + p3 - max_values - min_values
        ticks[p1, :, :] = max_values - median_values
    return ticks


def pair_sum_law_from_single_site_law(length: int, single_site_law: np.ndarray) -> np.ndarray:
    """L-fold convolution of the one-site pair-sum law, by FFT over the three axes."""
    shape = (2 * length + 1, 2 * length + 1, 2 * length + 1)
    grid = np.zeros(shape, dtype=float)
    grid[:3, :3, :3] = single_site_law

    axes = (0, 1, 2)
    spectrum = np.fft.rfftn(grid, axes=axes)
    spectrum **= length
    law = np.fft.irfftn(spectrum, s=shape, axes=axes)
    law = np.real_if_close(law, tol=1000).astype(float)
    law = np.maximum(law, 0.0)
    total = float(np.sum(law))
    if total <= 0.0:
        raise ValueError("Pair-sum law has nonpositive total mass.")
    return law / total


def hyperbolicity_pmf_from_rho(
    length: int, rho: np.ndarray, delta_tick_grid: np.ndarray
) -> np.ndarray:
    """Exact pmf of ``2*delta`` for a homogeneous product measure with one-site law ``rho``."""
    site_law = single_site_pair_sum_law(rho)
    pair_sum_law = pair_sum_law_from_single_site_law(length, site_law)
    tick_pmf = np.bincount(
        delta_tick_grid.ravel(), weights=pair_sum_law.ravel(), minlength=2 * length + 1
    )
    tick_pmf = np.maximum(tick_pmf, 0.0)
    return tick_pmf / float(np.sum(tick_pmf))

