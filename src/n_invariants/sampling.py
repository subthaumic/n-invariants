"""Empirical curvature measures and empirical invariant distributions, by sampling.

A finite metric measure space is given by its points ``X``, a metric, and a measure on the
points (uniform unless ``weights`` are given). Tuples of n points are drawn i.i.d. from the
measure, and only the distances within each tuple are computed, so no pairwise distance
matrix of X is formed.

:func:`empirical_curvature_measure`
    The empirical curvature measure ``mu_{N,n}``: the distances within each of N sampled
    n-tuples, one row per tuple, each an atom of mass ``1/N``.
:func:`empirical_distribution`
    The empirical distribution ``nu_{F,N} = F_#(mu_{N,n})`` of an invariant F, as a
    histogram. The tuples are processed in batches and only the histogram is kept, so
    memory does not grow with N.

:func:`n_invariants.sample_size` gives the N for a target accuracy and confidence. With
the same ``seed``, ``n_samples`` and ``batch_size``, both functions draw the same tuples:
:func:`empirical_distribution` is then exactly the histogram of F applied to the output of
:func:`empirical_curvature_measure`.

The metric is one of

``"euclidean"``
    ``X`` has shape ``(U, d)``.
``"hamming"``
    The number of coordinates in which two points differ; ``X`` has shape ``(U, L)``,
    e.g. aligned sequences encoded as integers.
``"precomputed"``
    ``X`` is a ``(U, U)`` distance matrix, and the points are its row indices.
a function ``metric(A, B)``
    Returns the distances between corresponding rows of two equally long arrays of
    points ``A = X[i]`` and ``B = X[j]``.

The N samples are split into batches of ``batch_size``, each with its own seed derived from
``seed``. The result depends only on ``(seed, n_samples, batch_size)``, so ``n_workers``
changes the speed but not the result.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

__all__ = ["empirical_curvature_measure", "empirical_distribution"]

DEFAULT_BATCH_SIZE = 100_000

# The batch task, published to forked worker processes instead of being pickled.
_TASK = None


def empirical_curvature_measure(
    X, n, n_samples, *, metric, weights=None, seed=None, batch_size=None, n_workers=1
):
    """The empirical curvature measure ``mu_{N,n}`` of ``N = n_samples`` sampled n-tuples.

    Parameters
    ----------
    X : array_like
        The points, indexable by integer arrays, or a distance matrix if
        ``metric="precomputed"``.
    n : int
        Points per tuple.
    n_samples : int
        Number of tuples N, e.g. from :func:`n_invariants.sample_size`.
    metric : str or callable
        ``"euclidean"``, ``"hamming"``, ``"precomputed"``, or a function
        ``metric(A, B)``; see the module docstring.
    weights : array_like of shape (U,), optional
        The measure on the points, up to normalization. Uniform if None.
    seed : int or numpy.random.SeedSequence, optional
        Together with ``n_samples`` and ``batch_size``, determines the result.
    batch_size : int, optional
        Tuples per batch, default 100,000.
    n_workers : int, default 1
        Worker processes. Parallel runs use the ``fork`` start method and run serially
        where it is unavailable.

    Returns
    -------
    ndarray of shape (n_samples, n(n-1)/2)
        The distances within each tuple, in scipy's condensed order
        ``(0,1), (0,2), ..., (n-2,n-1)``. Each row is an atom of mass ``1/n_samples``.
        This array holds all samples; for large ``n_samples`` use
        :func:`empirical_distribution`, which keeps only a histogram.
    """
    distances, U = _paired_distances(X, metric)
    p = _measure(weights, U)
    n = _check_n(n)

    def task(batch_seed, size):
        return _sample_rows(distances, U, n, p, batch_seed, size)

    return np.concatenate(_map(task, _batches(n_samples, batch_size, seed), n_workers), axis=0)


def empirical_distribution(
    X, invariant, n, n_samples, *, metric, weights=None, seed=None, batch_size=None,
    decimals=None, n_workers=1,
):
    """The empirical distribution ``nu_{F,N}`` of an invariant F, as a histogram.

    Parameters
    ----------
    X : array_like
        The points, indexable by integer arrays, or a distance matrix if
        ``metric="precomputed"``.
    invariant : callable
        The invariant F: maps condensed distance rows of shape ``(size, n(n-1)/2)`` to one
        value per row, an array of shape ``(size,)`` or ``(size, k)``. See
        :mod:`n_invariants.invariants`.
    n : int
        Points per tuple, as F requires.
    n_samples : int
        Number of tuples N, e.g. from :func:`n_invariants.sample_size`.
    metric : str or callable
        ``"euclidean"``, ``"hamming"``, ``"precomputed"``, or a function
        ``metric(A, B)``; see the module docstring.
    weights : array_like of shape (U,), optional
        The measure on the points, up to normalization. Uniform if None.
    seed : int or numpy.random.SeedSequence, optional
        Together with ``n_samples`` and ``batch_size``, determines the result.
    batch_size : int, optional
        Tuples per batch, default 100,000. Bounds the memory in use.
    decimals : int, optional
        Round the values of F to this many decimals before counting them. Useful for
        real-valued metrics, where unrounded values are almost all distinct.
    n_workers : int, default 1
        Worker processes. Parallel runs use the ``fork`` start method and run serially
        where it is unavailable.

    Returns
    -------
    values : ndarray of shape (V,) or (V, k)
        The distinct values of F, sorted.
    counts : ndarray of shape (V,), dtype int64
        How many of the ``n_samples`` tuples take each value.
    """
    distances, U = _paired_distances(X, metric)
    p = _measure(weights, U)
    n = _check_n(n)

    def task(batch_seed, size):
        rows = _sample_rows(distances, U, n, p, batch_seed, size)
        values = np.asarray(invariant(rows))
        if values.ndim not in (1, 2) or values.shape[0] != size:
            raise TypeError(
                "invariant must return one value per tuple, an array of shape (size,) or "
                "(size, k); the histogram forms hyperbolicity_vec and persistence_set_vec "
                "are not invariants in this sense."
            )
        if decimals is not None:
            values = np.round(values, decimals)
        return _histogram(values)

    return _merge(_map(task, _batches(n_samples, batch_size, seed), n_workers))


# ---------------------------------------------------------------------------
# Metric, measure and tuples
# ---------------------------------------------------------------------------
def _paired_distances(X, metric):
    """``(distances, U)``, where ``distances(i, j)`` are the distances of points i and j, row-wise."""
    if isinstance(metric, str) and metric == "precomputed":
        D = np.asarray(X, dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError('metric="precomputed" needs a square distance matrix X.')
        return (lambda i, j: D[i, j]), D.shape[0]

    points = np.asarray(X)
    if points.ndim == 0:
        raise ValueError("X must hold at least one point.")
    if callable(metric):
        return (lambda i, j: np.asarray(metric(points[i], points[j]), dtype=float)), len(points)
    if points.ndim == 1:
        points = points[:, None]
    if metric == "euclidean":
        points = points.astype(float, copy=False)
        return (lambda i, j: np.sqrt(np.sum((points[i] - points[j]) ** 2, axis=1))), len(points)
    if metric == "hamming":
        return (lambda i, j: np.count_nonzero(points[i] != points[j], axis=1).astype(float)), len(points)
    raise ValueError(
        f"Unknown metric {metric!r}. Use 'euclidean', 'hamming', 'precomputed' or a function."
    )


def _measure(weights, U):
    """Normalized sampling probabilities, or None for the uniform measure."""
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float)
    if w.shape != (U,):
        raise ValueError(f"weights must have shape ({U},), one weight per point.")
    if np.any(w < 0) or not np.sum(w) > 0:
        raise ValueError("weights must be nonnegative with a positive sum.")
    return w / np.sum(w)


def _check_n(n):
    n = int(n)
    if n < 2:
        raise ValueError("A tuple needs at least n = 2 points.")
    return n


def _sample_rows(distances, U, n, p, batch_seed, size):
    """Draw ``size`` n-tuples i.i.d. from the measure and return their condensed distance rows.

    Points are drawn with replacement, so a tuple may repeat a point; this is i.i.d.
    sampling from the measure, which is what the curvature measure is defined by.
    """
    idx = np.random.default_rng(batch_seed).choice(U, size=(size, n), p=p)
    rows = np.empty((size, n * (n - 1) // 2), dtype=float)
    for col, (a, b) in enumerate(combinations(range(n), 2)):
        rows[:, col] = distances(idx[:, a], idx[:, b])
    return rows


# ---------------------------------------------------------------------------
# Batches, workers and histograms
# ---------------------------------------------------------------------------
def _batches(n_samples, batch_size, seed):
    """The fixed decomposition into ``(seed, size)`` batches."""
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    batch = DEFAULT_BATCH_SIZE if batch_size is None else int(batch_size)
    if batch < 1:
        raise ValueError("batch_size must be at least 1.")
    sizes = [batch] * (n_samples // batch)
    if n_samples % batch:
        sizes.append(n_samples % batch)
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return list(zip(root.spawn(len(sizes)), sizes))


def _map(task, batches, n_workers):
    """``task(seed, size)`` for every batch, in batch order."""
    global _TASK

    n_workers = max(1, min(int(n_workers), len(batches)))
    if n_workers == 1:
        return [task(batch_seed, size) for batch_seed, size in batches]
    try:
        context = mp.get_context("fork")
    except ValueError:
        return [task(batch_seed, size) for batch_seed, size in batches]

    _TASK = task
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as executor:
            chunksize = max(1, len(batches) // (4 * n_workers))
            return list(executor.map(_run_task, batches, chunksize=chunksize))
    finally:
        _TASK = None


def _run_task(batch):
    batch_seed, size = batch
    return _TASK(batch_seed, size)


def _histogram(values):
    """``(distinct values, counts)`` of an array of shape ``(size,)`` or ``(size, k)``."""
    axis = 0 if values.ndim == 2 else None
    distinct, counts = np.unique(values, axis=axis, return_counts=True)
    return distinct, counts.astype(np.int64)


def _merge(histograms):
    """Sum batch histograms into one, with the distinct values sorted."""
    values = np.concatenate([h[0] for h in histograms], axis=0)
    counts = np.concatenate([h[1] for h in histograms])
    axis = 0 if values.ndim == 2 else None
    distinct, inverse = np.unique(values, axis=axis, return_inverse=True)
    total = np.zeros(distinct.shape[0], dtype=np.int64)
    np.add.at(total, inverse.reshape(-1), counts)
    return distinct, total
