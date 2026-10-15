"""Sampled n-tuples, empirical curvature measures and empirical invariant distributions.

A finite metric measure space is given by its points ``X``, a metric, and a measure on the
points (uniform unless ``weights`` are given). Tuples of n points are drawn i.i.d. from the
measure, and only the distances within each tuple are computed, so no pairwise distance
matrix of X is formed.

:func:`sample_tuples`
    The sampled n-tuples, lazily, in batches: the indices of their points and the
    distances among them.
:func:`empirical_curvature_measure`
    The empirical curvature measure ``mu_{N,n}``: the distances within each of N sampled
    n-tuples, one row per tuple, each an atom of mass ``1/N``. It holds all N rows, so it
    refuses to allocate more than ``max_bytes`` in memory, and can write to a file instead.
:func:`empirical_distributions`
    The empirical distributions ``nu_{F,N} = F_#(mu_{N,n})`` of several n-invariants, as
    histograms, all from the same tuples. The tuples are processed in batches and only the
    histograms are kept, so memory does not grow with N.
:func:`empirical_distribution`
    The same for a single n-invariant.

The sample
----------
The seed, the points with their weights, and n fix one infinite i.i.d. sequence of n-tuples.
``n_samples`` only says how many of them to use, and ``batch_size`` and ``n_workers`` change
the speed and the memory in use but not the tuples. So all four functions draw the same
tuples for the same seed, and a sample can be revisited later for other n-invariants.
Moreover, the first M tuples of a sample of size N are the sample of size M.

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
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations

import numpy as np

__all__ = [
    "sample_tuples",
    "empirical_curvature_measure",
    "empirical_distributions",
    "empirical_distribution",
]

DEFAULT_BATCH_SIZE = 100_000
DEFAULT_MAX_BYTES = 4 * 2**30

# Tuples per independently seeded block of the random sequence. Fixed, so that the tuples
# do not depend on batch_size; batches may cut through blocks.
BLOCK_SIZE = 1024

# Batch histograms merged at a time, so that their number does not grow with N.
_MERGE_EVERY = 64

# The batch task, published to forked worker processes instead of being pickled.
_TASK = None


def sample_tuples(X, n, n_samples, *, metric, weights=None, seed=None, batch_size=None):
    """Draw ``N = n_samples`` n-tuples lazily, one batch at a time.

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
        Fixes the tuples; see the module docstring.
    batch_size : int, optional
        Tuples per batch, default 100,000.

    Yields
    ------
    indices : ndarray of shape (size, n)
        The points of each tuple, as indices into ``X``.
    distances : ndarray of shape (size, n(n-1)/2)
        The distances within each tuple, in scipy's condensed order
        ``(0,1), (0,2), ..., (n-2,n-1)``.
    """
    sampler = _Sampler(X, n, metric, weights, seed)
    batches = _batches(n_samples, batch_size)
    return (sampler.draw(start, stop) for start, stop in batches)


def empirical_curvature_measure(
    X, n, n_samples, *, metric, weights=None, seed=None, batch_size=None, n_workers=1,
    progress=True, out=None, max_bytes=DEFAULT_MAX_BYTES,
):
    """The empirical curvature measure ``mu_{N,n}`` of ``N = n_samples`` sampled n-tuples.

    The result holds ``N * n(n-1)/2`` distances of 8 bytes each. Above ``max_bytes`` it is
    not allocated in memory: pass ``out`` to write it to a file instead, or use
    :func:`empirical_distributions`, which keeps only histograms.

    Parameters
    ----------
    X, n, n_samples, metric, weights, seed, batch_size
        As in :func:`sample_tuples`.
    n_workers : int, default 1
        Worker processes, or all available CPUs if 0 or negative. Parallel runs use the
        ``fork`` start method and run serially where it is unavailable.
    progress : bool, default True
        Print the percentage of batches done to stderr, in steps of 5%.
    out : str or path, optional
        Write the result to this ``.npy`` file and return it as a memory map.
    max_bytes : int or None, default 4 GiB
        The largest result to allocate in memory when ``out`` is not given. None removes
        the limit.

    Returns
    -------
    ndarray of shape (n_samples, n(n-1)/2)
        The distances within each tuple, in scipy's condensed order
        ``(0,1), (0,2), ..., (n-2,n-1)``. Each row is an atom of mass ``1/n_samples``.
    """
    sampler = _Sampler(X, n, metric, weights, seed)
    batches = _batches(n_samples, batch_size)
    shape = (batches[-1][1], sampler.n * (sampler.n - 1) // 2)
    nbytes = shape[0] * shape[1] * np.dtype(float).itemsize

    if out is not None:
        result = np.lib.format.open_memmap(out, mode="w+", dtype=float, shape=shape)
    elif max_bytes is not None and nbytes > max_bytes:
        raise ValueError(
            f"The empirical curvature measure of {shape[0]:,} tuples takes "
            f"{nbytes / 2**30:.1f} GiB, more than max_bytes = {max_bytes / 2**30:.1f} GiB. "
            "Pass out= to write it to a file, raise max_bytes, or use "
            "empirical_distributions, which keeps only histograms."
        )
    else:
        result = np.empty(shape)

    def task(start, stop):
        return sampler.draw(start, stop)[1]

    for (start, stop), rows in zip(batches, _ordered_map(task, batches, n_workers, progress)):
        result[start:stop] = rows
    if out is not None:
        result.flush()
    return result


def empirical_distributions(
    X, invariants, n, n_samples, *, metric, weights=None, seed=None, batch_size=None,
    decimals=None, n_workers=1, progress=True,
):
    """The empirical distributions of several n-invariants, from the same sampled tuples.

    The distances of each batch are computed once and passed to every n-invariant, so
    evaluating several costs little more than evaluating one.

    Parameters
    ----------
    X : array_like
        The points, indexable by integer arrays, or a distance matrix if
        ``metric="precomputed"``.
    invariants : mapping of str to n-invariant, or iterable of Invariant
        The n-invariants. An :class:`n_invariants.invariants.Invariant` on k points is
        evaluated on the first k points of each n-tuple, ``k <= n``. A plain function
        receives the full condensed rows of shape ``(size, n(n-1)/2)``. Each must return
        one value per row, an array of shape ``(size,)`` or ``(size, k)``.
    n : int
        Points per tuple, at least the n of every Invariant.
    n_samples, metric, weights, seed, batch_size
        As in :func:`sample_tuples`.
    decimals : int, optional
        Round the values to this many decimals before counting them. Useful for
        real-valued metrics, where unrounded values are almost all distinct.
    n_workers : int, default 1
        Worker processes, or all available CPUs if 0 or negative. Parallel runs use the
        ``fork`` start method and run serially where it is unavailable.
    progress : bool, default True
        Print the percentage of batches done to stderr, in steps of 5%.

    Returns
    -------
    dict of str to (values, counts)
        Per n-invariant, the distinct values, sorted, as an array of shape ``(V,)`` or
        ``(V, k)``, and how many of the ``n_samples`` tuples take each value.
    """
    sampler = _Sampler(X, n, metric, weights, seed)
    named = _named_invariants(invariants, sampler.n)
    batches = _batches(n_samples, batch_size)

    def task(start, stop):
        rows = sampler.draw(start, stop)[1]
        return {name: _histogram(_values(F, rows, decimals)) for name, F in named}

    return _reduced_map(task, batches, n_workers, progress)


def empirical_distribution(
    X, invariant, n, n_samples, *, metric, weights=None, seed=None, batch_size=None,
    decimals=None, n_workers=1, progress=True,
):
    """The empirical distribution ``nu_{F,N}`` of one n-invariant F, as a histogram.

    :func:`empirical_distributions` for a single n-invariant; the parameters are the same.

    Returns
    -------
    values : ndarray of shape (V,) or (V, k)
        The distinct values of F, sorted.
    counts : ndarray of shape (V,), dtype int64
        How many of the ``n_samples`` tuples take each value.
    """
    return empirical_distributions(
        X, {"F": invariant}, n, n_samples, metric=metric, weights=weights, seed=seed,
        batch_size=batch_size, decimals=decimals, n_workers=n_workers, progress=progress,
    )["F"]


# ---------------------------------------------------------------------------
# Metric, measure and tuples
# ---------------------------------------------------------------------------
class _Sampler:
    """Draws the tuples at positions ``start:stop`` of the sequence fixed by the seed."""

    def __init__(self, X, n, metric, weights, seed):
        self.distances, self.U = _paired_distances(X, metric)
        p = _measure(weights, self.U)
        if p is None:
            self.cdf = None
        else:
            self.cdf = np.cumsum(p)
            self.cdf /= self.cdf[-1]
        self.n = _check_n(n)
        self.root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)

    def _block(self, b):
        """The b-th block of BLOCK_SIZE tuples, as point indices."""
        child = np.random.SeedSequence(
            self.root.entropy, spawn_key=self.root.spawn_key + (b,), pool_size=self.root.pool_size
        )
        rng = np.random.default_rng(child)
        if self.cdf is None:
            return rng.integers(0, self.U, size=(BLOCK_SIZE, self.n))
        # Inverse transform sampling; points of weight zero have empty intervals.
        return np.searchsorted(self.cdf, rng.random((BLOCK_SIZE, self.n)), side="right")

    def draw(self, start, stop):
        """``(indices, distances)`` of the tuples ``start, ..., stop - 1``.

        Points are drawn with replacement, so a tuple may repeat a point; this is i.i.d.
        sampling from the measure, which is what the curvature measure is defined by.
        """
        first, last = start // BLOCK_SIZE, (stop - 1) // BLOCK_SIZE
        offset = first * BLOCK_SIZE
        idx = np.concatenate([self._block(b) for b in range(first, last + 1)])
        idx = idx[start - offset : stop - offset]

        rows = np.empty((stop - start, self.n * (self.n - 1) // 2), dtype=float)
        for col, (a, b) in enumerate(combinations(range(self.n), 2)):
            rows[:, col] = self.distances(idx[:, a], idx[:, b])
        return idx, rows


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


# ---------------------------------------------------------------------------
# n-invariants and histograms
# ---------------------------------------------------------------------------
def _named_invariants(invariants, n):
    """``[(name, F), ...]`` from a mapping or an iterable, checked against the tuple size n."""
    if callable(invariants) and not isinstance(invariants, Mapping):
        raise TypeError("Pass the n-invariants as a mapping {name: F} or a list; "
                        "for a single one, use empirical_distribution.")
    if isinstance(invariants, Mapping):
        named = list(invariants.items())
    else:
        named = [(getattr(F, "name", None) or getattr(F, "__name__", None), F) for F in invariants]
    names = [name for name, _ in named]
    if len(set(names)) != len(names) or None in names:
        raise ValueError(f"The n-invariants need distinct names; got {names}.")
    for name, F in named:
        if getattr(F, "n", 0) > n:
            raise ValueError(f"{name} needs tuples of {F.n} points, but the sample has n = {n}.")
    return named


def _values(F, rows, decimals):
    values = np.asarray(F(rows))
    if values.ndim not in (1, 2) or values.shape[0] != rows.shape[0]:
        raise TypeError(
            "An n-invariant must return one value per tuple, an array of shape (size,) or "
            "(size, k); the histogram forms hyperbolicity_vec and persistence_set_vec are "
            "not n-invariants in this sense."
        )
    if decimals is not None:
        values = np.round(values, decimals)
    return values


def _histogram(values):
    """``(distinct values, counts)`` of an array of shape ``(size,)`` or ``(size, k)``."""
    axis = 0 if values.ndim == 2 else None
    distinct, counts = np.unique(values, axis=axis, return_counts=True)
    return distinct, counts.astype(np.int64)


def _merge(histograms):
    """Sum histograms into one, with the distinct values sorted."""
    values = np.concatenate([h[0] for h in histograms], axis=0)
    counts = np.concatenate([h[1] for h in histograms])
    axis = 0 if values.ndim == 2 else None
    distinct, inverse = np.unique(values, axis=axis, return_inverse=True)
    total = np.zeros(distinct.shape[0], dtype=np.int64)
    np.add.at(total, inverse.reshape(-1), counts)
    return distinct, total


def _merge_named(results):
    """Merge a list of ``{name: histogram}`` dictionaries name by name."""
    return {name: _merge([r[name] for r in results]) for name in results[0]}


def _histograms_of(task, batches):
    """Run ``task`` on the batches and merge as they come, ``_MERGE_EVERY`` at a time."""
    pending = []
    for start, stop in batches:
        pending.append(task(start, stop))
        if len(pending) > _MERGE_EVERY:
            pending = [_merge_named(pending)]
    return _merge_named(pending)


# ---------------------------------------------------------------------------
# Batches and workers
# ---------------------------------------------------------------------------
def _batches(n_samples, batch_size):
    """The ``(start, stop)`` positions of the batches."""
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    batch = DEFAULT_BATCH_SIZE if batch_size is None else int(batch_size)
    if batch < 1:
        raise ValueError("batch_size must be at least 1.")
    return [(start, min(start + batch, n_samples)) for start in range(0, n_samples, batch)]


def _available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _worker_count(n_workers, n_tasks):
    """Processes to use: all available CPUs if ``n_workers < 1``, at most one per task."""
    n_workers = int(n_workers)
    if n_workers < 1:
        n_workers = _available_cpus()
    return max(1, min(n_workers, n_tasks))


class _Progress:
    """Counts finished batches across forked workers and prints every 5% to stderr."""

    def __init__(self, total):
        self.total = total
        self.done = mp.Value("q", 0)  # shared memory, inherited by forked workers

    def step(self):
        # Printing under the lock keeps the percentages in order.
        with self.done.get_lock():
            self.done.value += 1
            percent = 100 * self.done.value // self.total
            if percent // 5 > 100 * (self.done.value - 1) // self.total // 5:
                print(f"  {percent - percent % 5:3d}% of batches ({datetime.now():%H:%M:%S})",
                      file=sys.stderr, flush=True)


def _counted(task, batches, progress):
    """``task``, reporting the progress over the batches if ``progress`` is set."""
    if not progress:
        return task
    counter = _Progress(len(batches))

    def counted_task(start, stop):
        result = task(start, stop)
        counter.step()
        return result

    return counted_task


def _fork_pool(n_workers):
    """A process pool whose workers inherit ``_TASK`` by forking, or None to run serially."""
    if n_workers <= 1:
        return None
    try:
        context = mp.get_context("fork")
    except ValueError:
        return None
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=context)


def _ordered_map(task, batches, n_workers, progress=False):
    """``task(start, stop)`` for every batch, yielded in batch order."""
    global _TASK

    n_workers = _worker_count(n_workers, len(batches))
    task = _counted(task, batches, progress)
    _TASK = task
    try:
        pool = _fork_pool(n_workers)
        if pool is None:
            yield from (task(start, stop) for start, stop in batches)
            return
        with pool:
            chunksize = max(1, len(batches) // (4 * n_workers))
            yield from pool.map(_run_batch, batches, chunksize=chunksize)
    finally:
        _TASK = None


def _reduced_map(task, batches, n_workers, progress=False):
    """The merged histograms of ``task(start, stop)`` over all batches.

    Each worker merges the histograms of a contiguous group of batches, so the memory in
    use depends on the number of workers rather than on N.
    """
    global _TASK

    n_workers = _worker_count(n_workers, len(batches))
    task = _counted(task, batches, progress)
    _TASK = task
    try:
        pool = _fork_pool(n_workers)
        if pool is None:
            return _histograms_of(task, batches)
        n_groups = min(len(batches), 4 * n_workers)
        groups = [batches[i * len(batches) // n_groups : (i + 1) * len(batches) // n_groups]
                  for i in range(n_groups)]
        with pool:
            return _merge_named(list(pool.map(_run_group, groups)))
    finally:
        _TASK = None


def _run_batch(batch):
    start, stop = batch
    return _TASK(start, stop)


def _run_group(group):
    return _histograms_of(_TASK, group)
