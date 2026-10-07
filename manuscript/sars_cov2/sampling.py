"""The sampler of the SARS-CoV-2 pipeline: invariant distributions on a Hamming space.

The point set is a ``uint8`` array of sequences, the metric is the Hamming distance, and
the measure is a weight vector over the points. To estimate the distribution of an n-point
invariant, tuples of points are drawn i.i.d. from the measure, the distances within each
tuple are computed directly from the sequences, and the invariant is histogrammed over
the tuples.

No pairwise distance matrix is formed, so memory use does not grow quadratically with the
number of points. For the roughly 2.7 million SARS-CoV-2 haplotypes of the article, the
condensed ``uint16`` distance matrix would take about 5.7 TiB, while the number of samples
needed for a given accuracy does not depend on the number of points (see
:mod:`n_invariants.bounds`).

The sample count is split into fixed, independently seeded chunks that are merged by
summation, so a run is reproducible from ``(seed, n_samples, batch_size)`` alone and does
not depend on ``n_workers`` or on the order in which workers finish.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

from n_invariants.invariants import condensed_rows_to_square, hyperbolicity_vec, persistence_set_vec

__all__ = [
    "sample_tuples",
    "merge_histogram_dicts",
    "condensed_hamming_rows",
    "random_distribution_k_parallel",
    "hyperbolicity_distribution_random_parallel",
    "persistence_set_random_parallel",
]

# Published to forked workers via copy-on-write; never pickled.
_WORKER_SEQ = None  # uint8[U, L]: the point set
_WORKER_U = None  # number of points (rows of _WORKER_SEQ)


def sample_tuples(N, n_samples, n, p=None, rng=None):
    """Draw ``n_samples`` n-tuples of indices from ``range(N)``, i.i.d. under ``p``.

    Parameters
    ----------
    N : int
        Number of points to draw from.
    n_samples : int
        Number of tuples.
    n : int
        Points per tuple.
    p : ndarray of shape (N,), optional
        Sampling weights, i.e. the measure on the point set. Uniform if None.
    rng : numpy.random.Generator or int, optional
        Generator, or a seed for one.

    Returns
    -------
    ndarray of shape (n_samples, n)
        Point indices. Drawn with replacement, so a tuple may repeat a point; this is
        i.i.d. sampling from the measure, which is what the curvature measure is defined
        by, not sampling of distinct subsets.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    return rng.choice(N, size=(n_samples, n), p=p)


def merge_histogram_dicts(dict_list):
    """Sum a list of ``{value: count}`` histograms into one.

    This is how partial results from independently seeded sample chunks are combined:
    because the chunks are disjoint and the counts additive, merging by summation makes
    the total independent of how the work was split across workers.
    """
    keys = set()
    for D in dict_list:
        keys.update(list(D.keys()))

    merged = {}
    for key in keys:
        merged[key] = 0
        for idx in range(len(dict_list)):
            if key in dict_list[idx]:
                merged[key] += dict_list[idx][key]

    return merged


def condensed_hamming_rows(seq, idx):
    """For each sampled tuple, the Hamming distances among its own members.

    This is a per-tuple operation, not an all-pairs matrix: row i of the output holds only
    the ``binom(n, 2)`` distances between the n points of tuple i (a single distance when
    ``n = 2``).

    Parameters
    ----------
    seq : ndarray of shape (U, L), dtype uint8
        The point set. Constant columns may be dropped beforehand: a column on which all
        points agree adds 0 to every pairwise distance, so all distances, and hence all
        invariants, are unchanged.
    idx : ndarray of shape (size, n), dtype int
        Row indices of the n points in each of ``size`` sampled tuples.

    Returns
    -------
    ndarray of shape (size, binom(n, 2)), dtype float64
        Distances in scipy-condensed order, pairs enumerated as
        ``combinations(range(n), 2)``, i.e. ``(0,1), (0,2), ..., (n-2,n-1)``.
    """
    size, n = idx.shape
    n_pairs = n * (n - 1) // 2
    rows = np.empty((size, n_pairs), dtype=np.float64)
    for col, (a, b) in enumerate(combinations(range(n), 2)):
        # Peak memory is ~3 * size * L bytes per pair.
        rows[:, col] = np.count_nonzero(seq[idx[:, a]] != seq[idx[:, b]], axis=1)
    return rows


def _sampling_worker(chunks, n, square, p, f, f_args, f_kwargs):
    """Process a list of ``(seed, size)`` chunks and return their merged histogram."""
    seq = _WORKER_SEQ
    U = _WORKER_U

    f_list = []
    for seed, size in chunks:
        rng = np.random.default_rng(seed)
        idx = sample_tuples(U, size, n, p=p, rng=rng)  # (size, n)
        rows = condensed_hamming_rows(seq, idx)  # (size, binom(n, 2))
        dm_sample = condensed_rows_to_square(rows, n) if square else rows
        f_list.append(f(dm_sample, *f_args, **f_kwargs))

    return merge_histogram_dicts(f_list) if f_list else {}


def random_distribution_k_parallel(
    f,
    seq,
    U,
    n_samples,
    n,
    square=False,
    batch_size=None,
    p=None,
    seed=None,
    n_workers=None,
    *args,
    **kwargs,
):
    """Sample ``n_samples`` n-tuples and return the histogram of ``f`` over them.

    Parameters
    ----------
    f : callable
        Invariant histogram callback, e.g. :func:`n_invariants.invariants.hyperbolicity_vec`.
        Receives ``(size, n, n)`` blocks when ``square`` is True, else the
        ``(size, binom(n, 2))`` condensed rows.
    seq : ndarray of shape (U, L), dtype uint8
        The point set X.
    U : int
        Number of points (rows of ``seq``).
    n_samples : int
        Total number of tuples to draw. See :func:`~n_invariants.bounds.sample_size` for how
        to choose it.
    n : int
        Tuple size: 2 for the pairwise distance, 4 for hyperbolicity, ``2k+2`` for
        degree-k persistence.
    square : bool, default False
        Pass square blocks to ``f`` instead of condensed rows.
    batch_size : int, optional
        Maximum tuples per chunk; bounds peak memory. One chunk if None.
    p : ndarray of shape (U,), optional
        Sampling weights over the points, i.e. the measure. Uniform if None.
    seed : int or numpy.random.SeedSequence, optional
        Seeds the chunk decomposition. Together with ``n_samples`` and ``batch_size``
        this determines the result exactly.
    n_workers : int, optional
        Worker processes. Defaults to ``os.cpu_count()``. Affects only the speed, not
        the result.

    Returns
    -------
    dict
        ``{value: count}`` over all ``n_samples`` tuples.
    """
    global _WORKER_SEQ, _WORKER_U

    n_samples = int(n_samples)
    if n_workers is None or n_workers < 0:
        n_workers = os.cpu_count() or 1

    # Fixed decomposition into chunks of at most `batch_size` samples.
    chunk = n_samples if batch_size is None else int(batch_size)
    chunk = max(1, min(chunk, n_samples))
    sizes = [chunk] * (n_samples // chunk)
    residue = n_samples - chunk * len(sizes)
    if residue:
        sizes.append(residue)

    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    chunk_specs = list(zip(ss.spawn(len(sizes)), sizes))

    n_workers = int(max(1, min(n_workers, len(chunk_specs))))
    worker_chunks = [chunk_specs[i::n_workers] for i in range(n_workers)]

    if n_workers == 1:
        return _serial_from_chunks(worker_chunks[0], f, seq, U, n, square, p, args, kwargs)

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("fork start method unavailable; running serially.")
        return _serial_from_chunks(chunk_specs, f, seq, U, n, square, p, args, kwargs)

    # Publish seq to the workers via copy-on-write (never pickled).
    _WORKER_SEQ = seq
    _WORKER_U = U
    try:
        results = []
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(_sampling_worker, worker_chunks[i], n, square, p, f, args, kwargs)
                for i in range(n_workers)
            ]
            for future in futures:
                results.append(future.result())
    finally:
        _WORKER_SEQ = None
        _WORKER_U = None

    return merge_histogram_dicts(results)


def _serial_from_chunks(chunks, f, seq, U, n, square, p, f_args, f_kwargs):
    """In-process fallback: run the given chunks without a worker pool."""
    global _WORKER_SEQ, _WORKER_U
    _WORKER_SEQ = seq
    _WORKER_U = U
    try:
        return _sampling_worker(chunks, n, square, p, f, f_args, f_kwargs)
    finally:
        _WORKER_SEQ = None
        _WORKER_U = None


# ---------------------------------------------------------------------------
# Invariant-specific wrappers: fix the callback and the tuple size.
# ---------------------------------------------------------------------------
def hyperbolicity_distribution_random_parallel(
    seq, U, n_samples, batch_size=None, p=None, seed=None, n_workers=None, round=4
):
    """Sampled distribution of the four-point hyperbolicity deficit. Returns ``{deficit: count}``."""
    return random_distribution_k_parallel(
        hyperbolicity_vec,
        seq,
        U,
        n_samples,
        n=4,
        square=False,
        batch_size=batch_size,
        p=p,
        seed=seed,
        n_workers=n_workers,
        round=round,
    )


def persistence_set_random_parallel(
    seq, U, n_samples, k, batch_size=None, p=None, seed=None, n_workers=None
):
    """Sampled degree-k persistence distribution. Returns ``{(birth, death): count}``."""
    return random_distribution_k_parallel(
        persistence_set_vec,
        seq,
        U,
        n_samples,
        n=2 * k + 2,
        square=True,
        batch_size=batch_size,
        p=p,
        seed=seed,
        n_workers=n_workers,
    )
