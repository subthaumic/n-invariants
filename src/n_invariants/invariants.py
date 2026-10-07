"""Metric n-point invariants of the article, evaluated on batches of sampled tuples.

An *n-point invariant* is a function of the ``n(n-1)/2`` pairwise distances among n points
of a metric space. Each function below takes a batch of tuples as condensed distance rows,
an array of shape ``(size, n(n-1)/2)`` with the pairs in scipy's condensed order
``(0,1), (0,2), ..., (n-2,n-1)``, and returns one value per tuple:

==============================  ========  ===============================================
Function                        n         Value per tuple
==============================  ========  ===============================================
:func:`pairwise_distance`       2         the distance
:func:`hyperbolicity_deficit`   4         the four-point hyperbolicity deficit
:func:`vr_persistence`          2k + 2    degree-k Vietoris-Rips persistence, as
                                          ``(birth, death)``
==============================  ========  ===============================================

These are the invariants that :func:`n_invariants.empirical_distribution` takes. Their
Lipschitz and range constants are listed in :mod:`n_invariants.bounds`.

:func:`hyperbolicity_vec` and :func:`persistence_set_vec` are histogram forms of the two
four-point invariants that return ``{value: count}`` dictionaries. The samplers in
``manuscript/`` use them.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

__all__ = [
    "pairwise_distance",
    "hyperbolicity_deficit",
    "vr_persistence",
    "hyperbolicity_vec",
    "persistence_set_vec",
    "condensed_rows_to_square",
]


def _points_per_tuple(n_pairs: int) -> int:
    """The n with ``n(n-1)/2 = n_pairs``."""
    n = int(round((1 + np.sqrt(1 + 8 * n_pairs)) / 2))
    if n * (n - 1) // 2 != n_pairs:
        raise ValueError(f"{n_pairs} columns are not the pairs of any number of points.")
    return n


def pairwise_distance(rows):
    """The distance of each sampled pair.

    Parameters
    ----------
    rows : ndarray of shape (size, 1)
        Condensed distance rows of sampled pairs (n = 2).

    Returns
    -------
    ndarray of shape (size,)
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 1:
        raise ValueError("pairwise_distance expects rows of shape (size, 1), i.e. n = 2.")
    return rows[:, 0]


def hyperbolicity_deficit(rows):
    """The four-point hyperbolicity deficit of each sampled quadruple.

    For four points the three pairings of the six distances into opposite-side pairs give
    three sums ``S_1, S_2, S_3``. The deficit is
    ``max_i (S_i - max_{j != i} S_j) / 2``, floored at zero, i.e. half the gap between the
    largest and second-largest of the three sums. It vanishes exactly when the quadruple
    satisfies the four-point condition, i.e. embeds in a tree.

    Parameters
    ----------
    rows : ndarray of shape (size, 6)
        Condensed distance rows of sampled quadruples, with the pairs
        ``(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)``.

    Returns
    -------
    ndarray of shape (size,)
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError("hyperbolicity_deficit expects rows of shape (size, 6), i.e. n = 4.")
    d_12 = rows[:, 0]
    d_13 = rows[:, 1]
    d_14 = rows[:, 2]
    d_23 = rows[:, 3]
    d_24 = rows[:, 4]
    d_34 = rows[:, 5]

    # The three ways of pairing the four points into two opposite sides.
    S_1 = d_12 + d_34
    S_2 = d_14 + d_23
    S_3 = d_13 + d_24

    # Each delta_i is the margin by which S_i exceeds the larger of the other two.
    delta_1 = S_1 - np.maximum(S_2, S_3)
    delta_2 = S_2 - np.maximum(S_1, S_3)
    delta_3 = S_3 - np.maximum(S_1, S_2)

    delta_max = np.max(np.array([delta_1, delta_2, delta_3]), axis=0)
    delta_max = np.maximum(delta_max, 0)  # only the largest sum can have a positive margin
    return delta_max / 2


def vr_persistence(rows):
    """The degree-k Vietoris-Rips persistence of each sampled ``(2k+2)``-tuple.

    On ``n = 2k + 2`` points the degree-k Vietoris-Rips persistence diagram has at most
    one off-diagonal point, whose birth and death are read directly off the distance
    matrix: the class is born once every point has all but one of the others within the
    scale, and dies once some point has all others within the scale. Hence

    * ``birth = max_i  (second largest distance from point i)``,
    * ``death = min_i  (largest distance from point i)``,

    and the diagram is empty when ``birth >= death``, which is recorded as ``(0, 0)``.
    At ``k = 1`` this is the (4,1)-persistence of the article.

    Parameters
    ----------
    rows : ndarray of shape (size, n(n-1)/2)
        Condensed distance rows of sampled n-tuples, with n even.

    Returns
    -------
    ndarray of shape (size, 2)
        ``(birth, death)`` per tuple, ``(0, 0)`` for an empty diagram.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2:
        raise ValueError("vr_persistence expects rows of shape (size, n(n-1)/2).")
    n = _points_per_tuple(rows.shape[1])
    if n % 2:
        raise ValueError(f"vr_persistence needs an even number of points, n = 2k + 2; got n = {n}.")
    return _persistence_pairs(condensed_rows_to_square(rows, n))


def _persistence_pairs(dms):
    """``(birth, death)`` per square distance block, ``(0, 0)`` for an empty diagram."""
    # Per row, the two largest entries, sorted ascending along the last axis.
    max_2 = np.partition(dms, -2, axis=2)[:, :, -2:]
    max_2 = np.sort(max_2, axis=2)

    tb = np.max(max_2[:, :, 0], axis=1)  # birth: largest second-largest
    td = np.min(max_2[:, :, 1], axis=1)  # death: smallest largest
    pds = np.vstack((tb, td)).T

    # An empty diagram, recorded at the origin rather than as a zero-length interval.
    pds[np.where(tb >= td)[0], :] = 0
    return pds


def hyperbolicity_vec(dm_row_sq, round=4):
    """Histogram of :func:`hyperbolicity_deficit` over many sampled quadruples.

    Parameters
    ----------
    dm_row_sq : ndarray of shape (n_samples, 6)
        Condensed distance rows of sampled quadruples.
    round : int, default 4
        Decimal places to round the deficit to before histogramming. Guards against
        distinct floating-point representations of the same value fragmenting the
        histogram; on integer Hamming distances the deficit is a half-integer anyway.

    Returns
    -------
    dict
        ``{deficit: count}``.
    """
    delta_max = np.round(hyperbolicity_deficit(dm_row_sq), decimals=round)

    values, counts = np.unique(delta_max, return_counts=True)
    return {values[idx]: counts[idx] for idx in range(values.shape[0])}


def persistence_set_vec(dms):
    """Histogram of :func:`vr_persistence` over many sampled ``(2k+2)``-point sets.

    Parameters
    ----------
    dms : ndarray of shape (n_samples, n, n)
        Symmetric distance matrices, one per sampled point set, with ``n = 2k + 2``.

    Returns
    -------
    dict
        ``{(birth, death): count}``, with ``(0, 0)`` counting the empty diagrams.
    """
    pds = _persistence_pairs(dms)

    values, counts = np.unique(pds, axis=0, return_counts=True)
    return {tuple(values[i, :]): counts[i] for i in range(values.shape[0])}


def condensed_rows_to_square(rows, n):
    """Condensed pair-distance rows to symmetric square blocks.

    ``(size, binom(n, 2)) -> (size, n, n)``. Pairs are enumerated as
    ``combinations(range(n), 2)``, matching scipy's condensed ordering.
    """
    size = rows.shape[0]
    blocks = np.zeros((size, n, n), dtype=rows.dtype)
    for idx, (a, b) in enumerate(combinations(range(n), 2)):
        blocks[:, a, b] = rows[:, idx]
        blocks[:, b, a] = rows[:, idx]
    return blocks
