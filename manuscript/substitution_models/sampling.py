"""The Monte Carlo sampler of the substitution models: (4,1)-persistence distributions.

For a homogeneous product measure ``rho^{(x)L}`` the six Hamming distances of a quadruple
depend on each site only through which of the six pairs it mismatches. The sampler
therefore draws, for each quadruple, a multinomial count of the L sites over the at most
``2^6`` mismatch patterns of :func:`single_site_pattern_law`, sums those counts into the
six distances, and evaluates :func:`n_invariants.invariants.vr_persistence` on them. No
sequences are built.

The distance and hyperbolicity distributions need no sampling; their exact laws are in
``models.py``.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from n_invariants.invariants import vr_persistence

__all__ = ["single_site_pattern_law", "sample_distribution"]


def single_site_pattern_law(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Law of one site's six pairwise mismatch indicators, as ``(patterns, probabilities)``.

    A site contributes to the six pairwise distances only through which of the six pairs
    it mismatches. Enumerating the ``k^4`` symbol assignments collapses them onto at most
    ``2^6`` patterns, and the six distances of a quadruple are then a multinomial count
    over those patterns. This lets :func:`sample_distribution` sample quadruples without
    building sequences.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1 or rho.size < 2:
        raise ValueError("rho must be a one-dimensional probability vector.")
    rho = rho / float(np.sum(rho))

    pattern_probs: defaultdict[tuple[int, int, int, int, int, int], float] = defaultdict(float)
    alphabet_size = rho.size
    for x1 in range(alphabet_size):
        for x2 in range(alphabet_size):
            for x3 in range(alphabet_size):
                for x4 in range(alphabet_size):
                    probability = rho[x1] * rho[x2] * rho[x3] * rho[x4]
                    pattern_probs[
                        (
                            int(x1 != x2),
                            int(x1 != x3),
                            int(x1 != x4),
                            int(x2 != x3),
                            int(x2 != x4),
                            int(x3 != x4),
                        )
                    ] += float(probability)

    patterns = np.asarray(list(pattern_probs.keys()), dtype=np.int16)
    probabilities = np.asarray(list(pattern_probs.values()), dtype=float)
    probabilities /= float(np.sum(probabilities))
    return patterns, probabilities


def sample_distribution(
    length: int,
    patterns: np.ndarray,
    probabilities: np.ndarray,
    samples: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Monte Carlo four-point persistence distribution for a homogeneous product measure.

    Draws ``samples`` quadruples in batches of ``batch_size``, each as a multinomial count
    of the ``length`` sites over the six-edge mismatch patterns, sums those counts into
    the six Hamming distances, and histograms the resulting birth-lifetime pairs.

    Returns ``(birth, lifetime, count, n_trivial)``, sorted lexicographically by
    ``(birth, lifetime)``, where ``n_trivial`` counts the quadruples with no class.
    """
    histogram: Counter[tuple[int, int]] = Counter()
    nonzero_total = 0
    rng = np.random.default_rng(seed)

    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        site_pattern_counts = rng.multinomial(length, probabilities, size=size)
        edges = site_pattern_counts @ patterns
        diagrams = vr_persistence(edges)
        nonzero = diagrams[:, 1] > diagrams[:, 0]
        birth = diagrams[nonzero, 0]
        lifetime = diagrams[nonzero, 1] - birth
        nonzero_total += int(birth.size)
        if birth.size:
            pairs = np.column_stack([birth, lifetime]).astype(np.int16, copy=False)
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            for pair, count in zip(unique_pairs, counts, strict=True):
                histogram[(int(pair[0]), int(pair[1]))] += int(count)

    if histogram:
        keys = np.asarray(list(histogram.keys()), dtype=np.int16)
        counts = np.asarray(list(histogram.values()), dtype=np.int64)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys = keys[order]
        counts = counts[order]
    else:
        keys = np.empty((0, 2), dtype=np.int16)
        counts = np.empty(0, dtype=np.int64)

    return keys[:, 0], keys[:, 1], counts, samples - nonzero_total
