"""The package sampler: the pushforward relation, metrics, measures and reproducibility."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from n_invariants import empirical_curvature_measure, empirical_distribution
from n_invariants.invariants import hyperbolicity_deficit, pairwise_distance, vr_persistence

RNG = np.random.default_rng(0)
POINTS = RNG.normal(size=(60, 3))
SEQUENCES = RNG.integers(0, 4, size=(60, 30), dtype=np.uint8)


def test_curvature_measure_has_one_row_of_distances_per_tuple():
    K = empirical_curvature_measure(POINTS, 4, 1234, metric="euclidean", seed=0, batch_size=500)
    assert K.shape == (1234, 6)
    assert np.all(K >= 0)


@pytest.mark.parametrize("invariant, n", [
    (pairwise_distance, 2), (hyperbolicity_deficit, 4), (vr_persistence, 4),
])
def test_distribution_is_the_pushforward_of_the_curvature_measure(invariant, n):
    """With the same seed and batches, nu_{F,N} is exactly F applied to mu_{N,n}."""
    kwargs = dict(metric="hamming", seed=5, batch_size=700)
    K = empirical_curvature_measure(SEQUENCES, n, 5000, **kwargs)
    values, counts = empirical_distribution(SEQUENCES, invariant, n, 5000, **kwargs)

    F = invariant(K)
    expected_values, expected_counts = np.unique(F, axis=0 if F.ndim == 2 else None,
                                                 return_counts=True)
    assert np.array_equal(values, expected_values)
    assert np.array_equal(counts, expected_counts)
    assert counts.sum() == 5000


def test_result_is_independent_of_the_number_of_workers():
    kwargs = dict(metric="euclidean", seed=11, batch_size=1000)
    serial = empirical_distribution(POINTS, hyperbolicity_deficit, 4, 9000, decimals=3, **kwargs)
    parallel = empirical_distribution(POINTS, hyperbolicity_deficit, 4, 9000, decimals=3,
                                      n_workers=3, **kwargs)
    assert all(np.array_equal(a, b) for a, b in zip(serial, parallel))

    K_serial = empirical_curvature_measure(POINTS, 4, 9000, **kwargs)
    K_parallel = empirical_curvature_measure(POINTS, 4, 9000, n_workers=3, **kwargs)
    assert np.array_equal(K_serial, K_parallel)


def test_metrics_agree_with_the_precomputed_distance_matrix():
    kwargs = dict(seed=2, batch_size=400)
    hamming_matrix = (SEQUENCES[:, None, :] != SEQUENCES[None, :, :]).sum(axis=2)
    assert np.array_equal(
        empirical_curvature_measure(SEQUENCES, 4, 1000, metric="hamming", **kwargs),
        empirical_curvature_measure(hamming_matrix, 4, 1000, metric="precomputed", **kwargs),
    )
    assert np.allclose(
        empirical_curvature_measure(POINTS, 4, 1000, metric="euclidean", **kwargs),
        empirical_curvature_measure(cdist(POINTS, POINTS), 4, 1000, metric="precomputed", **kwargs),
    )


def test_a_metric_function_is_applied_to_corresponding_rows():
    def manhattan(A, B):
        return np.abs(A - B).sum(axis=1)

    K = empirical_curvature_measure(POINTS, 2, 300, metric=manhattan, seed=3)
    matrix = cdist(POINTS, POINTS, metric="cityblock")
    assert np.allclose(K, empirical_curvature_measure(matrix, 2, 300, metric="precomputed", seed=3))


def test_a_concentrated_measure_is_respected():
    """All the mass on points 0 and 1: only their distance and 0 can occur."""
    weights = np.zeros(len(POINTS))
    weights[[0, 1]] = 1.0
    values, counts = empirical_distribution(POINTS, pairwise_distance, 2, 2000,
                                            metric="euclidean", weights=weights, seed=4)
    assert set(values) <= {0.0, np.linalg.norm(POINTS[0] - POINTS[1])}
    assert counts.sum() == 2000


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        empirical_curvature_measure(POINTS, 4, 10, metric="taxicab")
    with pytest.raises(ValueError):
        empirical_curvature_measure(POINTS, 4, 10, metric="euclidean", weights=np.ones(3))
    with pytest.raises(ValueError):
        empirical_curvature_measure(POINTS, 1, 10, metric="euclidean")
    with pytest.raises(TypeError):
        empirical_curvature_measure(POINTS, 4, 10)  # the metric must be chosen
