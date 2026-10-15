"""The package sampler: the pushforward relation, metrics, measures and reproducibility."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from n_invariants import (
    empirical_curvature_measure,
    empirical_distribution,
    empirical_distributions,
    sample_tuples,
)
from n_invariants.invariants import (
    HYPERBOLICITY_DEFICIT,
    PAIRWISE_DISTANCE,
    VR_PERSISTENCE_4_1,
    hyperbolicity_deficit,
    pairwise_distance,
    vr_persistence,
)

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


# ---------------------------------------------------------------------------
# The sample is fixed by the seed
# ---------------------------------------------------------------------------
def test_the_sample_does_not_depend_on_batch_size_or_workers():
    kwargs = dict(metric="hamming", seed=7)
    K = empirical_curvature_measure(SEQUENCES, 4, 5000, batch_size=5000, **kwargs)
    for batch_size, n_workers in [(7, 1), (1000, 3), (4999, 2), (None, 0)]:
        assert np.array_equal(K, empirical_curvature_measure(
            SEQUENCES, 4, 5000, batch_size=batch_size, n_workers=n_workers, **kwargs))


def test_a_smaller_sample_is_the_beginning_of_a_larger_one():
    K = empirical_curvature_measure(POINTS, 3, 4000, metric="euclidean", seed=8)
    assert np.array_equal(empirical_curvature_measure(POINTS, 3, 1500, metric="euclidean", seed=8),
                          K[:1500])


def test_sample_tuples_yields_the_tuples_of_the_curvature_measure_in_batches():
    kwargs = dict(metric="hamming", seed=9, batch_size=1000)
    batches = list(sample_tuples(SEQUENCES, 4, 2500, **kwargs))
    assert [len(idx) for idx, _ in batches] == [1000, 1000, 500]

    idx = np.concatenate([idx for idx, _ in batches])
    rows = np.concatenate([rows for _, rows in batches])
    assert idx.shape == (2500, 4)
    assert np.array_equal(rows, empirical_curvature_measure(SEQUENCES, 4, 2500, **kwargs))
    # Column 0 is the distance between the first two points of each tuple.
    assert np.array_equal(rows[:, 0], (SEQUENCES[idx[:, 0]] != SEQUENCES[idx[:, 1]]).sum(axis=1))


def test_sample_tuples_checks_its_input_before_iterating():
    with pytest.raises(ValueError):
        sample_tuples(POINTS, 4, 10, metric="taxicab")


def test_points_of_weight_zero_are_never_drawn():
    weights = np.zeros(len(POINTS))
    weights[[3, 17, 42]] = [1.0, 2.0, 0.5]
    idx = np.concatenate([idx for idx, _ in sample_tuples(POINTS, 4, 3000, metric="euclidean",
                                                          weights=weights, seed=1)])
    assert set(np.unique(idx)) == {3, 17, 42}


# ---------------------------------------------------------------------------
# Several n-invariants from one sample
# ---------------------------------------------------------------------------
def test_several_n_invariants_from_one_sample_match_one_at_a_time():
    kwargs = dict(metric="hamming", seed=10, batch_size=700)
    together = empirical_distributions(SEQUENCES, [HYPERBOLICITY_DEFICIT, VR_PERSISTENCE_4_1],
                                       4, 5000, n_workers=2, **kwargs)
    assert list(together) == ["hyperbolicity_deficit", "vr_persistence_4_1"]
    for F in (HYPERBOLICITY_DEFICIT, VR_PERSISTENCE_4_1):
        values, counts = empirical_distribution(SEQUENCES, F, 4, 5000, **kwargs)
        assert np.array_equal(together[F.name][0], values)
        assert np.array_equal(together[F.name][1], counts)


def test_a_pair_n_invariant_reads_the_first_pair_of_each_quadruple():
    """The restriction to the first two points maps mu_4 to mu_2 (Prop 4.4)."""
    kwargs = dict(metric="hamming", seed=12)
    K = empirical_curvature_measure(SEQUENCES, 4, 3000, **kwargs)
    values, counts = empirical_distributions(SEQUENCES, [PAIRWISE_DISTANCE], 4, 3000,
                                             **kwargs)["pairwise_distance"]
    expected_values, expected_counts = np.unique(K[:, 0], return_counts=True)
    assert np.array_equal(values, expected_values)
    assert np.array_equal(counts, expected_counts)


def test_a_mapping_names_the_n_invariants_and_plain_functions_see_full_rows():
    result = empirical_distributions(POINTS, {"deficit": hyperbolicity_deficit,
                                              "first pair": PAIRWISE_DISTANCE},
                                     4, 1000, metric="euclidean", seed=13, decimals=6)
    assert set(result) == {"deficit", "first pair"}
    assert all(counts.sum() == 1000 for _, counts in result.values())


def test_n_invariants_are_checked_against_the_sample():
    kwargs = dict(metric="euclidean", seed=0)
    with pytest.raises(ValueError, match="needs tuples of 4 points"):
        empirical_distributions(POINTS, [HYPERBOLICITY_DEFICIT], 3, 10, **kwargs)
    with pytest.raises(ValueError, match="distinct names"):
        empirical_distributions(POINTS, [PAIRWISE_DISTANCE, PAIRWISE_DISTANCE], 4, 10, **kwargs)
    with pytest.raises(TypeError, match="mapping"):
        empirical_distributions(POINTS, HYPERBOLICITY_DEFICIT, 4, 10, **kwargs)


# ---------------------------------------------------------------------------
# The full empirical curvature measure
# ---------------------------------------------------------------------------
def _reported_percentages(stderr):
    return [int(line.split("%")[0]) for line in stderr.splitlines() if "% of batches" in line]


@pytest.mark.parametrize("n_workers", [1, 3])
def test_progress_is_reported_once_per_five_percent_of_all_batches(capfd, n_workers):
    """The workers share one counter, so every 5% appears once and in order."""
    empirical_distributions(SEQUENCES, [HYPERBOLICITY_DEFICIT], 4, 40 * 50, metric="hamming",
                            seed=0, batch_size=50, n_workers=n_workers)
    assert _reported_percentages(capfd.readouterr().err) == list(range(5, 101, 5))

    empirical_curvature_measure(SEQUENCES, 4, 3 * 50, metric="hamming", seed=0, batch_size=50,
                                n_workers=n_workers)
    assert _reported_percentages(capfd.readouterr().err) == [30, 65, 100]


def test_progress_can_be_turned_off(capfd):
    empirical_distribution(SEQUENCES, HYPERBOLICITY_DEFICIT, 4, 500, metric="hamming", seed=0,
                           batch_size=50, n_workers=2, progress=False)
    empirical_curvature_measure(SEQUENCES, 4, 500, metric="hamming", seed=0, progress=False)
    captured = capfd.readouterr()
    assert captured.out == captured.err == ""


def test_the_curvature_measure_stays_below_max_bytes_unless_written_to_a_file(tmp_path):
    kwargs = dict(metric="euclidean", seed=14)
    with pytest.raises(ValueError, match="max_bytes"):
        empirical_curvature_measure(POINTS, 4, 1000, max_bytes=1000, **kwargs)

    path = tmp_path / "K.npy"
    K = empirical_curvature_measure(POINTS, 4, 1000, max_bytes=1000, out=path, **kwargs)
    assert np.array_equal(np.load(path), empirical_curvature_measure(POINTS, 4, 1000, **kwargs))
    assert np.array_equal(K, np.load(path))
