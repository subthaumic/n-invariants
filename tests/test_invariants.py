"""The four-point invariants, checked against quadruples whose value is known by hand."""

import numpy as np
import pytest

from n_invariants.invariants import (
    HYPERBOLICITY_DEFICIT,
    PAIRWISE_DISTANCE,
    VR_PERSISTENCE_4_1,
    Invariant,
    condensed_rows_to_square,
    hyperbolicity_deficit,
    hyperbolicity_vec,
    pairwise_distance,
    persistence_set_vec,
    vr_persistence,
)

# Condensed order for four points: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).


def test_hyperbolicity_vanishes_on_a_four_point_tree():
    """Two pairs at distance 2, split by a bridge of length 10: a tree, so deficit 0.

    Sums of opposite sides: 2+2 = 4, 12+12 = 24, 12+12 = 24. The largest ties with the
    median, so the gap, and hence the deficit, is zero.
    """
    rows = np.array([[2.0, 12.0, 12.0, 12.0, 12.0, 2.0]])
    assert hyperbolicity_vec(rows) == {0.0: 1}


def test_hyperbolicity_of_a_unit_square():
    """Four corners of a square with side 1 and diagonal 2.

    Sums: 1+1 = 2 (two pairings) and 2+2 = 4. Gap 4 - 2 = 2, so the deficit is 1.
    """
    rows = np.array([[1.0, 2.0, 1.0, 1.0, 2.0, 1.0]])
    assert hyperbolicity_vec(rows) == {1.0: 1}


def test_hyperbolicity_is_never_negative():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 50, size=(5000, 6)).astype(float)
    assert min(hyperbolicity_vec(rows)) >= 0.0


def test_hyperbolicity_counts_are_conserved():
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 30, size=(4321, 6)).astype(float)
    assert sum(hyperbolicity_vec(rows).values()) == 4321


def test_persistence_of_a_hamming_rectangle():
    """A homoplasy rectangle with sides a=3, b=5 and diagonals a+b=8.

    Its degree-one class is born at max(a, b) = 5 and dies at a + b = 8.
    """
    a, b = 3.0, 5.0
    rows = np.array([[a, a + b, b, b, a + b, a]])
    blocks = condensed_rows_to_square(rows, 4)
    assert persistence_set_vec(blocks) == {(max(a, b), a + b): 1}


def test_persistence_is_trivial_on_a_degenerate_quadruple():
    """All distances equal: the class is born and dies at the same scale, so it is empty."""
    rows = np.full((1, 6), 7.0)
    blocks = condensed_rows_to_square(rows, 4)
    assert persistence_set_vec(blocks) == {(0.0, 0.0): 1}


def test_persistence_lifetime_is_the_short_side_of_the_rectangle():
    """For a rectangle with sides a <= b, birth = b and death = a + b, so lifetime = a."""
    for a, b in [(1.0, 1.0), (1.0, 9.0), (4.0, 6.0)]:
        rows = np.array([[a, a + b, b, b, a + b, a]])
        (birth, death), = persistence_set_vec(condensed_rows_to_square(rows, 4))
        assert death - birth == min(a, b)


def test_condensed_rows_to_square_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(2)
    rows = rng.integers(1, 20, size=(50, 6)).astype(float)
    blocks = condensed_rows_to_square(rows, 4)
    assert blocks.shape == (50, 4, 4)
    assert np.array_equal(blocks, blocks.transpose(0, 2, 1))
    assert np.all(np.diagonal(blocks, axis1=1, axis2=2) == 0)
    # Condensed order must round-trip: entry k is the k-th pair of combinations(range(4), 2).
    assert np.array_equal(blocks[:, 0, 1], rows[:, 0])
    assert np.array_equal(blocks[:, 2, 3], rows[:, 5])


def test_histogram_forms_count_the_values_of_the_array_invariants():
    """hyperbolicity_vec and persistence_set_vec are histograms of the per-tuple values."""
    rng = np.random.default_rng(3)
    rows = rng.integers(0, 12, size=(2000, 6)).astype(float)

    values, counts = np.unique(hyperbolicity_deficit(rows), return_counts=True)
    assert hyperbolicity_vec(rows) == dict(zip(values, counts))

    pairs, counts = np.unique(vr_persistence(rows), axis=0, return_counts=True)
    expected = {tuple(pair): count for pair, count in zip(pairs, counts)}
    assert persistence_set_vec(condensed_rows_to_square(rows, 4)) == expected


def test_vr_persistence_of_a_hamming_rectangle():
    a, b = 3.0, 5.0
    rows = np.array([[a, a + b, b, b, a + b, a], [7.0] * 6])
    assert vr_persistence(rows).tolist() == [[b, a + b], [0.0, 0.0]]


def test_invariants_reject_rows_of_the_wrong_width():
    rows = np.ones((3, 6))
    assert np.array_equal(pairwise_distance(rows[:, :1]), np.ones(3))
    with pytest.raises(ValueError):
        pairwise_distance(rows)
    with pytest.raises(ValueError):
        hyperbolicity_deficit(rows[:, :3])
    with pytest.raises(ValueError):
        vr_persistence(np.ones((3, 3)))  # three points: no degree-k class on an odd number
    with pytest.raises(ValueError):
        vr_persistence(np.ones((3, 5)))  # five columns are not the pairs of any n


def test_invariant_defaults_to_the_lipschitz_constant_as_range_constant():
    F = Invariant(hyperbolicity_deficit, n=4, lip=2)
    assert (F.lam, F.name, F.real_valued) == (2, "hyperbolicity_deficit", False)


def test_invariant_rejects_invalid_constants():
    with pytest.raises(ValueError):
        Invariant(pairwise_distance, n=1, lip=1)
    with pytest.raises(ValueError):
        Invariant(pairwise_distance, n=2, lip=0)
    with pytest.raises(ValueError):
        Invariant(pairwise_distance, n=2, lip=1, lam=0)


def test_the_article_constants():
    """Prop 4.9 and Section 8.4."""
    constants = [(F.n, F.lip, F.lam, F.real_valued)
                 for F in (PAIRWISE_DISTANCE, HYPERBOLICITY_DEFICIT, VR_PERSISTENCE_4_1)]
    assert constants == [(2, 1, 1, True), (4, 2, 1, True), (4, 1, 1, False)]


def test_invariant_on_larger_tuples_reads_their_first_points():
    """On five points, the pairs among the first four are columns 0, 1, 2, 4, 5, 7."""
    rows = np.random.default_rng(4).integers(0, 9, size=(300, 10)).astype(float)
    assert np.array_equal(HYPERBOLICITY_DEFICIT(rows), hyperbolicity_deficit(rows[:, [0, 1, 2, 4, 5, 7]]))
    assert np.array_equal(PAIRWISE_DISTANCE(rows), rows[:, 0])
    assert np.array_equal(VR_PERSISTENCE_4_1(rows[:, [0, 1, 2, 4, 5, 7]]),
                          vr_persistence(rows[:, [0, 1, 2, 4, 5, 7]]))


def test_invariant_rejects_tuples_with_too_few_points():
    with pytest.raises(ValueError, match="at least 4 points"):
        HYPERBOLICITY_DEFICIT(np.ones((3, 3)))
