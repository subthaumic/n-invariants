"""The JC and HB substitution models and the exact laws derived from them."""

import itertools

import numpy as np
import pytest
from manuscript_modules import load

from models import (
    binomial_pmf_with_coefficients,
    build_delta_tick_grid,
    build_hb_generator,
    distance_probability_from_site_distribution,
    hb_factor,
    hb_single_site_distribution,
    hb_stationary_distribution,
    hyperbolicity_pmf_from_rho,
    jc_distance_probability,
    jc_single_site_distribution,
    log_binomial_coefficients,
    single_site_pair_sum_law,
)

single_site_pattern_law = load("substitution_models/sampling.py").single_site_pattern_law

MANUSCRIPT_PHI = np.array([0.0, 0.0, -10.0, -10.0])


def test_jc_starts_at_a_point_mass_and_relaxes_to_uniform():
    assert np.allclose(jc_single_site_distribution(0.0, 4, 1.0), [1, 0, 0, 0])
    assert np.allclose(jc_single_site_distribution(1e4, 4, 1.0), 0.25)


def test_jc_law_is_a_probability_vector():
    for t in (0.05, 0.1, 0.5, 1.0):
        rho = jc_single_site_distribution(t, 4, 1.0)
        assert np.isclose(rho.sum(), 1.0)
        assert np.all(rho >= 0)


def test_hb_generator_rows_sum_to_zero():
    q = build_hb_generator(4, 1.0, MANUSCRIPT_PHI)
    assert np.allclose(q.sum(axis=1), 0.0)
    assert np.all(np.diagonal(q) <= 0)


def test_hb_factor_limits_to_one_at_equal_fitness():
    assert hb_factor(0.0) == 1.0
    assert np.isclose(hb_factor(1e-12), 1.0)


def test_hb_relaxes_to_its_stationary_law():
    late = hb_single_site_distribution(500.0, 4, 1.0, MANUSCRIPT_PHI)
    assert np.allclose(late, hb_stationary_distribution(MANUSCRIPT_PHI), atol=1e-8)


def test_hb_stationary_law_is_proportional_to_exp_phi():
    stationary = hb_stationary_distribution(MANUSCRIPT_PHI)
    expected = np.exp(MANUSCRIPT_PHI) / np.exp(MANUSCRIPT_PHI).sum()
    assert np.allclose(stationary, expected)
    # phi = (0, 0, -10, -10) strongly suppresses the last two symbols.
    assert stationary[0] > 100 * stationary[2]


def test_hb_with_a_flat_fitness_profile_reduces_to_jc():
    flat = np.zeros(4)
    for t in (0.05, 0.5, 2.0):
        assert np.allclose(
            hb_single_site_distribution(t, 4, 1.0, flat),
            jc_single_site_distribution(t, 4, 1.0),
            atol=1e-10,
        )


def test_jc_distance_probability_matches_the_site_law_route():
    """1 - c_2(rho_t) computed from the law must equal the closed-form JC expression."""
    for t in (0.05, 0.1, 0.5, 1.0, 3.0):
        rho = jc_single_site_distribution(t, 4, 1.0)
        assert np.isclose(
            distance_probability_from_site_distribution(rho),
            jc_distance_probability(t, 4, 1.0),
        )


@pytest.mark.parametrize("length", [1, 5, 100])
def test_binomial_pmf_is_a_normalized_distribution(length):
    log_binom = log_binomial_coefficients(length)
    pmf = binomial_pmf_with_coefficients(length, 0.3, log_binom)
    assert pmf.shape == (length + 1,)
    assert np.isclose(pmf.sum(), 1.0)
    assert np.isclose((pmf * np.arange(length + 1)).sum(), length * 0.3)


def test_binomial_pmf_handles_the_degenerate_probabilities():
    log_binom = log_binomial_coefficients(10)
    assert binomial_pmf_with_coefficients(10, 0.0, log_binom)[0] == 1.0
    assert binomial_pmf_with_coefficients(10, 1.0, log_binom)[-1] == 1.0


def test_single_site_pair_sum_law_is_normalized_and_exchangeable():
    """Under a product measure the three opposite-side pairings are exchangeable.

    Relabelling the four points permutes the three pair sums, so the joint law must be
    invariant under every permutation of its axes.
    """
    law = single_site_pair_sum_law(jc_single_site_distribution(0.5, 4, 1.0))
    assert law.shape == (3, 3, 3)
    assert np.isclose(law.sum(), 1.0)
    assert np.all(law >= 0)
    for permutation in itertools.permutations(range(3)):
        assert np.allclose(law, law.transpose(permutation))


def test_delta_tick_grid_is_max_minus_median():
    ticks = build_delta_tick_grid(3)
    for p1, p2, p3 in [(0, 0, 0), (4, 2, 1), (6, 6, 2), (1, 5, 3)]:
        ordered = sorted((p1, p2, p3))
        assert ticks[p1, p2, p3] == ordered[2] - ordered[1]


def test_hyperbolicity_pmf_is_normalized_and_supported_on_even_ticks_for_binary():
    """A binary alphabet moves the pair sums in steps of two, so odd ticks are empty.

    This is the observation in the article that half-integer deficits vanish for k = 2.
    """
    length = 8
    ticks = build_delta_tick_grid(length)
    rho = np.array([0.5, 0.5])
    pmf = hyperbolicity_pmf_from_rho(length, rho, ticks)
    assert np.isclose(pmf.sum(), 1.0)
    assert np.allclose(pmf[1::2], 0.0, atol=1e-12)


def test_hyperbolicity_pmf_puts_half_integer_mass_on_a_four_letter_alphabet():
    length = 6
    ticks = build_delta_tick_grid(length)
    rho = np.full(4, 0.25)
    pmf = hyperbolicity_pmf_from_rho(length, rho, ticks)
    assert np.isclose(pmf.sum(), 1.0)
    assert pmf[1::2].sum() > 0.0


def test_single_site_pattern_law_is_a_normalized_law_over_mismatch_patterns():
    patterns, probabilities = single_site_pattern_law(np.full(4, 0.25))
    assert patterns.shape[1] == 6
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0)
    assert np.all((patterns == 0) | (patterns == 1))
    # The all-match pattern occurs with probability sum(rho^3 * ...) = 1/k^3 for uniform rho.
    all_match = np.flatnonzero((patterns == 0).all(axis=1))
    assert np.isclose(probabilities[all_match].sum(), 1 / 4**3)
