"""The sample-size bounds, including the sample counts quoted in the article."""

import numpy as np
import pytest

from n_invariants.bounds import expected_wasserstein_bound, fournier_constant, sample_size, wasserstein_bound


def test_manuscript_substitution_model_sample_count():
    """Section 8.3 / Appendix A: the persistence sample count for Figure 6."""
    assert int(sample_size(epsilon=0.1, alpha=0.995, n=4, p=2)) == 117_207_232


@pytest.mark.parametrize(
    ("epsilon", "n", "p", "lip", "expected"),
    [
        (0.001, 2, 1, 1, 13_235_677),      # pairwise distance
        (0.1, 4, 1, 2, 3_376_863_048),     # four-point hyperbolicity (2-Lipschitz)
        (0.1, 4, 1, 1, 53_230_360),        # degree-one four-point persistence
    ],
)
def test_manuscript_sars_sample_counts(epsilon, n, p, lip, expected):
    """Section 8.4: the three sample counts stored in the precomputed SARS npz files.

    All three invariants have range constant 1 (Cor 6.5 with lambda = 1).
    """
    assert int(sample_size(epsilon, 0.95, n, p, lip=lip, lam=1)) == expected


def test_bound_is_independent_of_the_number_of_points():
    """N depends only on the accuracy, the confidence and the invariant.

    There is no argument for the number of points, so this is a statement about the
    signature. It is the property the SARS-CoV-2 analysis relies on.
    """
    from inspect import signature

    assert set(signature(sample_size).parameters) == {
        "epsilon", "alpha", "n", "p", "lip", "lam", "N", "method",
    }


def test_tighter_accuracy_and_confidence_cost_more_samples():
    base = sample_size(0.1, 0.95, 4, 1)
    assert sample_size(0.05, 0.95, 4, 1) > base
    assert sample_size(0.1, 0.99, 4, 1) > base


def test_lipschitz_default_range_is_the_curvature_measure_at_scaled_accuracy():
    """With lam = lip, Cor 6.5 is Cor 6.3 at accuracy epsilon / lip.

    Both solve the same equation up to a factor lip^p, so they agree to solver precision.
    """
    for n, p, lip in [(2, 1, 3.0), (4, 1, 2.0), (4, 2, 0.5)]:
        assert sample_size(0.1, 0.95, n, p, lip=lip) == pytest.approx(
            sample_size(0.1 / lip, 0.95, n, p), rel=1e-6
        )


def test_smaller_range_constant_needs_fewer_samples():
    """lam enters only the concentration term, so lam < lip helps, but only modestly."""
    default = sample_size(0.1, 0.95, 4, 1, lip=2)
    refined = sample_size(0.1, 0.95, 4, 1, lip=2, lam=1)
    assert refined < default < 1.01 * refined


@pytest.mark.parametrize(
    ("n", "p", "epsilon", "lip", "lam"),
    [(2, 1, 0.01, 1, 1), (4, 1, 0.1, 1, 1), (4, 2, 0.1, 1, 1), (4, 1, 0.1, 2, 1),
     (5, 1, 0.3, 1.5, None)],
)
def test_returned_sample_count_achieves_the_target_accuracy(n, p, epsilon, lip, lam):
    n_samples = sample_size(epsilon, 0.95, n, p, lip=lip, lam=lam)
    assert wasserstein_bound(n_samples, 0.95, n, p, lip=lip, lam=lam) <= epsilon * (1 + 1e-9)
    assert wasserstein_bound(n_samples - 1, 0.95, n, p, lip=lip, lam=lam) > epsilon * (1 - 1e-9)


def test_expectation_bound_scales_with_lipschitz_constant_and_diameter():
    """Cor 6.2: replacing mu_n by nu_F rescales the bound of Thm 6.1 by Lip(F)^p."""
    base = expected_wasserstein_bound(10**6, 4, 1)
    assert expected_wasserstein_bound(10**6, 4, 1, D=3, lip=2) == pytest.approx(6 * base)


def test_weed_bach_is_sharper_than_markov():
    kwargs = dict(epsilon=0.1, alpha=0.95, n=4, p=1)
    with pytest.warns(UserWarning):
        markov = sample_size(**kwargs, method="Markov")
    assert sample_size(**kwargs, method="WB") < markov


def test_mcdiarmid_refuses_the_p_below_two_regime_instead_of_guessing():
    """Its deviation term scales as N^(1/p - 1/2), which does not decay for p <= 2.

    The defining equation then has no root, and fsolve would otherwise hand back its last
    iterate as if it were an answer.
    """
    for p in (1, 2):
        with pytest.warns(UserWarning), pytest.raises(ValueError, match="no usable solution"):
            sample_size(0.1, 0.95, 4, p, method="MD")


def test_boundary_case_without_a_sample_count_is_rejected():
    with pytest.raises(ValueError, match="boundary case"):
        fournier_constant(4, 3)          # p == n(n-1)/4 == 3
    assert np.isfinite(fournier_constant(4, 3, N=10**6))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unrecognized method"):
        sample_size(0.1, 0.95, 4, 1, method="nope")


def test_boundary_case_p_equals_half_dimension_uses_the_sample_count():
    """At p == n(n-1)/4 the constant carries a log(sqrt(N)) term, so it grows with N."""
    assert fournier_constant(4, 3, N=10**4) < fournier_constant(4, 3, N=10**8)
    assert np.isfinite(fournier_constant(4, 3, N=10**6))
