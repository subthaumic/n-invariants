"""The sample-size bounds, including the sample counts quoted in the article."""

import numpy as np
import pytest

from n_invariants.bounds import (
    _real_valued_constant,
    fournier_constant,
    guaranteed_accuracy,
    sample_size,
    sample_size_real_valued,
    wasserstein_bound,
    wasserstein_bound_real_valued,
    wasserstein_moment_bound,
    wasserstein_moment_bound_real_valued,
)
from n_invariants.invariants import (
    HYPERBOLICITY_DEFICIT,
    PAIRWISE_DISTANCE,
    VR_PERSISTENCE_4_1,
    Invariant,
)


def test_manuscript_substitution_model_sample_count():
    """Section 8.3 / Appendix A: the persistence sample count for Figure 6."""
    assert int(sample_size(epsilon=0.1, alpha=0.995, n=4, p=2)) == 117_207_232


def test_manuscript_sars_sample_count_and_accuracies():
    """Section 8.4: one sample of quadruples per cutoff, sized for mu_4 (Cor 6.3).

    Each bound holds at the confidence that a union bound over three accuracies at each of
    five cutoffs turns into 0.95 overall. On this sample the (4,1)-persistence is accurate
    to 0.1 through mu_4, and the real-valued distance and hyperbolicity deficit to 5.6e-4
    (Prop 6.6, which with range constant 1 is the bound for n = 2).
    """
    alpha = 1 - 0.05 / 15
    n_samples = int(sample_size(0.1, alpha, 4, 1))
    assert n_samples == 53_433_249
    assert wasserstein_bound(n_samples, alpha, 4, 1) <= 0.1
    assert wasserstein_bound(n_samples, alpha, 2, 1) == pytest.approx(5.6130e-4, rel=1e-4)


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
    base = wasserstein_moment_bound(10**6, 4, 1)
    assert wasserstein_moment_bound(10**6, 4, 1, D=3, lip=2) == pytest.approx(6 * base)


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


# ---------------------------------------------------------------------------
# Prop 6.6: real-valued n-invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1, 1.5, 2, 3])
def test_real_valued_constant_is_the_one_dimensional_constant_of_thm_6_1(p):
    """C(1,p) of Prop 6.6 is C(d,p) of Thm 6.1 at d = 1, i.e. for pairs."""
    assert _real_valued_constant(p) == pytest.approx(fournier_constant(2, p))


@pytest.mark.parametrize(("p", "lam"), [(1, 1), (2, 0.5), (3, 2)])
def test_real_valued_bounds_do_not_depend_on_n(p, lam):
    """Prop 6.6 is the argument of Thm 6.1 on the real line: the numbers of pairs with lip = lam."""
    assert wasserstein_bound_real_valued(10**6, 0.99, p, lam=lam) == pytest.approx(
        wasserstein_bound(10**6, 0.99, 2, p, lip=lam, lam=lam))
    assert wasserstein_moment_bound_real_valued(10**6, p, D=3, lam=lam) == pytest.approx(
        wasserstein_moment_bound(10**6, 2, p, D=3, lip=lam))


@pytest.mark.parametrize(("epsilon", "p", "lam"), [(0.1, 1, 1), (0.01, 2, 1), (0.05, 1, 0.5)])
def test_real_valued_sample_size_achieves_the_target_accuracy(epsilon, p, lam):
    n_samples = sample_size_real_valued(epsilon, 0.95, p, lam=lam)
    assert wasserstein_bound_real_valued(n_samples, 0.95, p, lam=lam) <= epsilon * (1 + 1e-12)
    assert wasserstein_bound_real_valued(n_samples - 1, 0.95, p, lam=lam) > epsilon


def test_real_valued_rate_needs_far_fewer_samples_for_the_hyperbolicity_deficit():
    """At accuracy 0.1: 1,324 quadruples by Prop 6.6, against 3.4e9 through mu_4 (Cor 6.5)."""
    assert sample_size_real_valued(0.1, 0.95, 1) == 1324
    assert sample_size(0.1, 0.95, 4, 1, lip=2, lam=1) > 10**9


def test_real_valued_bound_needs_p_at_least_one():
    with pytest.raises(ValueError, match="p >= 1"):
        sample_size_real_valued(0.1, 0.95, 0.5)


# ---------------------------------------------------------------------------
# guaranteed_accuracy
# ---------------------------------------------------------------------------
def test_guaranteed_accuracy_reproduces_the_sars_accuracies():
    """Section 8.4: mu_4 bounds the persistence, Prop 6.6 the two real-valued n-invariants."""
    alpha = 1 - 0.05 / 15
    result = guaranteed_accuracy(53_433_249, {"distance": PAIRWISE_DISTANCE,
                                  "hyperbolicity": HYPERBOLICITY_DEFICIT,
                                  "persistence": VR_PERSISTENCE_4_1}, p=1, alpha=alpha)
    assert list(result) == ["distance", "hyperbolicity", "persistence"]
    assert [c.statement for c in result.values()] == ["Prop 6.6", "Prop 6.6", "Cor 6.3"]
    assert result["persistence"].accuracy == pytest.approx(wasserstein_bound(53_433_249, alpha, 4, 1))
    assert result["persistence"].accuracy <= 0.1
    assert result["hyperbolicity"].accuracy == pytest.approx(5.6130e-4, rel=1e-4)
    assert result["distance"].accuracy == result["hyperbolicity"].accuracy


def test_guaranteed_accuracy_takes_the_sharpest_bound():
    """Through mu_k at lam = lip (Cor 6.3), its refinement below lip (Cor 6.5), or Prop 6.6."""
    general = Invariant(lambda rows: rows[:, 0], n=4, lip=2, name="general")
    refined = Invariant(lambda rows: rows[:, 0], n=4, lip=2, lam=1, name="refined")
    real = Invariant(lambda rows: rows[:, 0], n=4, lip=2, lam=1, real_valued=True, name="real")
    result = guaranteed_accuracy(10**8, [general, refined, real], p=1)
    assert [c.statement for c in result.values()] == ["Cor 6.3", "Cor 6.5", "Prop 6.6"]
    assert result["real"].accuracy < result["refined"].accuracy < result["general"].accuracy
    assert result["general"].accuracy == pytest.approx(2 * wasserstein_bound(10**8, 0.999, 4, 1))


def test_guaranteed_accuracy_records_its_parameters_and_defaults_to_high_confidence():
    (c,) = guaranteed_accuracy(10**6, [PAIRWISE_DISTANCE], p=2).values()
    assert (c.invariant, c.n_samples, c.alpha, c.p) == ("pairwise_distance", 10**6, 0.999, 2)


def test_guaranteed_accuracy_needs_the_constants_of_each_n_invariant():
    with pytest.raises(TypeError, match="carries no constants"):
        guaranteed_accuracy(100, {"first distance": lambda rows: rows[:, 0]}, p=1)
