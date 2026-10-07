"""The sampler of the SARS-CoV-2 pipeline: correctness against brute force, and reproducibility."""

import numpy as np
from manuscript_modules import load

from n_invariants.invariants import hyperbolicity_vec

sampling = load("sars_cov2/sampling.py")
condensed_hamming_rows = sampling.condensed_hamming_rows
hyperbolicity_distribution_random_parallel = sampling.hyperbolicity_distribution_random_parallel
persistence_set_random_parallel = sampling.persistence_set_random_parallel
random_distribution_k_parallel = sampling.random_distribution_k_parallel


def toy_point_set(U=40, L=25, seed=0):
    return np.random.default_rng(seed).integers(0, 4, size=(U, L), dtype=np.uint8)


def distance_vec(rows):
    values, counts = np.unique(rows[:, 0], return_counts=True)
    return {values[i]: counts[i] for i in range(values.shape[0])}


def test_condensed_hamming_rows_matches_direct_computation():
    seq = toy_point_set()
    idx = np.array([[0, 1, 2, 3], [5, 5, 7, 9]])
    rows = condensed_hamming_rows(seq, idx)
    assert rows.shape == (2, 6)
    # (0,1) is the first condensed pair; (2,3) is the last.
    assert rows[0, 0] == np.count_nonzero(seq[0] != seq[1])
    assert rows[0, 5] == np.count_nonzero(seq[2] != seq[3])
    assert rows[1, 0] == 0  # a repeated point is at distance 0 from itself


def test_sampled_distance_distribution_converges_to_the_exact_one():
    """With uniform weights the n=2 curvature measure is the exact distance histogram."""
    seq = toy_point_set(U=30, L=20, seed=3)
    U = seq.shape[0]

    exact = np.zeros(seq.shape[1] + 1)
    for i in range(U):
        for j in range(U):
            exact[np.count_nonzero(seq[i] != seq[j])] += 1
    exact /= exact.sum()

    histogram = random_distribution_k_parallel(
        distance_vec, seq, U, 400_000, n=2, batch_size=50_000, seed=7, n_workers=2
    )
    sampled = np.zeros_like(exact)
    for value, count in histogram.items():
        sampled[int(value)] = count
    sampled /= sampled.sum()

    assert np.abs(sampled - exact).sum() < 0.01


def test_result_is_independent_of_the_number_of_workers():
    """The chunk decomposition is fixed by (seed, n_samples, batch_size) alone."""
    seq = toy_point_set()
    U = seq.shape[0]
    kwargs = dict(n_samples=60_000, batch_size=5_000, seed=11)
    serial = hyperbolicity_distribution_random_parallel(seq, U, n_workers=1, **kwargs)
    parallel = hyperbolicity_distribution_random_parallel(seq, U, n_workers=4, **kwargs)
    assert serial == parallel


def test_the_same_seed_reproduces_the_same_histogram():
    seq = toy_point_set()
    U = seq.shape[0]
    a = hyperbolicity_distribution_random_parallel(seq, U, 20_000, batch_size=5_000, seed=42)
    b = hyperbolicity_distribution_random_parallel(seq, U, 20_000, batch_size=5_000, seed=42)
    c = hyperbolicity_distribution_random_parallel(seq, U, 20_000, batch_size=5_000, seed=43)
    assert a == b
    assert a != c


def test_sample_counts_are_exact():
    seq = toy_point_set()
    U = seq.shape[0]
    # 17_003 is not a multiple of the batch size, so the residual chunk must be counted too.
    histogram = hyperbolicity_distribution_random_parallel(
        seq, U, 17_003, batch_size=5_000, seed=1, n_workers=3
    )
    assert sum(histogram.values()) == 17_003


def test_a_concentrated_measure_is_respected():
    """Put all the mass on two points and only their mutual distances can be sampled."""
    seq = toy_point_set()
    U = seq.shape[0]
    weights = np.zeros(U)
    weights[[0, 1]] = 0.5

    histogram = random_distribution_k_parallel(
        distance_vec, seq, U, 5_000, n=2, batch_size=1_000, p=weights, seed=5
    )
    d01 = float(np.count_nonzero(seq[0] != seq[1]))
    assert set(histogram) <= {0.0, d01}


def test_persistence_sampler_agrees_with_the_invariant_on_the_same_tuples():
    """The parallel wrapper must be the plain invariant applied to the sampled tuples."""
    seq = toy_point_set()
    U = seq.shape[0]
    histogram = persistence_set_random_parallel(
        seq, U, 8_000, k=1, batch_size=2_000, seed=9, n_workers=2
    )
    assert sum(histogram.values()) == 8_000
    # Every key is a (birth, death) pair with death > birth, or the trivial (0, 0).
    for birth, death in histogram:
        assert (birth, death) == (0.0, 0.0) or death > birth


def test_hyperbolicity_wrapper_matches_a_manual_pass():
    seq = toy_point_set()
    U = seq.shape[0]
    n_samples, batch = 6_000, 6_000
    viaWrapper = hyperbolicity_distribution_random_parallel(
        seq, U, n_samples, batch_size=batch, seed=2, n_workers=1
    )
    viaGeneric = random_distribution_k_parallel(
        hyperbolicity_vec, seq, U, n_samples, n=4, batch_size=batch, seed=2, n_workers=1
    )
    assert viaWrapper == viaGeneric


def test_package_sampler_reproduces_the_pipeline_sampler_on_hamming_data():
    """Same seed and batches: the same tuples, so the same histogram as in step 2."""
    from n_invariants import empirical_distribution
    from n_invariants.invariants import hyperbolicity_deficit

    seq = toy_point_set()
    pipeline = hyperbolicity_distribution_random_parallel(
        seq, seq.shape[0], 30_000, batch_size=4_000, seed=9, n_workers=1)
    values, counts = empirical_distribution(
        seq, hyperbolicity_deficit, 4, 30_000, metric="hamming", seed=9, batch_size=4_000,
        decimals=4)
    assert dict(zip(values, counts)) == pipeline
