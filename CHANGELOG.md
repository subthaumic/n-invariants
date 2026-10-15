# Changelog

## 1.2.0 (2026-10-15)

The package now specifically bounds the $n$-th curvature measure, samples it once, reads off one or more $n$-invariants, and returns improved sample guarantees for each if available.

### Added

- `Invariant`, an $n$-invariant together with the constants its bounds need: `n`, `lip`, `lam` and whether it is real-valued. The article's three are predefined as `PAIRWISE_DISTANCE`, `HYPERBOLICITY_DEFICIT` and `VR_PERSISTENCE_4_1`.
- `empirical_distributions`, the distributions of several $n$-invariants from the same sampled tuples in one pass. An `Invariant` on $n'<n$ points evaluates on the first $n'$ points of larger tuples.
- `sample_tuples`, the sampled tuples lazily, batch by batch, with their point indices.
- `sample_size_real_valued`, `wasserstein_bound_real_valued` and `wasserstein_moment_bound_real_valued`: the bounds for real-valued $n$-invariants (Prop. 6.6), with the rate $N^{-1/2}$ whatever $n$ is.
- `guaranteed_accuracy`, the sharpest accuracy that a given number of samples guarantees for each of several $n$-invariants, and the statement it comes from.
- The commands `n-invariants bound-real-valued` and `n-invariants accuracy`.
- `empirical_curvature_measure` can write to a `.npy` file (`out`).
- `empirical_curvature_measure`, `empirical_distributions` and `empirical_distribution` print the percentage of batches done to stderr, in steps of 5% (`progress=False` turns it off).
- `n_workers` of 0 or less uses all CPUs available to the process.

### Changed

- `expected_wasserstein_bound` is renamed `wasserstein_moment_bound`: it bounds the $p$-th moment $E[W_p^p]$, not $W_p$.
- The sample is fixed by the seed, the points with their weights, and `n`. `batch_size` and `n_workers` no longer change the tuples, and the first $M$ tuples of a sample of size $N$ are the sample of size $M$. **For a given seed, the sampled tuples differ from those of 1.1.0.**
- `empirical_curvature_measure` refuses to allocate more than `max_bytes` (default 4 GiB) in memory.
- The command line confidence defaults to 0.999 instead of 0.95.
- Statement numbers follow the current preprint (Prop. 4.9, Lemma 3.13), and the SARS-CoV-2 example uses the single sample of quadruples per cutoff of Section 8.4.

### Article code (`manuscript/`)

- Step 2 of the SARS-CoV-2 pipeline draws one sample of 53,433,249 quadruples per cutoff, sized for the curvature measure $\mu_4$ at accuracy 0.1 (Cor. 6.3), and computes all three distributions from it. The results, Figure 7 and the statistics of Section 8.4 are recomputed from these samples.
- The last cutoff is 2024-01-31, the last collection date in the data, instead of 2024-03-01.
- Step 2 reports its progress and records when each cutoff started and finished.
- `manuscript/` still runs on release v1.1.0 of the package.

## 1.1.0 (2026-10-07)

- The article code in `manuscript/` is its own uv project, pinned to a release of the package and to the library versions the figures were made with.

## 1.0.0 (2026-10-07)

- First release.
