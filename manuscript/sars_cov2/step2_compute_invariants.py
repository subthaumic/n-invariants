"""Step 2: estimate the distributions of three metric n-point invariants at each cutoff.

All three are computed from one sample of quadruples per cutoff:

    pairwise distance                  of the first pair (x_1, x_2) of each quadruple
    four-point hyperbolicity deficit   of the quadruple
    (4,1)-persistence                  of the quadruple

The number of quadruples comes from ``n_invariants.sample_size`` for the curvature measure
mu_4 (Corollary 6.3), so the one sample serves every Lipschitz 4-point invariant at once
(Remark 6.4). It does not depend on the number of points U.

Each cutoff date reweights the fixed point set: the slice measure is
p = bincount(uid[dates <= cutoff]) / (its sum), and tuples are sampled from (point set, p).

Reads:  work/curvature_set/{unique.npy, unique_meta.npz} (step 1),
        work/curvature_set/{sequence_to_unique.npy, collection_dates.tsv} (step 0).
Writes: results/invariants/invariants_<cutoff>.npz  (one per date slice).
"""

import gc
import os
import sys
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np

from n_invariants.bounds import sample_size, wasserstein_bound
from sampling import joint_invariants_random_parallel
from sars_utils import HAPLOTYPE_META, HAPLOTYPES, SEQ_TO_UNIQUE, display_time, read_record_dates

folder_invariants = Path(__file__).resolve().parent / 'results' / 'invariants'
folder_invariants.mkdir(parents=True, exist_ok=True)

# One slice per cutoff date: the reweighting to every haplotype seen on or before it.
# Five slices tracing the pandemic's variant phases. Omicron is the last WHO variant of
# concern; everything after it is an Omicron descendant/recombinant (XBB, then JN.1), which
# is what the final full-dataset slice captures. The alignment was downloaded 24 Feb 2024
# (see Appendix B) and its last collection date is 2024-01-31, so the final cutoff is the
# complete dataset.
CUTOFF_DATES = [
    '2020-06-30',  # pre-VOC: ancestral + D614G (Alpha not designated until Dec 2020)
    '2021-04-30',  # pre-Delta: Alpha-dominant, Beta/Gamma co-circulating (Delta a VOC May 2021)
    '2021-10-31',  # pre-Omicron: Delta-dominated (Omicron designated 26 Nov 2021)
    '2022-03-01',  # Omicron: BA.1/BA.2 wave, pre-recombinant (XBB emerges Aug-Oct 2022)
    # '2022-08-01',  # ALT Omicron (replaces 2022-03-01): full BA.1/2/4/5 radiation, before XBB
    #                # was first sequenced. Keep EXPECTED_CUTOFFS in
    #                # ../figures/figure_7_sars_geometry.py in sync when swapping.
    '2024-01-31',  # full dataset (last collection date): Omicron recombinants / XBB era
]

# ----------------------------
# Sampling parameters
# ----------------------------
p_order = 1     # Wasserstein distance of order p
q = np.inf      # mu has finite moments of order q
m = np.inf      # equip K_n(X) with the L^m metric
n = 4           # tuple size: one sample of quadruples serves all three invariants

# The sample size is set by mu_4 (Corollary 6.3): every 1-Lipschitz 4-point invariant, in
# particular the (4,1)-persistence, is then within epsilon * diam(X) in W_p. The distance and
# the hyperbolicity deficit are real-valued with range constant 1, i.e.
# diam F(K_4(X)) <= diam(X), so Proposition 6.6 bounds their error on the same sample at the
# much faster rate N^{-1/2} (eps_real below).
epsilon = 0.1   # accuracy for mu_4, i.e. for every 1-Lipschitz 4-point invariant (as in Section 8.3)
alpha = 0.95    # simultaneous confidence for all bounds at all cutoffs

# Union bound over three bounds per cutoff: Corollary 6.3 for mu_4 (which covers the
# persistence) and Proposition 6.6 for the distance and for the hyperbolicity deficit.
n_events = 3 * len(CUTOFF_DATES)
alpha_event = 1 - (1 - alpha) / n_events

# Lipschitz constants (Prop 4.9): hyperbolicity is 2-Lipschitz, degree-1 persistence and the
# pairwise distance are 1-Lipschitz. Only the persistence relies on them here; the two
# real-valued invariants use Proposition 6.6, which needs only their range constant 1.
L_hyperbolicity = 2
L_persistence = 1
L_distance = 1

# Existing results are never replaced unless this is True.
OVERWRITE = False

# On-the-fly Hamming distances hold ~3 * batch_size * L_poly bytes per worker at peak (two
# gathered (batch, L_poly) uint8 arrays plus their boolean). At L_poly ~ 27k and batch_size ~ 1e4,
# this is ~0.8 GiB per worker at batch_size 1e4; the whole pool then peaks near n_workers * that.
batch_size = 1e4
seed = 402

# 1 = serial, -1 = all cores. The point-set array is shared across workers via
# copy-on-write, not copied.
n_workers = -1

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def compute_invariants():
    started = datetime.now()
    print(f'Started {started:{TIME_FORMAT}}')

    seq = np.load(HAPLOTYPES)                        # uint8[U, L_poly] point set
    meta = np.load(HAPLOTYPE_META)
    U, L, L_poly = int(meta['U']), int(meta['L']), int(meta['L_poly'])
    assert seq.shape[0] == U, (seq.shape, U)
    uid = np.load(SEQ_TO_UNIQUE)                     # (N,) int32
    dates = read_record_dates()                      # (N,) datetime64[D]
    print(f'U={U:,} haplotypes | N={uid.size:,} records | L={L} sites '
          f'({L_poly:,} polymorphic) | array {seq.nbytes / 2**30:.2f} GiB')

    n_samples = int(sample_size(epsilon, alpha_event, n, p_order))
    # Achieved accuracies at this N, in units of diam(X). Proposition 6.6 for a real-valued
    # invariant with range constant 1 is the d = 1 case of the same bound, i.e. the n = 2 call.
    eps_mu4 = float(wasserstein_bound(n_samples, alpha_event, n, p_order))
    eps_real = float(wasserstein_bound(n_samples, alpha_event, 2, p_order))
    print(f'quadruples per cutoff: {n_samples:,} (confidence {alpha} over {n_events} bounds, '
          f'{alpha_event:.5f} each)')
    print(f'accuracy: persistence {eps_mu4:.4f}, distance and hyperbolicity {eps_real:.2e} '
          f'(units of diam X)')

    for snap_idx, cutoff in enumerate(CUTOFF_DATES):
        out_path = folder_invariants / f'invariants_{cutoff}.npz'
        if out_path.is_file() and not OVERWRITE:
            print(f'-- {cutoff}: {out_path.name} exists; skipping (set OVERWRITE = True).')
            continue

        cutoff64 = np.datetime64(cutoff, 'D')
        mask = dates <= cutoff64
        w = np.bincount(uid[mask], minlength=U).astype(np.float64)
        M = int(w.sum())                             # sequences in this slice
        if M == 0:
            print(f'-- {cutoff}: no sequences on or before cutoff; skipping.')
            continue
        U_slice = int((w > 0).sum())                 # distinct haplotypes present
        pvec = w / w.sum()                           # slice measure over the U haplotypes

        print('--------------------------')
        print(f'Snapshot {cutoff}: M={M:,} sequences, U={U_slice:,} haplotypes')

        t0 = time()
        started_slice = datetime.now()
        print(f'Sampling quadruples... (started {started_slice:{TIME_FORMAT}})')
        dm_dict, hyp_hist, bd_dict = joint_invariants_random_parallel(
            seq, U, n_samples, batch_size=batch_size, p=pvec,
            seed=np.random.SeedSequence([seed, snap_idx, 0]), n_workers=n_workers)
        runtime = time() - t0
        finished_slice = datetime.now()
        print(f'  finished {finished_slice:{TIME_FORMAT}}, elapsed {display_time(runtime)}')

        # `diam` is the largest sampled distance between the first two points of a quadruple.
        diam = int(max(dm_dict)) if dm_dict else 0

        # Save the raw sampled histograms (dicts) + metadata. Plotting is a separate step
        # that reads these back; this script only computes. All three histograms come from
        # the same n_samples quadruples; the per-invariant counts are kept for the readers.
        np.savez(
            out_path,
            dm_dict=dm_dict, hyp_hist=hyp_hist, bd_dict=bd_dict,
            cutoff_date=str(cutoff),
            n_sequences=M, n_haplotypes=U_slice, n_sites=int(L), diam=diam,
            n_samples=n_samples,
            n_samples_distance=n_samples,
            n_samples_hyperbolicity=n_samples,
            n_samples_persistence=n_samples,
            L_hyperbolicity=L_hyperbolicity, L_persistence=L_persistence, L_distance=L_distance,
            batch_size=batch_size, seed=seed,
            p=p_order, q=q, m=m, n=n,
            epsilon=epsilon, epsilon_persistence=eps_mu4,
            epsilon_distance=eps_real, epsilon_hyperbolicity=eps_real,
            alpha=alpha, alpha_event=alpha_event, n_events=n_events,
            runtime_seconds=runtime,
            started=started_slice.isoformat(timespec='seconds'),
            finished=finished_slice.isoformat(timespec='seconds'),
            n_workers=(os.cpu_count() or 1) if n_workers < 0 else n_workers)

        del hyp_hist, bd_dict, dm_dict, w, pvec
        gc.collect()

    finished = datetime.now()
    print('--------------------------')
    print(f'Finished {finished:{TIME_FORMAT}}, '
          f'elapsed {display_time((finished - started).total_seconds())}')


if __name__ == '__main__':
    # Line-buffered, so the progress shows up as it happens when the output is piped to tee.
    sys.stdout.reconfigure(line_buffering=True)
    compute_invariants()
