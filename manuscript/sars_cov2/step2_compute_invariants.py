"""Step 2: estimate the distributions of three metric n-point invariants at each cutoff.

    pairwise distance                  n = 2
    four-point hyperbolicity deficit   n = 4
    (4,1)-persistence                  n = 4

The number of samples for a target accuracy and confidence comes from
``n_invariants.sample_size`` (Corollary 6.5) and does not depend on the number of points U.

Each cutoff date reweights the fixed point set: the slice measure is
p = bincount(uid[dates <= cutoff]) / (its sum), and tuples are sampled from (point set, p).

Reads:  work/curvature_set/{unique.npy, unique_meta.npz} (step 1),
        work/curvature_set/{sequence_to_unique.npy, collection_dates.tsv} (step 0).
Writes: results/invariants/invariants_<cutoff>.npz  (one per date slice).
"""

import gc
from pathlib import Path
from time import time

import numpy as np

from n_invariants.bounds import sample_size
from sampling import (
    hyperbolicity_distribution_random_parallel,
    persistence_set_random_parallel,
    random_distribution_k_parallel,
)
from sars_utils import HAPLOTYPE_META, HAPLOTYPES, SEQ_TO_UNIQUE, display_time, read_record_dates

folder_invariants = Path(__file__).resolve().parent / 'results' / 'invariants'
folder_invariants.mkdir(parents=True, exist_ok=True)

# One slice per cutoff date: the reweighting to every haplotype seen on or before it.
# Five slices tracing the pandemic's variant phases. Omicron is the last WHO variant of
# concern; everything after it is an Omicron descendant/recombinant (XBB, then JN.1), which
# is what the final full-dataset slice captures. The alignment was downloaded 24 Feb 2024
# (see Appendix B), so the 2024-03-01 cutoff is the complete dataset.
CUTOFF_DATES = [
    '2020-06-30',  # pre-VOC: ancestral + D614G (Alpha not designated until Dec 2020)
    '2021-04-30',  # pre-Delta: Alpha-dominant, Beta/Gamma co-circulating (Delta a VOC May 2021)
    '2021-10-31',  # pre-Omicron: Delta-dominated (Omicron designated 26 Nov 2021)
    '2022-03-01',  # Omicron: BA.1/BA.2 wave, pre-recombinant (XBB emerges Aug-Oct 2022)
    # '2022-08-01',  # ALT Omicron (replaces 2022-03-01): full BA.1/2/4/5 radiation, before XBB
    #                # was first sequenced. Needs a fresh hyperbolicity run for this slice; keep
    #                # EXPECTED_CUTOFFS in ../figures/figure_7_sars_geometry.py in sync when swapping.
    '2024-03-01',  # full dataset: Omicron recombinants / XBB era
]

# ----------------------------
# Sampling parameters
# ----------------------------
p_order = 1     # Wasserstein distance of order p
q = np.inf      # mu has finite moments of order q
m = np.inf      # equip K_n(X) with the L^m metric
n = 4           # index of the curvature set for hyperbolicity / persistence

epsilon = 0.1       # maximum error for the four-point invariants (hyperbolicity, persistence)
epsilon_dist = 0.001  # separate, tighter target for the n=2 distance distribution; it is cheap
                      # (N grows only as eps^-2 for n=2), so ~1.3e7 samples cost seconds
alpha = 0.95    # confidence level

# Which invariants to (re)compute. Set one to False to keep whatever is already stored in
# invariants_<cutoff>.npz and only recompute the rest, then re-save with the untouched
# invariant carried over verbatim. This lets us bump the distance/persistence samples
# without ever paying for hyperbolicity again (it is ~64x more expensive: L=2 in (L/eps)^6).
# On a slice that has no existing npz yet, a disabled invariant is computed anyway (there is
# nothing to carry over) and a warning is printed.
RECOMPUTE = {
    'distance':      True,
    'hyperbolicity': True,
    'persistence':   True,
}

# Lipschitz constants: hyperbolicity is 2-Lipschitz, degree-1 persistence and the
# pairwise distance are 1-Lipschitz (Prop 4.8). All three have range constant 1, i.e.
# diam F(K_n(X)) <= diam(X), so the sample counts are N_lambda=1 of Corollary 6.5.
L_hyperbolicity = 2
L_persistence = 1
L_distance = 1
lam = 1

# On-the-fly Hamming distances hold ~3 * batch_size * L_poly bytes per worker at peak (two
# gathered (batch, L_poly) uint8 arrays plus their boolean). At L_poly ~ 27k and batch_size ~ 1e4,
# this is ~0.8 GiB per worker at batch_size 1e4; the whole pool then peaks near n_workers * that.
batch_size = 1e4
seed = 402

# 1 = serial, -1 = all cores. The point-set array is shared across workers via
# copy-on-write, not copied.
n_workers = -1


def distance_vec(dm_row_sq):
    """Histogram of pairwise distances (n=2 curvature set). ``dm_row_sq`` is (size, 1)."""
    d = dm_row_sq[:, 0]
    vals, counts = np.unique(d, return_counts=True)
    return {vals[idx]: counts[idx] for idx in range(vals.shape[0])}


def compute_invariants():
    seq = np.load(HAPLOTYPES)                        # uint8[U, L_poly] point set
    meta = np.load(HAPLOTYPE_META)
    U, L, L_poly = int(meta['U']), int(meta['L']), int(meta['L_poly'])
    assert seq.shape[0] == U, (seq.shape, U)
    uid = np.load(SEQ_TO_UNIQUE)                     # (N,) int32
    dates = read_record_dates()                      # (N,) datetime64[D]
    print(f'U={U:,} haplotypes | N={uid.size:,} records | L={L} sites '
          f'({L_poly:,} polymorphic) | array {seq.nbytes / 2**30:.2f} GiB')

    n_samples_hyp = sample_size(epsilon, alpha, n, p_order, lip=L_hyperbolicity, lam=lam)
    n_samples_per = sample_size(epsilon, alpha, n, p_order, lip=L_persistence, lam=lam)
    n_samples_dist = sample_size(epsilon_dist, alpha, 2, p_order, lip=L_distance, lam=lam)
    print(f'samples: distance={n_samples_dist:.2e} hyperbolicity={n_samples_hyp:.2e} '
          f'persistence={n_samples_per:.2e}')
    print('recompute: ' + ', '.join(f'{k}={v}' for k, v in RECOMPUTE.items()))

    for snap_idx, cutoff in enumerate(CUTOFF_DATES):
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

        # Anything not (re)computed this run is carried over verbatim from the existing npz.
        out_path = folder_invariants / f'invariants_{cutoff}.npz'
        prev = dict(np.load(out_path, allow_pickle=True)) if out_path.is_file() else {}

        # RECOMPUTE keys -> the array name each invariant is stored under in the npz.
        STORE_KEY = {'distance': 'dm_dict', 'hyperbolicity': 'hyp_hist', 'persistence': 'bd_dict'}

        def resolve(name, recompute_fn):
            """Recompute `name` if RECOMPUTE asks or it is absent from `prev`; else carry over."""
            key = STORE_KEY[name]
            if RECOMPUTE[name] or key not in prev:
                if not RECOMPUTE[name]:
                    print(f'  ! {name} not in existing npz -> computing it anyway')
                print(f'Computing {name}...')
                t0 = time()
                result = recompute_fn()
                print(f'  {display_time(time() - t0)}')
                return result, True
            print(f'Keeping stored {name} (RECOMPUTE[{name!r}]=False)')
            return prev[key], False

        # Distance distribution (n=2 curvature set), sampled on the fly.
        dm_obj, dist_new = resolve('distance', lambda: random_distribution_k_parallel(
            distance_vec, seq, U, n_samples_dist, n=2, square=False,
            batch_size=batch_size, p=pvec,
            seed=np.random.SeedSequence([seed, snap_idx, 2]), n_workers=n_workers))
        dm_dict = dm_obj.item() if isinstance(dm_obj, np.ndarray) else dm_obj

        # Hyperbolicity distribution (by far the most expensive; usually carried over).
        hyp_obj, hyp_new = resolve('hyperbolicity', lambda: hyperbolicity_distribution_random_parallel(
            seq, U, n_samples_hyp, batch_size=batch_size, p=pvec,
            seed=np.random.SeedSequence([seed, snap_idx, 0]), n_workers=n_workers))
        hyp_hist = hyp_obj.item() if isinstance(hyp_obj, np.ndarray) else hyp_obj

        # Degree-1 persistence set.
        bd_obj, per_new = resolve('persistence', lambda: persistence_set_random_parallel(
            seq, U, n_samples_per, 1, batch_size=batch_size, p=pvec,
            seed=np.random.SeedSequence([seed, snap_idx, 1]), n_workers=n_workers))
        bd_dict = bd_obj.item() if isinstance(bd_obj, np.ndarray) else bd_obj

        # `diam` is the sampled max pairwise distance; refresh it only when distance was
        # recomputed, otherwise keep the stored estimate.
        if dist_new:
            diam = int(max(dm_dict)) if dm_dict else 0
        else:
            diam = int(prev['diam'])

        # Per-invariant sample counts / epsilons record what produced the stored
        # histogram: the fresh value if recomputed, else the value carried over from `prev`.
        n_dist = n_samples_dist if dist_new else int(prev['n_samples_distance'])
        n_hyp = n_samples_hyp if hyp_new else int(prev['n_samples_hyperbolicity'])
        n_per = n_samples_per if per_new else int(prev['n_samples_persistence'])
        eps_dist_used = epsilon_dist if dist_new else float(prev.get('epsilon_distance', epsilon))

        # Save the raw sampled histograms (dicts) + metadata. Plotting is a separate step
        # that reads these back; this script only computes.
        np.savez(
            out_path,
            dm_dict=dm_dict, hyp_hist=hyp_hist, bd_dict=bd_dict,
            cutoff_date=str(cutoff),
            n_sequences=M, n_haplotypes=U_slice, n_sites=int(L), diam=diam,
            n_samples_distance=n_dist,
            n_samples_hyperbolicity=n_hyp,
            n_samples_persistence=n_per,
            L_hyperbolicity=L_hyperbolicity, L_persistence=L_persistence, L_distance=L_distance,
            batch_size=batch_size, seed=seed,
            p=p_order, q=q, m=m, n=n,
            epsilon=epsilon, epsilon_distance=eps_dist_used, alpha=alpha)

        del hyp_hist, bd_dict, dm_dict, w, pvec, prev
        gc.collect()


if __name__ == '__main__':
    compute_invariants()
