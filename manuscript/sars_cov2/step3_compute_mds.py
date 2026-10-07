#!/usr/bin/env python3
"""Step 3: 2D metric-MDS embedding of a subsample of the haplotypes, for Figure 7.

MDS needs a distance matrix, which is too large to compute over all U haplotypes. The
script therefore draws a uniform subsample of MAX_POINTS haplotypes and computes only
their distance matrix. Each embedded haplotype is tagged with its first-appearance date,
which determines its colour in Figure 7.

Optionally, the alignment's reference genome (REFERENCE_ACCESSION) is pinned into the
subsample and its row flagged in the outputs, so the plot can mark the root of the
radiation.

Reads:  work/curvature_set/unique.npy (step 1),
        work/curvature_set/{sequence_to_unique.npy, collection_dates.tsv} (step 0).
Writes:
    results/mds/mds_embedding_max<MAX_POINTS>.npz
    results/mds/mds_embedding_max<MAX_POINTS>.csv
        haplotype_index,mds_1,mds_2,first_seen_date,is_reference
"""

from __future__ import annotations

import sys
import time
from inspect import signature
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

from sars_utils import HAPLOTYPES, SEQ_TO_UNIQUE, read_record_accessions, read_record_dates

try:
    from sklearn.manifold import MDS
except ImportError:
    MDS = None


MDS_DIR = Path(__file__).resolve().parent / "results" / "mds"

# ----------------------------
# Configuration (edit here)
# ----------------------------
MAX_POINTS = 50000      # uniform subsample size to embed (landmark block is MAX_POINTS x MAX_POINTS)
                        # 50,000 is the size shown in Figure 7. The article's run took
                        # ~27 h; the resulting embedding is stored in results/mds/.

# None draws a plain uniform subsample. Set to the alignment's reference genome,
# "EPI_ISL_402124" (hCoV-19/Wuhan/WIV04/2019), to pin it into the subsample and flag it,
# so the plot can mark the root of the radiation.
REFERENCE_ACCESSION: str | None = None

OVERWRITE = False

SUBSAMPLE_SEED: int | None = 42
MDS_RANDOM_STATE: int | None = 42
MDS_N_INIT = 4
MDS_MAX_ITER = 300
MDS_JOBS = 1


def load_first_seen_dates(n_haplotypes: int) -> np.ndarray:
    """First-appearance date per haplotype as datetime64[D] (min over records)."""
    uid = np.load(SEQ_TO_UNIQUE)               # int32[N]
    dates = read_record_dates()                # datetime64[D], row-aligned with uid
    if uid.shape[0] != dates.shape[0]:
        raise ValueError(
            f"sequence_to_unique has {uid.shape[0]:,} records but "
            f"collection_dates has {dates.shape[0]:,}."
        )

    days = dates.astype("int64")
    first_days = np.full(n_haplotypes, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_days, uid, days)
    if np.any(first_days == np.iinfo(np.int64).max):
        raise ValueError("Some haplotype has no dated record; cannot set a first-appearance date.")
    return first_days.astype("datetime64[D]")


def find_reference_haplotype(accession: str | None) -> int | None:
    """Haplotype index of the reference genome, or None if it is unset or did not survive filtering."""
    if not accession:
        return None
    uid = np.load(SEQ_TO_UNIQUE)                       # int32[N], row-aligned with accessions
    accs = read_record_accessions()
    if accs.shape[0] != uid.shape[0]:
        raise ValueError(
            f"sequence_to_unique has {uid.shape[0]:,} records but "
            f"collection_dates has {accs.shape[0]:,}."
        )
    rows = np.flatnonzero(accs == accession)
    if rows.size == 0:
        print(f"[warn] reference accession {accession} not found among records; "
              f"embedding without it", flush=True)
        return None
    return int(uid[rows[0]])


def select_indices(n_points: int, max_points: int, rng: np.random.Generator,
                   pin: int | None = None) -> np.ndarray:
    """Uniform subsample of ``max_points`` indices, always including ``pin`` if given."""
    if n_points <= max_points:
        return np.arange(n_points, dtype=int)
    chosen = rng.choice(n_points, size=max_points, replace=False)
    if pin is not None and pin not in chosen:
        # Swap the pinned index in for the first non-pinned draw, keeping the size at max_points.
        chosen[0] = pin
    return np.sort(np.unique(chosen))


def landmark_distance_block(seq: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Symmetric Hamming distance matrix of the landmark subsample, computed on the fly.

    ``pdist(..., 'hamming')`` returns the *fraction* of differing sites; multiplying by the
    number of (polymorphic) columns recovers the integer Hamming distance, which equals the
    full-length Hamming distance since constant columns were dropped.
    """
    sub = seq[indices]                                 # (m, L_poly) uint8
    L_poly = sub.shape[1]
    condensed = pdist(sub, metric="hamming") * L_poly  # (m*(m-1)/2,)
    dm = squareform(condensed)
    np.fill_diagonal(dm, 0.0)
    return dm


def compute_mds_embedding(
    dm: np.ndarray,
    random_state: int | None,
    n_init: int,
    max_iter: int,
    n_jobs: int | None,
) -> np.ndarray:
    if MDS is None:
        raise RuntimeError(
            "Missing dependency 'scikit-learn'. Install it before running this script."
        )

    parameters = signature(MDS).parameters
    mds_kwargs = dict(
        n_components=2,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    if "n_jobs" in parameters:
        mds_kwargs["n_jobs"] = n_jobs
    if "init" in parameters:
        mds_kwargs["init"] = "random"
    if "metric_mds" in parameters:
        mds_kwargs["metric_mds"] = True
        mds_kwargs["metric"] = "precomputed"
    else:
        if "metric" in parameters:
            mds_kwargs["metric"] = True
        if "dissimilarity" in parameters:
            mds_kwargs["dissimilarity"] = "precomputed"
        else:
            raise RuntimeError(
                "Installed scikit-learn MDS does not support precomputed distances."
            )
    model = MDS(**mds_kwargs)
    return model.fit_transform(dm)


def output_paths(max_points: int) -> tuple[Path, Path]:
    stem = f"mds_embedding_max{max_points}"
    return MDS_DIR / f"{stem}.npz", MDS_DIR / f"{stem}.csv"


def write_outputs(
    max_points: int,
    indices: np.ndarray,
    mds_embedding: np.ndarray,
    first_seen: np.ndarray,
    total_points: int,
    reference_index: int | None,
    reference_row: int,
    mds_random_state: int | None,
    subsample_seed: int | None,
    mds_seconds: float,
    run_seconds: float,
) -> tuple[Path, Path]:
    MDS_DIR.mkdir(parents=True, exist_ok=True)
    out_npz, out_csv = output_paths(max_points)

    np.savez(
        out_npz,
        computed_at_unix_time=time.time(),
        mds_variant="metric_precomputed_sklearn_landmark",
        requested_max_points=max_points,
        embedding_indices=indices,
        mds_embedding=mds_embedding,
        first_seen_date=first_seen,
        n_haplotypes_total=total_points,
        n_haplotypes_used=indices.shape[0],
        reference_accession=REFERENCE_ACCESSION or "",
        reference_index=-1 if reference_index is None else reference_index,
        reference_row=reference_row,  # row within the embedding, or -1 if absent
        mds_n_init=MDS_N_INIT,
        mds_max_iter=MDS_MAX_ITER,
        mds_n_jobs=-1 if MDS_JOBS is None else MDS_JOBS,
        mds_random_state=-1 if mds_random_state is None else mds_random_state,
        subsample_seed=-1 if subsample_seed is None else subsample_seed,
        mds_seconds=mds_seconds,
        run_seconds=run_seconds,
    )

    with out_csv.open("w", encoding="utf-8") as handle:
        handle.write("haplotype_index,mds_1,mds_2,first_seen_date,is_reference\n")
        for row, (idx, (x, y), day) in enumerate(zip(indices, mds_embedding, first_seen)):
            is_ref = int(row == reference_row)
            handle.write(
                f"{int(idx)},{x:.8f},{y:.8f},{np.datetime_as_string(day, unit='D')},{is_ref}\n"
            )

    return out_npz, out_csv


def process_max_points(
    seq: np.ndarray,
    first_seen: np.ndarray,
    max_points: int,
) -> None:
    out_npz, out_csv = output_paths(max_points)
    if not OVERWRITE and out_npz.is_file() and out_csv.is_file():
        print(
            f"[info] outputs already exist for max_points={max_points}; "
            "set OVERWRITE=True to recompute",
            flush=True,
        )
        return

    total_points = seq.shape[0]
    subsample_seed = SUBSAMPLE_SEED
    mds_random_state = MDS_RANDOM_STATE

    run_start = time.perf_counter()
    rng = np.random.default_rng() if subsample_seed is None else np.random.default_rng(subsample_seed)
    reference_index = find_reference_haplotype(REFERENCE_ACCESSION)
    indices = select_indices(total_points, max_points, rng, pin=reference_index)
    reference_row = int(np.flatnonzero(indices == reference_index)[0]) if reference_index is not None else -1
    if reference_index is not None:
        print(f"[info] reference {REFERENCE_ACCESSION} pinned as haplotype {reference_index} "
              f"(embedding row {reference_row})", flush=True)
    dm_used = landmark_distance_block(seq, indices)

    print(
        f"[info] running metric MDS with max_points={max_points} "
        f"(n_used={indices.shape[0]}/{total_points}, landmark block {dm_used.shape})",
        flush=True,
    )
    mds_start = time.perf_counter()
    mds_embedding = compute_mds_embedding(
        dm_used,
        random_state=mds_random_state,
        n_init=MDS_N_INIT,
        max_iter=MDS_MAX_ITER,
        n_jobs=MDS_JOBS,
    )
    mds_seconds = time.perf_counter() - mds_start
    run_seconds = time.perf_counter() - run_start

    out_npz, out_csv = write_outputs(
        max_points=max_points,
        indices=indices,
        mds_embedding=mds_embedding,
        first_seen=first_seen[indices],
        total_points=total_points,
        reference_index=reference_index,
        reference_row=reference_row,
        mds_random_state=mds_random_state,
        subsample_seed=subsample_seed,
        mds_seconds=mds_seconds,
        run_seconds=run_seconds,
    )
    print(
        f"[info] finished max_points={max_points} in {run_seconds:.2f}s (mds={mds_seconds:.2f}s) "
        f"-> {out_npz.name}, {out_csv.name}",
        flush=True,
    )


def main() -> int:
    if MDS is None:
        raise RuntimeError("Missing dependency 'scikit-learn'.")

    load_start = time.perf_counter()
    seq = np.load(HAPLOTYPES)
    if seq.ndim != 2 or seq.shape[0] < 3:
        raise ValueError(f"Need a (U, L_poly) point set with U >= 3, got {seq.shape}")
    first_seen = load_first_seen_dates(seq.shape[0])
    print(
        f"[info] loaded {seq.shape[0]:,} haplotypes ({seq.shape[1]:,} polymorphic sites) in "
        f"{time.perf_counter() - load_start:.2f}s",
        flush=True,
    )

    process_max_points(seq, first_seen, MAX_POINTS)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
