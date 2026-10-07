"""Step 1: store the haplotypes as a compact array for the sampler.

    1. Load the U distinct haplotypes (length L) into a dense uint8 array (~U*L bytes).
    2. Drop every constant column (these add 0 to every pairwise distance).
    -> work/curvature_set/unique.npy                   uint8[U, L_poly]  the point set
    -> work/curvature_set/polymorphic_columns.npy      int32[L_poly]     kept column indices in [0, L)
    -> work/curvature_set/unique_meta.npz              U, L, L_poly
"""
import os
import sys
import time

import numpy as np

from sars_utils import (
    CURVATURE_SET_DIR,
    HAPLOTYPE_META,
    HAPLOTYPES,
    MULTIPLICITY,
    POLY_COLUMNS,
    UNIQUE_FASTA,
    iter_records,
)

REPORT_EVERY = 100_000


def load_unique_alignment(fasta, U):
    """Dense uint8[U, L] of the U distinct haplotypes (raw ASCII bytes; only != matters)."""
    arr = None
    L = None
    for i, (_, seq) in enumerate(iter_records(fasta)):
        row = np.frombuffer(seq, dtype=np.uint8)
        if arr is None:
            L = row.size
            arr = np.empty((U, L), dtype=np.uint8)
        elif row.size != L:
            raise ValueError(f'haplotype {i} has length {row.size}, expected {L} '
                             '(unique.fasta is not a fixed-width alignment)')
        if i >= U:
            raise ValueError(f'unique.fasta has more than the expected U={U:,} records')
        arr[i] = row
        if (i + 1) % REPORT_EVERY == 0:
            print(f'  loaded {i+1:>12,} / {U:,} haplotypes', file=sys.stderr, flush=True)
    if arr is None:
        raise ValueError('unique.fasta is empty')
    if i + 1 != U:
        raise ValueError(f'unique.fasta has {i+1:,} records, expected U={U:,}')
    return arr, L


def build_haplotype_array(fasta=UNIQUE_FASTA, multiplicity=MULTIPLICITY):
    if not os.path.exists(multiplicity):
        print(f'measure not found at {multiplicity}; run step0_produce_alignment.py first.',
              file=sys.stderr)
        return
    if os.path.exists(HAPLOTYPES):
        print(f'{HAPLOTYPES} exists -> nothing to do.', file=sys.stderr)
        return

    os.makedirs(CURVATURE_SET_DIR, exist_ok=True)
    U = int(np.load(multiplicity).size)
    print(f'point set: U={U:,} distinct haplotypes',
          file=sys.stderr)

    t0 = time.time()
    arr, L = load_unique_alignment(fasta, U)
    print(f'loaded {U:,} x {L:,} uint8 array ({arr.nbytes / 2**30:.1f} GiB) in '
          f'{(time.time()-t0)/60:.1f} min', file=sys.stderr)

    # A column is constant iff every haplotype agrees with the first one there.
    t0 = time.time()
    poly = np.any(arr != arr[0], axis=0)
    keep = np.nonzero(poly)[0].astype(np.int32)
    compact = np.ascontiguousarray(arr[:, keep])
    del arr
    print(f'polymorphic columns: {keep.size:,} of {L:,} '
          f'({100 * keep.size / L:.2f}%) kept in {(time.time()-t0):.1f}s',
          file=sys.stderr)
    print(f'point set array: {compact.shape[0]:,} x {compact.shape[1]:,} uint8 '
          f'({compact.nbytes / 2**30:.2f} GiB)', file=sys.stderr)

    np.save(HAPLOTYPES, compact)
    np.save(POLY_COLUMNS, keep)
    np.savez(HAPLOTYPE_META, U=U, L=int(L), L_poly=int(keep.size))
    print(f'-> {HAPLOTYPES}', file=sys.stderr)
    print(f'-> {POLY_COLUMNS}', file=sys.stderr)
    print(f'-> {HAPLOTYPE_META}', file=sys.stderr)


if __name__ == '__main__':
    build_haplotype_array()
