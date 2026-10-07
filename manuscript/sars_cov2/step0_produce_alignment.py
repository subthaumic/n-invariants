"""Step 0: build the metric measure space from the aligned GISAID FASTA.

Turns a nucleotide alignment into the metric measure space the rest of the pipeline works
on, with the distinct sequences as points and their observed multiplicities as measure.
It runs in three stages, each streaming the alignment record by record.

Stage 1 (coverage and date filter).
Drops every sequence that contains a character other than A/C/G/T (either case) or '-',
or whose header lacks a complete YYYY-MM-DD collection date.
    -> work/intermediate/complete.fasta

Stage 2 (gap removal).
Hamming distances are taken over the alphabet A/C/G/T, so all gaps ('-') are removed. A
gap can be the signature indel of a variant, so the sequences carrying it must be kept.
  (1) Columns: a column is kept iff at most a fraction THETA of the sequences have a gap
      there. A common indel (an insertion, or a clade-defining deletion) has a gap in many
      sequences, so its column is dropped; a well-covered site has a gap in only a few
      sequences (ragged ends, private gaps) and is kept.
  (2) Sequences: of the remaining sequences, those with a gap in a kept column are
      dropped, i.e. rare private gaps where almost all other sequences have a base.
The result is a fixed-width alignment over A/C/G/T.
    -> work/intermediate/ungapped.fasta                 N sequences of length L

Stage 3 (deduplication and measure).
Each record is assigned the index of its distinct sequence (haplotype), numbered by order
of first occurrence; counting these indices gives each haplotype's multiplicity. Every
distinct haplotype becomes a point of X (MIN_MULTIPLICITY = 1). A larger MIN_MULTIPLICITY
also drops every haplotype seen fewer than that many times, as a denoising option.
    -> work/curvature_set/unique.fasta                  U distinct haplotypes (first occurrence kept)
    -> work/curvature_set/sequence_to_unique.npy        int32[N]: per record in
                                                        work/intermediate/ungapped.fasta, its haplotype
                                                        index in [0, U)
    -> work/curvature_set/unique_multiplicity.npy       int64[U]: number of records per haplotype
    -> work/curvature_set/collection_dates.tsv          N rows, aligned with
                                                        sequence_to_unique: 'epi_isl<TAB>date'

Choosing THETA.
Stage 2 always prints a profile of how many columns and sequences survive each threshold
in PROFILE_THETAS. With THETA = None it stops after the profile and writes nothing, so that
a threshold can be chosen that balances sites kept against sequences kept. A second run
with that THETA reuses the cached scans and completes Stages 2 and 3. The article uses
THETA = 0.0005. Each stage skips its work if its output exists (Stage 1: complete.fasta,
Stage 2: ungapped.fasta, Stage 3: unique_multiplicity.npy); deleting an output reruns its
stage, e.g. ungapped.fasta after changing THETA.
"""
import os
import sys
import time

import hammingdist
import numpy as np

from sars_utils import (
    DATES_TSV,
    INTERMEDIATE_DIR,
    MULTIPLICITY,
    SEQ_TO_UNIQUE,
    UNIQUE_FASTA,
    WORK_DIR,
    epi_isl,
    iter_records,
    parse_date,
)

# --- config -----------------------------------------------------------------
# Input: a fixed-width, reference-aligned nucleotide FASTA. The article uses GISAID's
# msa_0222.fasta (see the README for the download details). Set its path with the
# SARS_ALIGNMENT environment variable, or edit the fallback below:
#
#     SARS_ALIGNMENT=/path/to/msa_0222.fasta uv run python manuscript/sars_cov2/step0_produce_alignment.py
#
GIANT_ALIGNMENT = os.environ.get(
    'SARS_ALIGNMENT',
    os.path.join(WORK_DIR, 'msa_0222.fasta'),
)
REPORT_EVERY = 100_000          # progress line every N sequences

# Stage 1: coverage and date filter
ALLOWED = b'ACGTacgt-'                                    # any other character -> sequence dropped
COMPLETE = os.path.join(INTERMEDIATE_DIR, 'complete.fasta')   # output

# Stage 2: gap removal
DASH = ord('-')                                          # gap character
# Keep a column iff at most this fraction of sequences are dashed there.
#   None -> print the PROFILE_THETAS profile and stop before Stage 3; set a value to write (see docstring).
THETA = 0.0005
PROFILE_THETAS = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01)  # thresholds shown in the profile
OUTPUT = os.path.join(INTERMEDIATE_DIR, 'ungapped.fasta')            # output
DASH_COUNTS = os.path.join(INTERMEDIATE_DIR, 'dash_column_counts.npz')     # cache: per-column dash counts
ROW_MIN_FRAC = os.path.join(INTERMEDIATE_DIR, 'row_min_dash_fraction.npy')  # cache: per-row min dash fraction

# Stage 3: deduplication and measure
MIN_MULTIPLICITY = 1            # 1 = keep every distinct haplotype; raise to drop rare ones (denoising)
# outputs (UNIQUE_FASTA, SEQ_TO_UNIQUE, MULTIPLICITY, DATES_TSV) are the shared artifacts from sars_utils
# -----------------------------------------------------------------------------


# --- stage 1 : drop low-coverage / undated sequences --------------------------

def filter_complete_sequences(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total = kept = dropped_char = dropped_date = 0
    t0 = time.time()
    with open(output_path, 'wb') as fout:
        for header, seq in iter_records(input_path):
            total += 1
            # base-clean iff removing every ALLOWED byte leaves nothing
            base_ok = len(seq.translate(None, ALLOWED)) == 0
            if not base_ok:
                dropped_char += 1
            elif parse_date(header) is None:
                dropped_date += 1
            else:
                fout.write(header)
                fout.write(seq)
                fout.write(b'\n')
                kept += 1
            if total % REPORT_EVERY == 0:
                dt = time.time() - t0
                print(f'  stage1 {total:>12,} read | {kept:>12,} kept '
                      f'| {total/dt:,.0f} seq/s | {dt/60:.1f} min',
                      file=sys.stderr, flush=True)
    print(f'stage 1  sequences read       : {total:,}', file=sys.stderr)
    print(f'stage 1  sequences kept       : {kept:,}'
          + (f'  ({100*kept/total:.2f}%)' if total else ''), file=sys.stderr)
    print(f'stage 1  dropped (bad char)   : {dropped_char:,}', file=sys.stderr)
    print(f'stage 1  dropped (no date)    : {dropped_date:,}', file=sys.stderr)
    return total, kept


# --- stage 2 : remove gaps by column threshold + residual row drop ------------

def count_dashes_per_column(path):
    """Pass 1: per-column dash count and the number of sequences."""
    counts = None
    L = None
    n = 0
    t0 = time.time()
    for _, seq in iter_records(path):
        arr = np.frombuffer(seq, dtype=np.uint8)
        if counts is None:
            L = arr.size
            counts = np.zeros(L, dtype=np.int64)
        elif arr.size != L:
            raise ValueError(f'record {n} has length {arr.size}, expected {L} '
                             '(input is not a fixed-width alignment)')
        counts += (arr == DASH)
        n += 1
        if n % REPORT_EVERY == 0:
            print(f'  stage2 count {n:>12,} scanned | {(time.time()-t0)/60:.1f} min',
                  file=sys.stderr, flush=True)
    if counts is None:
        counts = np.zeros(0, dtype=np.int64)
    return counts, n


def min_dash_frac_per_row(path, frac, n_rows):
    """Pass 2: per-row minimum dash-fraction over the columns it is dashed in.

    A sequence survives threshold theta iff every column it is dashed in is dropped
    (dash-fraction > theta), i.e. iff this minimum exceeds theta. Rows with no dash
    get +inf and always survive.
    """
    s = np.empty(n_rows, dtype=np.float32)
    t0 = time.time()
    for i, (_, seq) in enumerate(iter_records(path)):
        arr = np.frombuffer(seq, dtype=np.uint8)
        dashed = arr == DASH
        s[i] = frac[dashed].min() if dashed.any() else np.inf
        if (i + 1) % REPORT_EVERY == 0:
            print(f'  stage2 rowscan {i+1:>12,} scanned | {(time.time()-t0)/60:.1f} min',
                  file=sys.stderr, flush=True)
    return s


def load_or_count(path):
    if os.path.exists(DASH_COUNTS):
        with np.load(DASH_COUNTS) as d:
            print(f'stage 2  using cached column counts ({DASH_COUNTS})', file=sys.stderr)
            return d['counts'], int(d['n_rows'])
    counts, n = count_dashes_per_column(path)
    os.makedirs(os.path.dirname(DASH_COUNTS), exist_ok=True)
    np.savez(DASH_COUNTS, counts=counts, n_rows=n)
    return counts, n


def load_or_min_frac(path, frac, n_rows):
    if os.path.exists(ROW_MIN_FRAC):
        s = np.load(ROW_MIN_FRAC)
        if s.shape[0] == n_rows:
            print(f'stage 2  using cached row thresholds ({ROW_MIN_FRAC})', file=sys.stderr)
            return s
    s = min_dash_frac_per_row(path, frac, n_rows)
    os.makedirs(os.path.dirname(ROW_MIN_FRAC), exist_ok=True)
    np.save(ROW_MIN_FRAC, s)
    return s


def report_gap_profile(counts, n_rows, s_row, thetas):
    """Print, per threshold, how many columns and sequences survive."""
    L = counts.size
    frac = counts.astype(np.float64) / n_rows
    print(f'stage 2  columns total        : {L:,}', file=sys.stderr)
    print(f'stage 2  complete rows        : {n_rows:,}', file=sys.stderr)
    print(f'stage 2  dash-free columns     : {int((counts == 0).sum()):,}', file=sys.stderr)
    print('stage 2  threshold |   columns kept   |   sequences kept', file=sys.stderr)
    for th in thetas:
        cols = int((frac <= th).sum())
        rows = int((s_row > th).sum())
        print(f'stage 2  {th:8.3%} | {cols:>9,} ({100*cols/L:5.1f}%) '
              f'| {rows:>12,} ({100*rows/n_rows:5.1f}%)', file=sys.stderr)


def write_ungapped(path, output_path, keep_idx):
    """Pass 3: keep only keep_idx columns; drop rows still carrying a dash there."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = dropped = 0
    t0 = time.time()
    with open(output_path, 'wb') as fout:
        for header, seq in iter_records(path):
            sub = np.frombuffer(seq, dtype=np.uint8)[keep_idx]
            if (sub == DASH).any():
                dropped += 1
                continue
            fout.write(header)
            fout.write(sub.tobytes())
            fout.write(b'\n')
            n += 1
            if (n + dropped) % REPORT_EVERY == 0:
                print(f'  stage2 write {n+dropped:>12,} read | {n:>12,} written '
                      f'| {(time.time()-t0)/60:.1f} min', file=sys.stderr, flush=True)
    return n, dropped


def build_ungapped_alignment(input_path=COMPLETE, output_path=OUTPUT,
                             theta=THETA):
    counts, n_rows = load_or_count(input_path)
    if n_rows == 0:
        print('stage 2  empty input', file=sys.stderr)
        return 0, 0
    frac = counts.astype(np.float64) / n_rows
    s_row = load_or_min_frac(input_path, frac, n_rows)
    report_gap_profile(counts, n_rows, s_row, PROFILE_THETAS)

    if theta is None:
        print('stage 2  THETA is None -> profile only; set it to write.',
              file=sys.stderr)
        return 0, 0

    if os.path.exists(output_path):
        print(f'stage 2  {output_path} exists -> skipping write pass '
              '(delete it to rewrite, e.g. after changing THETA).', file=sys.stderr)
        return 0, 0

    keep_idx = np.nonzero(frac <= theta)[0]
    print(f'stage 2  chosen threshold      : {theta:.3%}', file=sys.stderr)
    print(f'stage 2  columns kept          : {keep_idx.size:,} of {counts.size:,}',
          file=sys.stderr)
    n_written, dropped = write_ungapped(input_path, output_path, keep_idx)
    print(f'stage 2  sequences written     : {n_written:,}  '
          f'(each now {keep_idx.size:,} sites)', file=sys.stderr)
    print(f'stage 2  sequences dropped (gap): {dropped:,}', file=sys.stderr)
    return n_written, keep_idx.size


# --- stage 3 : dedup + drop rare haplotypes + build the measure ---------------

def filter_min_multiplicity(input_path=OUTPUT,
                            unique_fasta=UNIQUE_FASTA, seq_to_unique=SEQ_TO_UNIQUE,
                            multiplicity=MULTIPLICITY, dates_tsv=DATES_TSV,
                            min_count=MIN_MULTIPLICITY):
    for p in (unique_fasta, seq_to_unique, dates_tsv):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    print('stage 3  deduplicating (hammingdist.fasta_sequence_indices)...', file=sys.stderr)
    t0 = time.time()
    reps_raw = hammingdist.fasta_sequence_indices(str(input_path))   # (N,) uint64, raw uid
    n = int(reps_raw.size)
    U_raw = int(reps_raw.max()) + 1 if n else 0
    counts_raw = np.bincount(reps_raw, minlength=U_raw)
    keep_raw = counts_raw >= min_count
    # Compact index among the kept haplotypes; dropping whole haplotypes never reorders the
    # survivors, so this is their order of first occurrence in the stage-2 output.
    remap = (np.cumsum(keep_raw) - 1).astype(np.int64)
    U = int(keep_raw.sum())
    print(f'  {n:,} records | {U_raw:,} raw unique | {(time.time()-t0)/60:.1f} min',
          file=sys.stderr)

    print(f'stage 3  writing unique fasta + measure (>= {min_count} copies)...',
          file=sys.stderr)
    t0 = time.time()
    seen = np.zeros(U, dtype=bool)
    final_uid = np.empty(n, dtype=np.int64)
    n_kept = 0
    with open(unique_fasta, 'wb') as fout_u, open(dates_tsv, 'w') as fdate:
        fdate.write('epi_isl\tdate\n')
        for i, (header, seq) in enumerate(iter_records(input_path)):
            raw_uid = reps_raw[i]
            if not keep_raw[raw_uid]:
                continue
            uid = remap[raw_uid]
            final_uid[n_kept] = uid
            n_kept += 1

            if not seen[uid]:
                seen[uid] = True
                fout_u.write(header)
                fout_u.write(seq)
                fout_u.write(b'\n')

            d = parse_date(header)
            fdate.write(f'{epi_isl(header)}\t{d.isoformat() if d else ""}\n')

            if (i + 1) % REPORT_EVERY == 0:
                print(f'  {i+1:>12,} read | {n_kept:>12,} kept '
                      f'| {(time.time()-t0)/60:.1f} min', file=sys.stderr, flush=True)

    # Persist the index/count arrays only once the unique fasta and dates tsv have finished
    # streaming, so a partial run never leaves the measure looking "done".
    np.save(seq_to_unique, final_uid[:n_kept].astype(np.int32))
    np.save(multiplicity, counts_raw[keep_raw].astype(np.int64))

    print(f'stage 3  records  (N)          : {n:,} -> {n_kept:,}'
          + (f'  ({100*n_kept/n:.2f}% kept)' if n else ''), file=sys.stderr)
    print(f'stage 3  unique   (U)          : {U_raw:,} -> {U:,}'
          + (f'  ({100*U/U_raw:.2f}% kept)' if U_raw else ''), file=sys.stderr)
    print(f'stage 3  elapsed               : {(time.time()-t0)/60:.1f} min', file=sys.stderr)
    return n_kept, U


def produce_alignment(giant=GIANT_ALIGNMENT, complete=COMPLETE, output=OUTPUT,
                      theta=THETA,
                      unique_fasta=UNIQUE_FASTA, multiplicity=MULTIPLICITY,
                      min_count=MIN_MULTIPLICITY):
    if os.path.exists(complete):
        print(f'stage 1  {complete} exists -> skipping stage 1', file=sys.stderr)
    else:
        print(f'Input : {giant}', file=sys.stderr)
        filter_complete_sequences(giant, complete)
        print(f'-> {complete}', file=sys.stderr)
    build_ungapped_alignment(complete, output, theta)

    if not os.path.exists(output):
        print(f'stage 2  {output} not written yet (theta unset) -> stopping before stage 3',
              file=sys.stderr)
        return
    print(f'-> {output}', file=sys.stderr)

    if os.path.exists(multiplicity):
        print(f'stage 3  {multiplicity} exists -> skipping stage 3', file=sys.stderr)
    else:
        filter_min_multiplicity(output, min_count=min_count)
    print(f'-> {unique_fasta}', file=sys.stderr)


if __name__ == '__main__':
    produce_alignment()
