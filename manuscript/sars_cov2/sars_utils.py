"""File layout and FASTA helpers shared by the SARS-CoV-2 pipeline.

The pipeline turns an aligned FASTA of SARS-CoV-2 genomes into a metric measure space,
with the distinct sequences (haplotypes) as points, the Hamming distance as metric and
the observed multiplicities as measure, and estimates distributions of metric n-point
invariants on it by sampling.

The numbered scripts (``step0_`` to ``step3_``) run in that order and communicate only
through the files listed below; none imports another. This module holds what several of
them need: the file paths, and the routines for reading the FASTA and parsing its headers.
Step 2 samples with ``sampling.py``, which evaluates the invariants of
``n_invariants.invariants``.
"""
import os
from datetime import date, timedelta

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))

# --- data layout -------------------------------------------------------------
# The scripts read and write exclusively through these paths, so they agree on the layout.
# work/ holds everything derived from the GISAID alignment and is never committed:
# work/intermediate/ holds step 0's intermediate alignment products; work/curvature_set/ holds the
# metric space X itself: its points, its measure, and the arrays derived from them.
WORK_DIR = os.path.join(_here, 'work')
INTERMEDIATE_DIR = os.path.join(WORK_DIR, 'intermediate')
CURVATURE_SET_DIR = os.path.join(WORK_DIR, 'curvature_set')

# The points of X: the distinct haplotypes, one FASTA record each (written by step 0, read by step 1).
UNIQUE_FASTA = os.path.join(CURVATURE_SET_DIR, 'unique.fasta')

# The measure and bookkeeping over the U haplotypes (written by step 0).
# int32[N]: record -> haplotype index
SEQ_TO_UNIQUE = os.path.join(CURVATURE_SET_DIR, 'sequence_to_unique.npy')
# int64[U]: number of records per haplotype
MULTIPLICITY = os.path.join(CURVATURE_SET_DIR, 'unique_multiplicity.npy')
# N rows, row-aligned: 'epi_isl<TAB>date'
DATES_TSV = os.path.join(CURVATURE_SET_DIR, 'collection_dates.tsv')

# The point set used for on-the-fly distances (written by step 1, read by steps 2 and 3).
# uint8[U, L_poly]: haplotypes, polymorphic columns only
HAPLOTYPES = os.path.join(CURVATURE_SET_DIR, 'unique.npy')
# int32[L_poly]: kept column indices in [0, L)
POLY_COLUMNS = os.path.join(CURVATURE_SET_DIR, 'polymorphic_columns.npy')
HAPLOTYPE_META = os.path.join(CURVATURE_SET_DIR, 'unique_meta.npz') # U, L, L_poly

# --- FASTA headers -----------------------------------------------------------
# GISAID headers look like:  >hCoV-19/name/year|EPI_ISL_id|YYYY-MM-DD|region
ID_FIELD = 1     # 0-based '|'-separated field holding the EPI_ISL accession
DATE_FIELD = 2   # 0-based '|'-separated field holding the collection date


def iter_records(path):
    """Yield ``(header_bytes_with_newline, sequence_bytes)`` for each FASTA record, streaming.

    The file is read one record at a time, so memory use does not grow with the number of
    sequences. Sequence lines are concatenated with line
    breaks stripped, so ``sequence_bytes`` is the raw residue string of the record.
    """
    header = None
    buf = []
    with open(path, 'rb') as f:
        for line in f:
            if line[:1] == b'>':
                if header is not None:
                    yield header, b''.join(buf)
                header = line
                buf = []
            else:
                buf.append(line.rstrip(b'\r\n'))
        if header is not None:
            yield header, b''.join(buf)


def parse_date(header):
    """Collection date as a ``datetime.date``, or ``None`` if the header has no full YYYY-MM-DD.

    Incomplete dates ('2020', '2020-01') and impossible ones ('2020-01-00') return ``None``.
    """
    fields = header.rstrip().split(b'|')
    if len(fields) <= DATE_FIELD:
        return None
    d = fields[DATE_FIELD]
    if len(d) != 10 or d[4:5] != b'-' or d[7:8] != b'-':
        return None
    try:
        return date(int(d[0:4]), int(d[5:7]), int(d[8:10]))
    except ValueError:
        return None


def epi_isl(header):
    """EPI_ISL accession from a FASTA header, or '' if the header has no such field."""
    fields = header.rstrip().split(b'|')
    return fields[ID_FIELD].decode() if len(fields) > ID_FIELD else ''


def read_record_dates(path=DATES_TSV):
    """Per-record collection dates as a ``datetime64[D]`` array, row-aligned with SEQ_TO_UNIQUE.

    Empty date strings become NaT, which compares False against any cutoff (so undated
    records are naturally excluded from date slices).
    """
    dates = []
    with open(path) as fh:
        next(fh)  # skip the 'epi_isl\tdate' header line
        for line in fh:
            dates.append(line.rstrip('\n').split('\t')[1])
    return np.array(dates, dtype='datetime64[D]')


def read_record_accessions(path=DATES_TSV):
    """Per-record EPI_ISL accessions as a string array, row-aligned with SEQ_TO_UNIQUE.

    The same row order as ``read_record_dates``; use it to map a known accession (e.g. the
    reference genome) back to its haplotype index via SEQ_TO_UNIQUE.
    """
    accs = []
    with open(path) as fh:
        next(fh)  # skip the 'epi_isl\tdate' header line
        for line in fh:
            accs.append(line.rstrip('\n').split('\t')[0])
    return np.array(accs, dtype=object)


def display_time(seconds):
    """Format a duration in seconds as ``H:MM:SS.ss``."""
    if seconds == np.round(seconds):
        # timedelta drops the fractional part entirely for whole seconds, which would
        # leave the trailing slice below cutting into the seconds field.
        seconds += 0.001
    return str(timedelta(seconds=seconds))[:-4]
