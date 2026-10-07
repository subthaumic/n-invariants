"""The SARS-CoV-2 pipeline of Section 8.4, run end to end on a synthetic alignment.

The GISAID data behind the article cannot be redistributed, so this test draws a small
stand-in from the Jukes-Cantor model instead: a few hundred aligned sequences in GISAID's
header format, dated so that every date slice of step 2 is populated. The pipeline's code
is copied into a temporary directory and the copies run unchanged, so all their output
lands there and the precomputed results are never touched. The only change is a coarser
accuracy in step 2, which would otherwise draw the article's ~3.4e9 quadruples per slice.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from models import jc_single_site_distribution, sample_sequences

pytest.importorskip("hammingdist", reason="step 0 needs the `sars` extra")
pytest.importorskip("sklearn")

PIPELINE = Path(__file__).resolve().parents[1] / "sars_cov2"

# One synthetic population per collection date, each drawn from the JC model at a later
# time. Every date falls into a different slice of step 2 (cutoffs 2020-06-30, 2021-04-30,
# 2021-10-31, 2022-03-01, 2024-03-01), so the slices grow by one population each.
LAYERS = [
    ("2020-04-15", 0.01),
    ("2021-02-01", 0.02),
    ("2021-08-01", 0.04),
    ("2022-01-15", 0.08),
    ("2023-06-01", 0.16),
]
SEQUENCES_PER_LAYER = 40
LENGTH = 300
BASES = np.frombuffer(b"ACGT", dtype=np.uint8)


def write_alignment(path: Path) -> int:
    """Write the synthetic alignment and return the number of records step 0 should keep."""
    rng = np.random.default_rng(0)
    records = []
    for date, time in LAYERS:
        rho = jc_single_site_distribution(time, alphabet_size=4, beta=1.0)
        for sequence in sample_sequences(rho, LENGTH, SEQUENCES_PER_LAYER, rng):
            i = len(records)
            header = f"hCoV-19/synthetic/{i}/{date[:4]}|EPI_SYN_{i:06d}|{date}|synthetic"
            records.append((header, BASES[sequence].tobytes().decode()))
    kept = len(records)
    # Two records that step 0 must drop: an ambiguous base, and an incomplete date.
    records.append(("hCoV-19/synthetic/ambiguous/2021|EPI_SYN_900001|2021-03-01|synthetic",
                    "N" + records[0][1][1:]))
    records.append(("hCoV-19/synthetic/undated/2021|EPI_SYN_900002|2021-03|synthetic",
                    records[1][1]))
    path.write_text("".join(f">{header}\n{sequence}\n" for header, sequence in records))
    return kept


def run(code_dir: Path, args: list[str], alignment: Path) -> None:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=code_dir,
        env={**os.environ, "SARS_ALIGNMENT": str(alignment)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pipeline_runs_end_to_end_on_a_synthetic_alignment(tmp_path):
    code_dir = tmp_path / "sars_cov2"
    shutil.copytree(PIPELINE, code_dir, ignore=shutil.ignore_patterns(
        "results", "work", "__pycache__", "*.md"))
    alignment = tmp_path / "synthetic_alignment.fasta"
    n_records = write_alignment(alignment)

    run(code_dir, ["step0_produce_alignment.py"], alignment)
    run(code_dir, ["step1_build_haplotype_array.py"], alignment)
    run(code_dir, ["-c", (
        "import step2_compute_invariants as step2\n"
        "step2.epsilon, step2.epsilon_dist, step2.n_workers = 1.0, 0.1, 1\n"
        "step2.compute_invariants()\n"
    )], alignment)
    run(code_dir, ["step3_compute_mds.py"], alignment)

    # Step 0: the two malformed records are gone, the rest are deduplicated with their
    # multiplicities. Step 1: the point set keeps every polymorphic column.
    curvature_set = code_dir / "work" / "curvature_set"
    multiplicity = np.load(curvature_set / "unique_multiplicity.npy")
    assert multiplicity.sum() == n_records
    points = np.load(curvature_set / "unique.npy")
    meta = np.load(curvature_set / "unique_meta.npz")
    assert points.shape == (multiplicity.size, int(meta["L_poly"]))
    assert int(meta["L"]) == LENGTH

    # Step 2: one file per cutoff, each slice holding the populations dated before it, and
    # each histogram holding exactly the number of samples the bound asked for.
    invariant_files = sorted((code_dir / "results" / "invariants").glob("invariants_*.npz"))
    assert len(invariant_files) == len(LAYERS)
    for k, path in enumerate(invariant_files, start=1):
        stored = np.load(path, allow_pickle=True)
        assert int(stored["n_sequences"]) == k * SEQUENCES_PER_LAYER
        distances = stored["dm_dict"].item()
        deficits = stored["hyp_hist"].item()
        assert sum(distances.values()) == int(stored["n_samples_distance"])
        assert sum(deficits.values()) == int(stored["n_samples_hyperbolicity"])
        assert 0 <= min(distances) and max(distances) <= LENGTH
        assert min(deficits) >= 0

    # Step 3: fewer points than the subsample size, so every haplotype is embedded.
    embedding = np.load(code_dir / "results" / "mds" / "mds_embedding_max50000.npz")
    assert embedding["mds_embedding"].shape == (multiplicity.size, 2)
