# SARS-CoV-2 pipeline

The code behind Section 8.4 and Appendix B. It estimates the distributions of the pairwise
distance, the four-point hyperbolicity deficit and the $(4,1)$-persistence of the distinct
haplotypes in an aligned GISAID FASTA, at five cumulative collection-date cutoffs.

## Data

The sequences are available from [GISAID](https://gisaid.org) after registration and may
not be redistributed. The article uses:

- the EpiCoV alignment `msa_0222.fasta`, downloaded on 24 February 2024;
- aligned to hCoV-19/Wuhan/WIV04/2019 (`EPI_ISL_402124`) with MAFFT v7.497;
- the records `EPI_SET_261006ze`, <https://doi.org/10.55876/gis8.261006ze>.

## Results

The precomputed results contain no sequences:

```
results/invariants/invariants_<cutoff>.npz   histograms of the three invariants, one file per cutoff
results/mds/mds_embedding_max50000.npz       metric-MDS coordinates and first-seen dates
```

Figure 7 and the statistics of Section 8.4 are computed from these files.

## Pipeline

```bash
uv sync --extra sars
export SARS_ALIGNMENT=/path/to/msa_0222.fasta
uv run python manuscript/sars_cov2/step0_produce_alignment.py
uv run python manuscript/sars_cov2/step1_build_haplotype_array.py
uv run python manuscript/sars_cov2/step2_compute_invariants.py
uv run python manuscript/sars_cov2/step3_compute_mds.py
```

| Step | Description | Output |
|---|---|---|
| 0 | filters by base and collection date, removes gap columns, deduplicates into haplotypes with multiplicities and dates | `work/curvature_set/` |
| 1 | stores the haplotypes as a `uint8` array without constant columns | `work/curvature_set/` |
| 2 | samples the three invariants at each cutoff, with sample sizes from `n_invariants.sample_size` (Cor. 6.5) | `results/invariants/` |
| 3 | embeds a subsample of 50,000 haplotypes by metric MDS, for Figure 7 | `results/mds/` |

- Settings are constants at the top of each script, set to the values used in the article.
- `sars_utils.py` holds the file layout and the FASTA helpers; `sampling.py` is the sampler
  of step 2.
- Intermediate files go to `work/`, which is not committed.
- Steps 2 and 3 write to `results/` and replace the precomputed files (step 3 only with
  `OVERWRITE = True`).
- The article's runs used 52 CPUs and about 700 GiB of memory; step 3 took about 27 hours.

## Test

Runs all four steps on a small synthetic alignment drawn from the Jukes–Cantor model, in a
temporary copy of this directory:

```bash
uv run pytest manuscript/tests/test_sars_pipeline.py
```
