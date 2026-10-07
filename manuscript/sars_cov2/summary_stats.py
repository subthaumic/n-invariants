#!/usr/bin/env python3
"""The SARS-CoV-2 statistics quoted in Section 8.4 of the article.

The text around Figure 7 quotes several numbers derived from its three distributions:
means, tail masses, the share of half-integer hyperbolicity deficits, and others. This
script recomputes all of them from the precomputed histograms that Figure 7 is drawn from.
It does no sampling and runs in about a second.

Run from the repository root::

    uv run python manuscript/sars_cov2/summary_stats.py

Definitions, as in the article:

* *variance-to-mean ratio*: of the sampled pairwise distance distribution.
* *distance tail mass*: the probability of a pairwise distance above ``TAIL_THRESHOLD``
  (200 substitutions), in parts per million.
* *half-integer share*: the mass of the hyperbolicity deficit distribution on
  half-integers. A quadruple has a half-integer deficit only when the largest and median
  opposite-side sums differ by an odd number, which requires a site with three or more
  alleles; the reference value for the uniform measure on the Hamming cube is 50%.
* *nonzero persistence mass*: the probability that a sampled quadruple has a nonzero
  degree-one class, in parts per million. The remaining mass is at the origin, reported
  by the sampler as ``(0, 0)``.
* *births*: of the nonzero classes only, summarized in three ways, since the article
  states their location approximately ("around 9", "around 84").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
INVARIANTS_DIR = SCRIPT_DIR / "results" / "invariants"

# The article's "thin tail of large distances ... below 30 ppm at all five cutoffs".
TAIL_THRESHOLD = 200


def histogram_arrays(entry) -> tuple[np.ndarray, np.ndarray]:
    """``{value: count}`` stored in an npz -> ``(values, counts)`` as float arrays."""
    mapping = entry.item() if isinstance(entry, np.ndarray) else entry
    keys = np.asarray(list(mapping.keys()), dtype=float)
    counts = np.asarray(list(mapping.values()), dtype=float)
    return keys, counts


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2.0)])


def slice_statistics(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)

    # --- pairwise distance ---------------------------------------------------
    dist_values, dist_counts = histogram_arrays(data["dm_dict"])
    dist_total = dist_counts.sum()
    dist_mean = float((dist_values * dist_counts).sum() / dist_total)
    dist_var = float((dist_counts * (dist_values - dist_mean) ** 2).sum() / dist_total)
    tail_ppm = float(dist_counts[dist_values > TAIL_THRESHOLD].sum() / dist_total * 1e6)

    # --- four-point hyperbolicity deficit ------------------------------------
    hyp_values, hyp_counts = histogram_arrays(data["hyp_hist"])
    hyp_total = hyp_counts.sum()
    p_zero = float(hyp_counts[hyp_values == 0].sum() / hyp_total)
    hyp_mean = float((hyp_values * hyp_counts).sum() / hyp_total)
    half_share = float(hyp_counts[(hyp_values % 1) != 0].sum() / hyp_total)

    # --- degree-one four-point persistence -----------------------------------
    pers_values, pers_counts = histogram_arrays(data["bd_dict"])
    pers_total = pers_counts.sum()
    trivial = (pers_values[:, 0] == 0) & (pers_values[:, 1] == 0)
    births = pers_values[~trivial, 0]
    lifetimes = pers_values[~trivial, 1] - pers_values[~trivial, 0]
    weights = pers_counts[~trivial]
    nonzero_ppm = float(weights.sum() / pers_total * 1e6)
    life_one_share = float(weights[lifetimes == 1].sum() / weights.sum())

    return {
        "cutoff": str(data["cutoff_date"]),
        "n_sequences": int(data["n_sequences"]),
        "n_haplotypes": int(data["n_haplotypes"]),
        "dist_mean": dist_mean,
        "dist_var_over_mean": dist_var / dist_mean,
        "dist_max": float(dist_values.max()),
        "dist_tail_ppm": tail_ppm,
        "hyp_p_zero": p_zero,
        "hyp_mean": hyp_mean,
        "hyp_max": float(hyp_values.max()),
        "hyp_half_share": half_share,
        "pers_nonzero_ppm": nonzero_ppm,
        "pers_life_one_share": life_one_share,
        "pers_life_max": float(lifetimes.max()),
        "pers_birth_mode": float(births[np.argmax(weights)]),
        "pers_birth_mean": float((births * weights).sum() / weights.sum()),
        "pers_birth_median": weighted_median(births, weights),
    }


def print_report(rows: list[dict]) -> None:
    first, last = rows[0], rows[-1]

    def table(title, columns):
        print(f"\n{title}")
        print("-" * len(title))
        head = f"{'cutoff':<12}" + "".join(f"{label:>{w}}" for label, w, _ in columns)
        print(head)
        for row in rows:
            print(f"{row['cutoff']:<12}" + "".join(f"{fmt(row):>{w}}" for _, w, fmt in columns))

    print("=" * 78)
    print("SARS-CoV-2 summary statistics  (Section 8.4)")
    print("=" * 78)
    print(f"\nRead from {INVARIANTS_DIR}")
    print(f"{len(rows)} date slices, cumulative: each is a reweighting to all haplotypes")
    print("seen on or before its cutoff.\n")

    print(f"{'cutoff':<12}{'sequences':>14}{'haplotypes':>13}")
    for row in rows:
        print(f"{row['cutoff']:<12}{row['n_sequences']:>14,}{row['n_haplotypes']:>13,}")

    table("Pairwise distance", [
        ("mean", 9, lambda r: f"{r['dist_mean']:.1f}"),
        ("var/mean", 10, lambda r: f"{r['dist_var_over_mean']:.1f}"),
        ("max", 7, lambda r: f"{r['dist_max']:.0f}"),
        (f"ppm >{TAIL_THRESHOLD}", 11, lambda r: f"{r['dist_tail_ppm']:.1f}"),
    ])
    print(f"\n  mean grows from {first['dist_mean']:.1f} to {last['dist_mean']:.1f} substitutions;")
    print(f"  variance-to-mean ratio from {first['dist_var_over_mean']:.1f} "
          f"to {last['dist_var_over_mean']:.1f};")
    print(f"  sampled maximum over all cutoffs {max(r['dist_max'] for r in rows):.0f}; "
          f"tail mass stays below {max(r['dist_tail_ppm'] for r in rows):.0f} ppm.")

    table("Four-point hyperbolicity deficit", [
        ("P(hyp=0)", 11, lambda r: f"{r['hyp_p_zero'] * 100:.1f}%"),
        ("mean", 9, lambda r: f"{r['hyp_mean']:.3f}"),
        ("max", 7, lambda r: f"{r['hyp_max']:.0f}"),
        ("half-int", 11, lambda r: f"{r['hyp_half_share'] * 100:.1f}%"),
    ])
    print(f"\n  P(hyp=0) falls from {first['hyp_p_zero'] * 100:.1f}% "
          f"to {last['hyp_p_zero'] * 100:.1f}%;")
    print(f"  mean deficit grows from {first['hyp_mean']:.3f} to {last['hyp_mean']:.3f}, "
          f"sampled maximum from {first['hyp_max']:.0f} to {last['hyp_max']:.0f},")
    print(f"  giving hyp(X) >= {last['hyp_max']:.0f} for the full dataset;")
    print(f"  half-integer mass grows from {first['hyp_half_share'] * 100:.1f}% "
          f"to {last['hyp_half_share'] * 100:.1f}%, against 50% for the uniform Hamming cube.")

    table("Degree-one four-point persistence", [
        ("ppm nonzero", 13, lambda r: f"{r['pers_nonzero_ppm']:.0f}"),
        ("lifetime=1", 12, lambda r: f"{r['pers_life_one_share'] * 100:.1f}%"),
        ("max life", 10, lambda r: f"{r['pers_life_max']:.0f}"),
        ("birth mode", 12, lambda r: f"{r['pers_birth_mode']:.0f}"),
        ("mean", 8, lambda r: f"{r['pers_birth_mean']:.1f}"),
        ("median", 8, lambda r: f"{r['pers_birth_median']:.0f}"),
    ])
    ppm = [r["pers_nonzero_ppm"] for r in rows]
    life1 = [r["pers_life_one_share"] * 100 for r in rows]
    print(f"\n  nonzero mass ranges from {min(ppm):.0f} to {max(ppm):.0f} ppm;")
    print(f"  the share with lifetime 1 ranges from {min(life1):.1f}% to {max(life1):.1f}%;")
    print(f"  sampled maximum lifetime {max(r['pers_life_max'] for r in rows):.0f};")
    print(f"  births drift upward from around {first['pers_birth_mode']:.0f} "
          f"to around {last['pers_birth_mode']:.0f}.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--invariants-dir",
        type=Path,
        default=INVARIANTS_DIR,
        help="Directory holding invariants_<cutoff>.npz (default: manuscript/sars_cov2/results/invariants).",
    )
    args = parser.parse_args()

    paths = sorted(args.invariants_dir.glob("invariants_*.npz"))
    if not paths:
        raise SystemExit(f"No invariants_*.npz found in {args.invariants_dir}")

    print_report([slice_statistics(path) for path in paths])


if __name__ == "__main__":
    main()
