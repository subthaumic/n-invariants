#!/usr/bin/env python3
"""Metric n-point invariants of the JC and HB substitution models (Section 8.3).

For both model families, at each time layer ``t = 0.05, 0.1, 0.5, 1`` and in the
stationary limit ``t = inf``, this computes

* the law of the pairwise distance, exactly, as ``Bin(L, 1 - c_2(rho_t))``;
* the law of the four-point hyperbolicity deficit, exactly, by FFT convolution of the
  one-site pair-sum law;
* the four-point persistence distribution, by Monte Carlo over site-pattern counts;
* a shared metric-MDS embedding of sequences sampled from every layer;

and writes them to ``results/invariants.npz``, which
``manuscript/figures/figure_6_substitution_models.py`` draws. The models and their exact
laws are in ``models.py`` (Appendix A).

The article's parameters are the defaults: alphabet size 4, sequence length ``L = 100``,
homogeneous product initial measure ``delta_0^{(x)L}``, JC total exit rate 1, and the HB
fitness profile ``phi = (0, 0, -10, -10)`` over the same symmetric JC baseline. The
persistence sample count ``N = 117,207,232`` comes from
``sample_size(epsilon=0.1, alpha=0.995, n=4, p=2)``; the union bound over the ten
model-time pairs turns the per-distribution confidence 0.995 into at least 0.95
simultaneously.

Everything random is seeded (``--seed``, default 20260707), so a rerun reproduces the
precomputed results. It takes about 40 minutes.

Run from ``manuscript/``::

    uv run python substitution_models/compute_invariants.py
"""

from __future__ import annotations

import argparse
import math
from inspect import signature
from pathlib import Path
from time import perf_counter

import numpy as np

from models import (
    binomial_pmf_with_coefficients,
    build_delta_tick_grid,
    distance_probability_from_site_distribution,
    hb_single_site_distribution,
    hb_stationary_distribution,
    hyperbolicity_pmf_from_rho,
    jc_distance_probability,
    jc_single_site_distribution,
    log_binomial_coefficients,
    sample_sequences,
)
from sampling import sample_distribution, single_site_pattern_law

try:
    from sklearn.manifold import MDS
except ImportError:  # pragma: no cover - handled at runtime.
    MDS = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "invariants.npz"
DEFAULT_LENGTH = 100
DEFAULT_ALPHABET_SIZE = 4
DEFAULT_BETA = 1.0
DEFAULT_HB_PHI = "0,0,-10,-10"
DEFAULT_GRID_TIMES = "0.05,0.1,0.5,1.0"
# sample_size(epsilon=0.1, alpha=0.995, n=4, p=2), where alpha=0.995
# gives at least 0.95 simultaneous confidence over the ten persistence panels
# by a union bound.  This is the production setting; use --persistence-samples
# with a smaller value for quick tests.
DEFAULT_PERSISTENCE_SAMPLES = 117_207_232
DEFAULT_BATCH_SIZE = 100_000
DEFAULT_MDS_SAMPLES_PER_LAYER = 160
DEFAULT_MDS_N_INIT = 1
DEFAULT_MDS_MAX_ITER = 500
DEFAULT_SEED = 20260707


def parse_float_list(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the JC/HB model data drawn in Figure 6."
    )
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH, help="Sequence length L.")
    parser.add_argument(
        "--alphabet-size",
        type=int,
        default=DEFAULT_ALPHABET_SIZE,
        help="Alphabet size k.",
    )
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="JC exit rate.")
    parser.add_argument(
        "--times",
        default=DEFAULT_GRID_TIMES,
        help="Comma-separated finite times.",
    )
    parser.add_argument(
        "--hb-phi",
        default=DEFAULT_HB_PHI,
        help="Comma-separated HB fitness values.",
    )
    parser.add_argument(
        "--persistence-samples",
        type=int,
        default=DEFAULT_PERSISTENCE_SAMPLES,
        help=(
            "Monte Carlo quadruples per model/time law for persistence "
            "(production default: 117,207,232 for relative accuracy 0.1 and at least "
            "0.95 simultaneous confidence across the ten default panels)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Persistence sampling batch size.",
    )
    parser.add_argument(
        "--mds-samples-per-layer",
        type=int,
        default=DEFAULT_MDS_SAMPLES_PER_LAYER,
        help="Sampled sequences per model/time layer for MDS.",
    )
    parser.add_argument(
        "--mds-n-init",
        type=int,
        default=DEFAULT_MDS_N_INIT,
        help="Random metric-MDS initializations.",
    )
    parser.add_argument(
        "--mds-max-iter",
        type=int,
        default=DEFAULT_MDS_MAX_ITER,
        help="Maximum metric-MDS SMACOF iterations.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base RNG seed.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output .npz (default: results/invariants.npz next to this script).",
    )
    return parser.parse_args()


def layer_label(time: float) -> str:
    if math.isinf(time):
        return r"$t=\infty$"
    return rf"$t={time:g}$"


def model_layers(
    alphabet_size: int,
    beta: float,
    times: list[float],
    hb_phi: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
    uniform_rho = np.full(alphabet_size, 1.0 / alphabet_size, dtype=float)
    jc_layers = [
        {
            "family": "JC",
            "time": time,
            "rho": jc_single_site_distribution(time, alphabet_size, beta),
        }
        for time in times
    ]
    jc_layers.append({"family": "JC limit", "time": math.inf, "rho": uniform_rho})

    hb_layers = [
        {
            "family": "HB",
            "time": time,
            "rho": hb_single_site_distribution(time, alphabet_size, beta, hb_phi),
        }
        for time in times
    ]
    hb_layers.append(
        {"family": "HB limit", "time": math.inf, "rho": hb_stationary_distribution(hb_phi)}
    )
    return {"JC": jc_layers, "HB": hb_layers}


def pairwise_hamming(sequences: np.ndarray) -> np.ndarray:
    distances = np.sum(sequences[:, None, :] != sequences[None, :, :], axis=2, dtype=np.float64)
    return distances


def metric_mds(
    distances: np.ndarray,
    *,
    seed: int,
    n_init: int,
    max_iter: int,
) -> np.ndarray:
    if MDS is None:
        raise RuntimeError(
            "Missing dependency 'scikit-learn'. Install it before running this script."
        )

    parameters = signature(MDS).parameters
    kwargs = {
        "n_components": 2,
        "n_init": n_init,
        "max_iter": max_iter,
        "random_state": seed,
        "eps": 1.0e-6,
    }
    if "init" in parameters:
        kwargs["init"] = "random"
    if "n_jobs" in parameters:
        kwargs["n_jobs"] = 1
    if "metric_mds" in parameters:
        kwargs["metric_mds"] = True
        kwargs["metric"] = "precomputed"
    else:
        if "metric" in parameters:
            kwargs["metric"] = True
        if "dissimilarity" in parameters:
            kwargs["dissimilarity"] = "precomputed"
        else:
            raise RuntimeError("Installed scikit-learn MDS cannot use precomputed distances.")

    return MDS(**kwargs).fit_transform(distances)


def mds_data_for_layers(
    layers: list[dict[str, object]],
    length: int,
    samples_per_layer: int,
    seed: int,
    mds_n_init: int,
    mds_max_iter: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sequences = []
    layer_index = []
    for index, layer in enumerate(layers):
        rho = layer["rho"]
        assert isinstance(rho, np.ndarray)
        seqs = sample_sequences(rho, length, samples_per_layer, rng)
        sequences.append(seqs)
        layer_index.extend([index] * samples_per_layer)
    all_sequences = np.vstack(sequences)
    coords = metric_mds(
        pairwise_hamming(all_sequences),
        seed=seed,
        n_init=mds_n_init,
        max_iter=mds_max_iter,
    )
    return {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "layer_index": np.asarray(layer_index, dtype=np.int16),
    }


def distance_pmfs(
    layers: list[dict[str, object]],
    model: str,
    length: int,
    alphabet_size: int,
    beta: float,
) -> list[np.ndarray]:
    log_binom = log_binomial_coefficients(length)
    pmfs = []
    for layer in layers:
        time = float(layer["time"])
        if model == "JC" and not math.isinf(time):
            p = jc_distance_probability(time, alphabet_size, beta)
        else:
            rho = layer["rho"]
            assert isinstance(rho, np.ndarray)
            p = distance_probability_from_site_distribution(rho)
        pmfs.append(binomial_pmf_with_coefficients(length, p, log_binom))
    return pmfs


def hyperbolicity_pmfs(
    layers: list[dict[str, object]],
    length: int,
) -> list[np.ndarray]:
    delta_grid = build_delta_tick_grid(length)
    pmfs = []
    for layer in layers:
        rho = layer["rho"]
        assert isinstance(rho, np.ndarray)
        pmfs.append(hyperbolicity_pmf_from_rho(length, rho, delta_grid))
    return pmfs


def persistence_results(
    layers: list[dict[str, object]],
    model: str,
    length: int,
    samples: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, np.ndarray | int | float]]:
    results = []
    for index, layer in enumerate(layers):
        label = layer_label(float(layer["time"]))
        print(
            f"  {model} {label} ({index + 1}/{len(layers)}): "
            f"{samples:,} quadruples...",
            flush=True,
        )
        started = perf_counter()
        rho = layer["rho"]
        assert isinstance(rho, np.ndarray)
        patterns, probabilities = single_site_pattern_law(rho)
        birth, lifetime, count, zero_count = sample_distribution(
            length=length,
            patterns=patterns,
            probabilities=probabilities,
            samples=samples,
            batch_size=batch_size,
            seed=seed + index * 10_000,
        )
        results.append(
            {
                "birth": birth,
                "lifetime": lifetime,
                "count": count,
                "probability": count.astype(float) / float(samples),
                "zero_count": zero_count,
                "nonzero_mass": float(np.sum(count) / samples),
            }
        )
        elapsed = perf_counter() - started
        print(f"    completed in {elapsed / 60.0:.1f} min", flush=True)
    return results


def compute_all_data(
    args: argparse.Namespace, layers_by_model: dict
) -> tuple[dict, dict, dict, dict]:
    print("Computing sampled metric-MDS panels...")
    mds_by_model = {
        model: mds_data_for_layers(
            layers, length=args.length, samples_per_layer=args.mds_samples_per_layer,
            seed=args.seed + (0 if model == "JC" else 100_000),
            mds_n_init=args.mds_n_init, mds_max_iter=args.mds_max_iter,
        )
        for model, layers in layers_by_model.items()
    }
    print("Computing exact distance and hyperbolicity laws...")
    distance_by_model = {
        model: distance_pmfs(layers, model, args.length, args.alphabet_size, args.beta)
        for model, layers in layers_by_model.items()
    }
    hyperbolicity_by_model = {
        model: hyperbolicity_pmfs(layers, args.length)
        for model, layers in layers_by_model.items()
    }
    print("Sampling persistence laws...")
    persistence_by_model = {
        model: persistence_results(
            layers, model=model, length=args.length, samples=args.persistence_samples,
            batch_size=args.batch_size, seed=args.seed + (200_000 if model == "JC" else 300_000),
        )
        for model, layers in layers_by_model.items()
    }
    return mds_by_model, distance_by_model, hyperbolicity_by_model, persistence_by_model


def save_results(path: Path, params: dict, data: tuple[dict, dict, dict, dict]) -> None:
    """Write the parameters and the four data dicts to one .npz, read by Figure 6."""
    mds_by_model, distance_by_model, hyperbolicity_by_model, persistence_by_model = data
    payload = {"models": np.asarray(list(mds_by_model.keys()))}
    payload.update(params)
    for model in mds_by_model:
        payload[f"{model}_mds_x"] = mds_by_model[model]["x"]
        payload[f"{model}_mds_y"] = mds_by_model[model]["y"]
        payload[f"{model}_mds_layer_index"] = mds_by_model[model]["layer_index"]
        payload[f"{model}_distance"] = np.asarray(distance_by_model[model])
        payload[f"{model}_hyperbolicity"] = np.asarray(hyperbolicity_by_model[model])
        payload[f"{model}_n_layers"] = np.asarray(len(persistence_by_model[model]))
        for i, result in enumerate(persistence_by_model[model]):
            payload[f"{model}_pers_{i}_birth"] = result["birth"]
            payload[f"{model}_pers_{i}_lifetime"] = result["lifetime"]
            payload[f"{model}_pers_{i}_count"] = result["count"]
            payload[f"{model}_pers_{i}_zero"] = np.asarray(result["zero_count"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def main() -> None:
    args = parse_args()
    if args.length <= 0:
        raise ValueError("length must be positive.")
    if args.alphabet_size < 2:
        raise ValueError("alphabet-size must be at least 2.")
    if args.persistence_samples <= 0:
        raise ValueError("persistence-samples must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.mds_samples_per_layer <= 0:
        raise ValueError("mds-samples-per-layer must be positive.")
    if args.mds_n_init <= 0:
        raise ValueError("mds-n-init must be positive.")
    if args.mds_max_iter <= 0:
        raise ValueError("mds-max-iter must be positive.")

    times = parse_float_list(args.times)
    if any(time <= 0 for time in times):
        raise ValueError("times must be positive.")
    hb_phi = np.asarray(parse_float_list(args.hb_phi), dtype=float)
    if hb_phi.shape != (args.alphabet_size,):
        raise ValueError(
            f"--hb-phi must contain exactly {args.alphabet_size} values, got {hb_phi.size}."
        )

    layers_by_model = model_layers(args.alphabet_size, args.beta, times, hb_phi)
    data = compute_all_data(args, layers_by_model)

    for model, results in data[3].items():
        for layer, result in zip(layers_by_model[model], results, strict=True):
            print(
                f"{model} {layer_label(float(layer['time']))}: "
                f"persistence nonzero mass {float(result['nonzero_mass']):.6g}"
            )

    params = {
        "length": np.asarray(args.length),
        "alphabet_size": np.asarray(args.alphabet_size),
        "beta": np.asarray(args.beta, dtype=float),
        "times": np.asarray(times, dtype=float),
        "hb_phi": np.asarray(hb_phi, dtype=float),
        "persistence_samples": np.asarray(args.persistence_samples),
        "batch_size": np.asarray(args.batch_size),
        "mds_samples_per_layer": np.asarray(args.mds_samples_per_layer),
        "mds_n_init": np.asarray(args.mds_n_init),
        "mds_max_iter": np.asarray(args.mds_max_iter),
        "seed": np.asarray(args.seed),
    }
    output = Path(args.output)
    save_results(output, params, data)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
