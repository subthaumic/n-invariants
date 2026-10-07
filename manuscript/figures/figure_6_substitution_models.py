#!/usr/bin/env python3
"""Figure 6: metric n-point invariants of the JC and HB substitution models (Section 8.3).

Rows are the two model families, and within each row the time layers
``t = 0.05, 0.1, 0.5, 1`` and the stationary limit ``t = inf`` run from light to dark.
Columns are a shared metric-MDS embedding of sampled sequences, then the distributions of
the pairwise distance, the four-point hyperbolicity deficit and the (4,1)-persistence.

The script only draws. It reads the precomputed data
``manuscript/substitution_models/results/invariants.npz``, computed by
``manuscript/substitution_models/compute_invariants.py``.

Run from the repository root::

    uv run python manuscript/figures/figure_6_substitution_models.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    FormatStrFormatter,
    LogFormatterMathtext,
)

from plotting import (  # noqa: E402
    FAMILY_CUBEHELIX,
    HYPERBOLICITY_XMIN,
    PERSISTENCE_YTICKS,
    Y_FLOOR,
    color_luminance,
    dense_contour_grid,
    density_grid,
    family_palette,
    lighten_color,
    padded_persistence_ymax,
    plot_model_contours,
    plot_observed_persistence_support,
    shade_hamming_exclusion,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA = MANUSCRIPT_DIR / "substitution_models" / "results" / "invariants.npz"
DEFAULT_FIGURES_DIR = SCRIPT_DIR / "output"


def resolve_output_stem(output_stem: str, figures_dir: Path) -> Path:
    path = Path(output_stem)
    if path.is_absolute():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / path


def displayed_delta_max(results: list[dict[str, object]], length: int) -> float:
    max_visible = 0.0
    for result in results:
        pmf = result["pmf"]
        assert isinstance(pmf, np.ndarray)
        visible_ticks = np.flatnonzero(pmf >= Y_FLOOR)
        if visible_ticks.size:
            max_visible = max(max_visible, float(visible_ticks[-1]) / 2.0)
    return min(length / 2.0, max(5.0, 5.0 * math.ceil(max_visible / 5.0)))


DEFAULT_OUTPUT_STEM = "substitution_model_invariants"
DEFAULT_SMOOTHING_SIGMA = 1.0
DEFAULT_CONTOUR_RESOLUTION = 6
DEFAULT_PERSISTENCE_YMAX = None
PROBABILITY_YMAX = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw Figure 6 from the data computed by compute_invariants.py."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Data written by manuscript/substitution_models/compute_invariants.py.",
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=DEFAULT_SMOOTHING_SIGMA,
        help="Gaussian smoothing sigma for persistence contours.",
    )
    parser.add_argument(
        "--contour-resolution",
        type=int,
        default=DEFAULT_CONTOUR_RESOLUTION,
        help="Interpolated contour-grid intervals per lattice unit.",
    )
    parser.add_argument(
        "--persistence-ymax",
        type=float,
        default=DEFAULT_PERSISTENCE_YMAX,
        help="Displayed maximum persistence; defaults to the padded observed maximum.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(DEFAULT_FIGURES_DIR),
        help="Directory for relative output stems.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output stem; a .pdf is written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Resolution for any rasterized PDF artists.",
    )
    return parser.parse_args()


def layer_color(family: str, time: float, times: list[float]) -> tuple[float, float, float]:
    """Stage colour for a model layer, spaced evenly by ordinal stage (t=inf limit = darkest)."""
    n_stages = len(times) + 1  # the finite times plus the t=inf limit
    if family in ("JC limit", "HB limit"):
        model = "JC" if family == "JC limit" else "HB"
        index = n_stages - 1
    else:
        model = family
        index = sorted(times).index(time)
    return tuple(family_palette(n_stages, **FAMILY_CUBEHELIX[model])[index])


def layer_label(time: float) -> str:
    if math.isinf(time):
        return r"$t=\infty$"
    return rf"$t={time:g}$"


def probability_ylim(pmfs: list[np.ndarray]) -> tuple[float, float]:
    positive_parts = [pmf[pmf > 0.0] for pmf in pmfs if np.any(pmf > 0.0)]
    if not positive_parts:
        return (Y_FLOOR, PROBABILITY_YMAX)
    positive = np.concatenate(positive_parts)
    return (max(Y_FLOOR, float(np.min(positive)) * 0.7), PROBABILITY_YMAX)


def probability_major_ticks(ymin: float) -> list[float]:
    lower_power = math.floor(math.log10(max(ymin, np.nextafter(0.0, 1.0))))
    return [
        10.0**power
        for power in range(0, lower_power - 1, -3)
        if 10.0**power >= ymin
    ]


def format_probability_axis(ax: plt.Axes, ylim: tuple[float, float]) -> None:
    ymin = ylim[0]
    ax.set_yscale("log")
    ax.set_ylim(ymin, PROBABILITY_YMAX)
    ax.yaxis.set_major_locator(FixedLocator(probability_major_ticks(ymin)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)


def format_persistence_axis(ax: plt.Axes, ymax: float) -> None:
    ax.set_ylim(0.0, float(ymax))
    ticks = [tick for tick in PERSISTENCE_YTICKS if tick <= ymax]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.grid(True, linestyle="--", alpha=0.35)


def bar_width(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.42
    diffs = np.diff(np.sort(np.unique(x)))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 0.42
    return float(np.min(positive) * 0.86)


def plot_distance_layer(
    ax: plt.Axes,
    layer: dict[str, object],
    pmf: np.ndarray,
    times: list[float],
    ylim: tuple[float, float],
) -> None:
    family = str(layer["family"])
    time = float(layer["time"])
    color = layer_color(family, time, times)
    x = np.arange(pmf.size, dtype=float)
    visible = pmf > 0.0
    ax.bar(
        x[visible],
        pmf[visible],
        width=bar_width(x),
        color=color,
        edgecolor="#4d4d4d",
        linewidth=0.22,
        alpha=1.0,
    )
    format_probability_axis(ax, ylim)


def plot_hyperbolicity_layer(
    ax: plt.Axes,
    layer: dict[str, object],
    pmf: np.ndarray,
    times: list[float],
    delta_xmax: float,
    ylim: tuple[float, float],
) -> None:
    family = str(layer["family"])
    time = float(layer["time"])
    color = layer_color(family, time, times)
    x = np.arange(pmf.size, dtype=float) / 2.0
    ticks = np.arange(pmf.size)
    visible = (x <= delta_xmax) & (pmf > 0.0)
    integer = visible & (ticks % 2 == 0)
    half_integer = visible & (ticks % 2 == 1)
    width = bar_width(x)
    if np.any(integer):
        ax.bar(
            x[integer],
            pmf[integer],
            width=width,
            color=color,
            edgecolor="#4d4d4d",
            linewidth=0.20,
            alpha=1.0,
            zorder=3,
        )
    if np.any(half_integer):
        ax.bar(
            x[half_integer],
            pmf[half_integer],
            width=width,
            color=lighten_color(color, 0.48),
            edgecolor="#4d4d4d",
            linewidth=0.20,
            alpha=1.0,
            zorder=4,
        )
    ax.set_xlim(HYPERBOLICITY_XMIN, delta_xmax)
    format_probability_axis(ax, ylim)


def plot_mds_panel(
    ax: plt.Axes,
    mds: dict[str, np.ndarray],
    layers: list[dict[str, object]],
    times: list[float],
    model: str,
) -> None:
    x = mds["x"]
    y = mds["y"]
    layer_index = mds["layer_index"]
    # Opaque markers: overlapping translucent points blend into false intermediate colours and
    # muddy the distinct stages. Draw lightest (earliest) last so the pale points stay visible.
    draw_order = sorted(
        range(len(layers)),
        key=lambda index: color_luminance(
            layer_color(str(layers[index]["family"]), float(layers[index]["time"]), times)
        ),
    )
    for index in draw_order:
        layer = layers[index]
        family = str(layer["family"])
        time = float(layer["time"])
        color = layer_color(family, time, times)
        mask = layer_index == index
        ax.scatter(
            x[mask],
            y[mask],
            color=color,
            s=13.5,
            edgecolors="#4d4d4d",
            linewidths=0.28,
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def plot_time_key(
    ax: plt.Axes,
    layers: list[dict[str, object]],
    times: list[float],
) -> None:
    ax.axis("off")
    handles = [
        Patch(
            facecolor=layer_color(str(layer["family"]), float(layer["time"]), times),
            edgecolor="#4d4d4d",
            linewidth=0.35,
            label=layer_label(float(layer["time"])),
        )
        for layer in layers
    ]
    ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
        mode="expand",
        ncol=len(handles),
        frameon=False,
        fontsize=7.8,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )


def plot_persistence_layer(
    ax: plt.Axes,
    layer: dict[str, object],
    result: dict[str, np.ndarray | int | float],
    times: list[float],
    length: int,
    samples: int,
    smoothing_sigma: float,
    contour_resolution: int,
    ymax: float,
) -> None:
    xlim = (0.0, float(length))
    ylim = (0.0, float(ymax))
    shade_hamming_exclusion(ax, length, xlim, ylim)
    ymax_int = int(math.ceil(ymax))
    density = density_grid(
        result,
        length=length,
        ymax=ymax_int,
        samples=samples,
        smoothing_sigma=smoothing_sigma,
    )
    x_grid, y_grid, density = dense_contour_grid(
        density,
        length=length,
        ymax=ymax_int,
        contour_resolution=contour_resolution,
    )
    family = str(layer["family"])
    time = float(layer["time"])
    color = layer_color(family, time, times)
    plot_model_contours(
        ax,
        x_grid,
        y_grid,
        density,
        color,
        fill=True,
        linestyle="-",
        linewidth=0.72,
        zorder=3,
    )
    ax.set_xlim(xlim)
    format_persistence_axis(ax, ymax)
    birth = result["birth"]
    lifetime = result["lifetime"]
    count = result["count"]
    assert isinstance(birth, np.ndarray)
    assert isinstance(lifetime, np.ndarray)
    assert isinstance(count, np.ndarray)
    plot_observed_persistence_support(ax, birth, lifetime, count, color, ymax)


def format_stack_axis(
    ax: plt.Axes,
    *,
    show_x: bool,
    show_y: bool,
) -> None:
    ax.tick_params(axis="both", which="major", labelsize=6.5, pad=1.0)
    ax.tick_params(axis="both", which="minor", labelsize=0, pad=1.0)
    if not show_x:
        ax.tick_params(labelbottom=False)
    if not show_y:
        ax.tick_params(labelleft=False)


def make_figure(
    layers_by_model: dict[str, list[dict[str, object]]],
    mds_by_model: dict[str, dict[str, np.ndarray]],
    distance_by_model: dict[str, list[np.ndarray]],
    hyperbolicity_by_model: dict[str, list[np.ndarray]],
    persistence_by_model: dict[str, list[dict[str, np.ndarray | int | float]]],
    length: int,
    times: list[float],
    persistence_samples: int,
    smoothing_sigma: float,
    contour_resolution: int,
    persistence_ymax: float,
) -> plt.Figure:
    models = ("JC", "HB")
    n_layers = len(layers_by_model[models[0]])
    if any(len(layers_by_model[model]) != n_layers for model in models):
        raise ValueError("All model families must have the same number of time layers.")

    fig_height = max(7.8, 2.05 * n_layers + 0.55)
    fig = plt.figure(figsize=(18.6, fig_height))
    height_ratios = [1.0] * n_layers + [0.62] + [1.0] * n_layers
    grid = fig.add_gridspec(
        len(height_ratios),
        4,
        height_ratios=height_ratios,
        width_ratios=[1.95, 1.08, 1.08, 1.22],
        hspace=0.08,
        wspace=0.16,
    )

    delta_xmax = max(
        displayed_delta_max(
            [
                {"pmf": pmf}
                for model in models
                for pmf in hyperbolicity_by_model[model]
            ],
            length,
        ),
        5.0,
    )
    hyperbolicity_x = np.arange(
        max(pmf.size for model in models for pmf in hyperbolicity_by_model[model]),
        dtype=float,
    ) / 2.0
    distance_ylim = probability_ylim(
        [pmf for model in models for pmf in distance_by_model[model]]
    )
    hyperbolicity_ylim = probability_ylim(
        [
            pmf[hyperbolicity_x[: pmf.size] <= delta_xmax]
            for model in models
            for pmf in hyperbolicity_by_model[model]
        ]
    )
    model_starts = {"JC": 0, "HB": n_layers + 1}
    mds_key_pairs = []
    column_title_axes: list[tuple[plt.Axes, str]] = []

    for model in models:
        start = model_starts[model]
        layers = layers_by_model[model]
        mds_block = grid[start : start + n_layers, 0].subgridspec(
            2,
            1,
            height_ratios=[1.0, 0.10],
            hspace=0.06,
        )
        mds_ax = fig.add_subplot(mds_block[0])
        time_key_ax = fig.add_subplot(mds_block[1])
        distance_axes = [
            fig.add_subplot(grid[start + layer_index, 1])
            for layer_index in range(n_layers)
        ]
        hyperbolicity_axes = [
            fig.add_subplot(grid[start + layer_index, 2])
            for layer_index in range(n_layers)
        ]
        persistence_axes = [
            fig.add_subplot(grid[start + layer_index, 3])
            for layer_index in range(n_layers)
        ]

        plot_mds_panel(mds_ax, mds_by_model[model], layers, times, model)
        plot_time_key(time_key_ax, layers, times)
        mds_key_pairs.append((mds_ax, time_key_ax))

        mds_ax.text(
            -0.18,
            0.5,
            "Jukes-Cantor" if model == "JC" else "Halpern-Bruno",
            transform=mds_ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
        )

        if model == "JC":
            column_title_axes = [
                (mds_ax, "Metric MDS"),
                (distance_axes[0], "Pairwise distance"),
                (hyperbolicity_axes[0], "Four-point hyperbolicity"),
                (persistence_axes[0], "Four-point persistence"),
            ]

        for layer_index, layer in enumerate(layers):
            show_x = layer_index == n_layers - 1
            show_y = layer_index == n_layers // 2
            distance_ax = distance_axes[layer_index]
            hyperbolicity_ax = hyperbolicity_axes[layer_index]
            persistence_ax = persistence_axes[layer_index]
            plot_distance_layer(
                distance_ax,
                layer,
                distance_by_model[model][layer_index],
                times,
                distance_ylim,
            )
            plot_hyperbolicity_layer(
                hyperbolicity_ax,
                layer,
                hyperbolicity_by_model[model][layer_index],
                times,
                delta_xmax,
                hyperbolicity_ylim,
            )
            plot_persistence_layer(
                persistence_ax,
                layer,
                persistence_by_model[model][layer_index],
                times,
                length,
                persistence_samples,
                smoothing_sigma,
                contour_resolution,
                persistence_ymax,
            )

            distance_ax.set_xlim(-0.5, length + 0.5)
            format_stack_axis(distance_ax, show_x=show_x, show_y=show_y)
            format_stack_axis(hyperbolicity_ax, show_x=show_x, show_y=show_y)
            format_stack_axis(persistence_ax, show_x=show_x, show_y=show_y)

            if show_y:
                distance_ax.set_ylabel("Probability", fontsize=8.0)
                hyperbolicity_ax.set_ylabel("Probability", fontsize=8.0)
                persistence_ax.set_ylabel("Persistence", fontsize=8.0)
            if show_x:
                distance_ax.set_xlabel("Hamming distance", fontsize=8.5)
                hyperbolicity_ax.set_xlabel(
                    "Hyperbolicity deficit",
                    fontsize=8.5,
                )
                persistence_ax.set_xlabel("Birth", fontsize=8.5)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.06, top=0.965, wspace=0.16, hspace=0.08)
    fig.canvas.draw()
    mds_vertical_shift = -0.012
    for mds_ax, time_key_ax in mds_key_pairs:
        mds_position = mds_ax.get_position()
        key_position = time_key_ax.get_position()
        mds_ax.set_position(
            [
                mds_position.x0,
                mds_position.y0 + mds_vertical_shift,
                mds_position.width,
                mds_position.height,
            ]
        )
        time_key_ax.set_position(
            [
                mds_position.x0,
                key_position.y0 + mds_vertical_shift,
                mds_position.width,
                key_position.height,
            ]
        )
    title_y = max(ax.get_position().y1 for ax, _ in column_title_axes) + 0.018
    for ax, title in column_title_axes:
        position = ax.get_position()
        fig.text(
            0.5 * (position.x0 + position.x1),
            title_y,
            title,
            ha="center",
            va="bottom",
            fontsize=13.5,
        )
    return fig


def load_results(path: Path):
    """The model layers, their parameters and the four data dicts stored in ``path``."""
    payload = dict(np.load(path))
    times = [float(time) for time in payload["times"]]
    length = int(payload["length"])
    samples = int(payload["persistence_samples"])

    models = [str(model) for model in payload["models"]]
    layers_by_model = {
        model: [{"family": model, "time": time} for time in times]
        + [{"family": f"{model} limit", "time": math.inf}]
        for model in models
    }
    mds_by_model, distance_by_model, hyperbolicity_by_model, persistence_by_model = {}, {}, {}, {}
    for model in models:
        mds_by_model[model] = {
            "x": payload[f"{model}_mds_x"], "y": payload[f"{model}_mds_y"],
            "layer_index": payload[f"{model}_mds_layer_index"],
        }
        distance_by_model[model] = [np.asarray(row) for row in payload[f"{model}_distance"]]
        hyperbolicity_by_model[model] = [
            np.asarray(row) for row in payload[f"{model}_hyperbolicity"]
        ]
        results = []
        for i in range(int(payload[f"{model}_n_layers"])):
            count = payload[f"{model}_pers_{i}_count"]
            results.append({
                "birth": payload[f"{model}_pers_{i}_birth"],
                "lifetime": payload[f"{model}_pers_{i}_lifetime"],
                "count": count,
                "probability": count.astype(float) / float(samples),
                "zero_count": int(payload[f"{model}_pers_{i}_zero"]),
                "nonzero_mass": float(np.sum(count) / samples),
            })
        persistence_by_model[model] = results
    data = (mds_by_model, distance_by_model, hyperbolicity_by_model, persistence_by_model)
    return layers_by_model, times, length, samples, data


def main() -> None:
    args = parse_args()
    if args.persistence_ymax is not None and args.persistence_ymax <= 0:
        raise ValueError("persistence-ymax must be positive when supplied.")

    output_stem = resolve_output_stem(args.output_stem, Path(args.figures_dir))
    layers_by_model, times, length, samples, data = load_results(Path(args.data))
    mds_by_model, distance_by_model, hyperbolicity_by_model, persistence_by_model = data

    persistence_ymax = args.persistence_ymax
    if persistence_ymax is None:
        observed_max = max(
            (
                float(np.max(result["lifetime"]))
                for results in persistence_by_model.values()
                for result in results
                if np.asarray(result["lifetime"]).size
            ),
            default=0.0,
        )
        persistence_ymax = float(padded_persistence_ymax(observed_max))
        print(
            f"Persistence axis: observed maximum {observed_max:g}, "
            f"displayed maximum {persistence_ymax:g}"
        )

    fig = make_figure(
        layers_by_model=layers_by_model,
        mds_by_model=mds_by_model,
        distance_by_model=distance_by_model,
        hyperbolicity_by_model=hyperbolicity_by_model,
        persistence_by_model=persistence_by_model,
        length=length,
        times=times,
        persistence_samples=samples,
        smoothing_sigma=args.smoothing_sigma,
        contour_resolution=args.contour_resolution,
        persistence_ymax=persistence_ymax,
    )

    out = output_stem.with_suffix(".pdf")
    # No CreationDate in the PDF, so a re-run reproduces the committed file byte for byte.
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", metadata={"CreationDate": None})
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
