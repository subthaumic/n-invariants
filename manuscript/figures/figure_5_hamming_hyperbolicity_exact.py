"""Figure 5: hyperbolicity deficits of the uniform measure on Hamming cubes (Section 8.2).

For the uniform measure on ``H(Sigma, L)`` the one-site law of the three opposite pair
sums is known in closed form, so the distribution of four-point hyperbolicity deficits is
computed exactly: the L-fold convolution of the one-site law, by FFT, pushed forward along
the gap between the largest and the median sum.

Rows are the alphabet sizes ``k = 2, 4, 8`` and columns the word lengths
``L = 2, 4, 16, 64``. Bars are the exact probabilities, shaded alternately for integer and
half-integer deficits. For ``k = 2`` every half-integer bin is empty, since a binary site
changes the pair sums in steps of two. The black curve is the Gaussian approximation, with
nodes at its values on the support ``l_k * Z_{>=0}``. There is no sampling, and the script
runs in a few seconds.

Run from ``manuscript/``::

    uv run python figures/figure_5_hamming_hyperbolicity_exact.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
from scipy.special import ndtr


def site_distribution_tensor(k: int) -> np.ndarray:
    if k < 2:
        raise ValueError("alphabet size k must be at least 2")

    tensor = np.zeros((3, 3, 3), dtype=float)
    k3 = float(k**3)

    tensor[0, 0, 0] = 1.0 / k3
    tensor[1, 1, 1] = 4.0 * (k - 1) / k3

    for triple in ((0, 2, 2), (2, 0, 2), (2, 2, 0)):
        tensor[triple] = (k - 1) / k3

    for triple in ((1, 2, 2), (2, 1, 2), (2, 2, 1)):
        tensor[triple] = 2.0 * (k - 1) * (k - 2) / k3

    tensor[2, 2, 2] = (k - 1) * (k - 2) * (k - 3) / k3

    total = tensor.sum()
    if not np.isclose(total, 1.0):
        raise RuntimeError(f"single-site tensor does not sum to 1 (got {total})")

    return tensor


def exact_pairing_distribution_fft(k: int, length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("length must be at least 1")

    shape = (2 * length + 1, 2 * length + 1, 2 * length + 1)
    axes = tuple(range(len(shape)))
    kernel = np.zeros(shape, dtype=float)
    kernel[:3, :3, :3] = site_distribution_tensor(k)

    spectrum = np.fft.rfftn(kernel, axes=axes)
    del kernel
    spectrum **= length
    distribution = np.fft.irfftn(spectrum, s=shape, axes=axes)
    del spectrum
    np.maximum(distribution, 0.0, out=distribution)
    distribution /= distribution.sum()
    return distribution


def site_difference_law(k: int) -> dict[tuple[int, int], float]:
    law: dict[tuple[int, int], float] = defaultdict(float)
    tensor = site_distribution_tensor(k)

    for a, b, c in np.argwhere(tensor > 0.0):
        law[(int(a - c), int(b - c))] += float(tensor[a, b, c])

    return dict(law)


def exact_difference_distribution_fft(k: int, length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("length must be at least 1")

    shape = (4 * length + 1, 4 * length + 1)
    axes = tuple(range(len(shape)))
    kernel = np.zeros(shape, dtype=float)

    for (dx, dy), mass in site_difference_law(k).items():
        kernel[dx % shape[0], dy % shape[1]] += mass

    spectrum = np.fft.rfftn(kernel, axes=axes)
    del kernel
    spectrum **= length
    distribution = np.fft.irfftn(spectrum, s=shape, axes=axes)
    del spectrum
    np.maximum(distribution, 0.0, out=distribution)
    distribution /= distribution.sum()
    return distribution


def hyperbolicity_distribution_from_pairing(
    pairing_distribution: np.ndarray, tol: float = 1e-14
) -> tuple[np.ndarray, np.ndarray]:
    masses_by_gap2: dict[int, float] = defaultdict(float)

    for index in np.argwhere(pairing_distribution > tol):
        triple = tuple(int(v) for v in index)
        values = sorted(triple)
        gap2 = values[2] - values[1]
        masses_by_gap2[gap2] += float(pairing_distribution[tuple(index)])

    gap2_values = np.array(sorted(masses_by_gap2), dtype=int)
    masses = np.array([masses_by_gap2[g] for g in gap2_values], dtype=float)
    masses /= masses.sum()
    return gap2_values / 2.0, masses


def signed_difference_coordinates(n: int) -> np.ndarray:
    if n % 2 != 1:
        raise ValueError("difference distribution side length must be odd")

    max_abs = (n - 1) // 2
    coordinates = np.arange(n, dtype=int)
    coordinates[coordinates > max_abs] -= n
    return coordinates


def hyperbolicity_distribution_from_differences(
    difference_distribution: np.ndarray,
    *,
    tol: float = 1e-14,
    chunk_rows: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if difference_distribution.ndim != 2:
        raise ValueError("difference distribution must be two-dimensional")
    if difference_distribution.shape[0] != difference_distribution.shape[1]:
        raise ValueError("difference distribution must be square")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be at least 1")

    coordinates = signed_difference_coordinates(difference_distribution.shape[0])
    max_gap2 = (difference_distribution.shape[0] - 1) // 2
    masses = np.zeros(max_gap2 + 1, dtype=float)
    y = coordinates[None, :]

    for start in range(0, difference_distribution.shape[0], chunk_rows):
        stop = min(start + chunk_rows, difference_distribution.shape[0])
        x = coordinates[start:stop, None]
        block = difference_distribution[start:stop, :]
        maximum = np.maximum(np.maximum(x, y), 0)
        minimum = np.minimum(np.minimum(x, y), 0)
        median = x + y - maximum - minimum
        gap2 = np.asarray(maximum - median, dtype=np.intp)
        weights = np.where(block > tol, block, 0.0)
        masses += np.bincount(gap2.ravel(), weights=weights.ravel(), minlength=max_gap2 + 1)

    support = np.nonzero(masses > 0.0)[0]
    x_vals = support / 2.0
    y_vals = masses[support]
    y_vals /= y_vals.sum()
    return x_vals, y_vals


def exact_hyperbolicity_distribution_fft(k: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    difference_distribution = exact_difference_distribution_fft(k, length)
    return hyperbolicity_distribution_from_differences(difference_distribution)


def single_site_coordinate_variance(k: int) -> float:
    return 2.0 * (k - 1) / (k**2)


def hyperbolicity_lattice_spacing(k: int) -> float:
    return 1.0 if k == 2 else 0.5


def heaviside_boundary_weight(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    weights = np.zeros_like(x, dtype=float)
    weights[x > 0.0] = 1.0
    weights[x == 0.0] = 0.5
    return weights


def standard_normal_spacing_density(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    density = (3.0 / np.sqrt(np.pi)) * np.exp(-(d**2) / 4.0) * ndtr(-d / np.sqrt(6.0))
    density[d < 0] = 0.0
    return density


def standard_normal_spacing_density_unrestricted(d: np.ndarray) -> np.ndarray:
    """Same analytic formula as standard_normal_spacing_density but without zeroing d < 0."""
    d = np.asarray(d, dtype=float)
    return (3.0 / np.sqrt(np.pi)) * np.exp(-(d**2) / 4.0) * ndtr(-d / np.sqrt(6.0))


def hyperbolicity_clt_mass(x: np.ndarray, k: int, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scale = np.sqrt(single_site_coordinate_variance(k) * length)
    spacing = hyperbolicity_lattice_spacing(k)
    return (
        heaviside_boundary_weight(x)
        * (2.0 * spacing / scale)
        * standard_normal_spacing_density(2.0 * x / scale)
    )


def hyperbolicity_clt_density_extension(x: np.ndarray, k: int, length: int) -> np.ndarray:
    """Analytic CLT density without Heaviside truncation, for showing the Gaussian continuation."""
    x = np.asarray(x, dtype=float)
    scale = np.sqrt(single_site_coordinate_variance(k) * length)
    spacing = hyperbolicity_lattice_spacing(k)
    return (2.0 * spacing / scale) * standard_normal_spacing_density_unrestricted(2.0 * x / scale)


def plot_distribution_grid(
    alphabet_sizes: list[int],
    lengths: list[int],
    output_path: Path,
    *,
    dpi: int = 200,
) -> None:
    n_rows = len(alphabet_sizes)
    n_cols = len(lengths)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.55 * n_cols, 2.25 * n_rows),
        squeeze=False,
        constrained_layout=True,
        sharex="col",
        sharey=True,
    )
    fig.get_layout_engine().set(hspace=0.08, wspace=0.06)

    exact_integer_color = "#737373"
    exact_half_color = "#bdbdbd"
    exact_edge_color = "#3f3f3f"
    smooth_color = "#111111"
    exact_bar_width = 0.5

    # Pre-compute all distributions.
    distributions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {
        (k, length): exact_hyperbolicity_distribution_fft(k, length)
        for k in alphabet_sizes
        for length in lengths
    }
    prediction_scatters: list[tuple[object, plt.Axes, float, float]] = []
    node_diameter_fractions = [1 / 8 if col < 2 else 1 / 4 for col in range(n_cols)]
    # Fixed x-limits per column: floor(L/2) is the maximum achievable hyperbolicity.
    col_max_x = {col: length // 2 for col, length in enumerate(lengths)}

    for row, k in enumerate(alphabet_sizes):
        for col, length in enumerate(lengths):
            ax = axes[row, col]
            x_vals, y_vals = distributions[(k, length)]
            spacing = hyperbolicity_lattice_spacing(k)
            x_max = col_max_x[col]

            # Exact FFT law uses the same bar language as the other distribution figures.
            is_half_integer = np.isclose(np.mod(x_vals, 1.0), 0.5, atol=1.0e-9)
            # Retain empty half-integer slots for k=2; for k>=3 every half-step is supported.
            for mask, color in ((~is_half_integer, exact_integer_color),
                                (is_half_integer, exact_half_color)):
                if np.any(mask):
                    ax.bar(x_vals[mask], y_vals[mask], width=exact_bar_width, color=color,
                           edgecolor=exact_edge_color, linewidth=0.28, alpha=0.90, zorder=2)

            # Smooth analytic approximation. At the boundary r=0 this shows the continuous value;
            # the lattice prediction below separately applies the theorem's half-weight.
            x_curve = np.linspace(0.0, x_max + spacing / 2.0, 800)
            y_curve = hyperbolicity_clt_density_extension(x_curve, k, length)
            ax.plot(x_curve, y_curve, color=smooth_color, linewidth=1.65, zorder=5)

            # Lattice CLT prediction: evaluate the asymptotic mass at every supported point.
            lattice = np.arange(0.0, x_max + 0.5 * spacing, spacing)
            predicted_mass = hyperbolicity_clt_mass(lattice, k, length)
            prediction_scatter = ax.scatter(
                lattice, predicted_mass, s=1.0, facecolor=smooth_color,
                edgecolor=smooth_color, linewidth=0.5, zorder=6,
            )
            prediction_scatters.append(
                (prediction_scatter, ax, exact_bar_width, node_diameter_fractions[col])
            )

            # Make the Heaviside half-weight at zero explicit rather than splicing the smooth curve.
            ax.vlines(0.0, predicted_mass[0], y_curve[0], color=smooth_color,
                      linewidth=1.2, zorder=7)

            ax.set_yscale("log")
            ax.set_ylim(1.0e-12, 2.0)
            ax.yaxis.set_major_locator(FixedLocator([1.0, 1e-3, 1e-6, 1e-9, 1e-12]))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.35)
            ax.spines[["top", "right"]].set_visible(False)

            if row == n_rows - 1:
                ax.set_xlabel("hyperbolicity deficit")
            if col == 0:
                ax.set_ylabel("probability")

            # Divide [0, x_max] into at most n_ticks_max−1 equal parts; round to lattice.
            n_ticks_max = 5
            n_lattice = int(round(x_max / spacing)) + 1
            n = min(n_ticks_max, n_lattice)
            raw = np.linspace(0.0, float(x_max), n)
            tick_vals = np.unique(np.round(raw / spacing) * spacing)
            ax.set_xticks(tick_vals)
            ax.xaxis.set_tick_params(labelsize=8)

    # Row labels: k = … on left spine of each row (rotated, outside the axes).
    for row, k in enumerate(alphabet_sizes):
        ax = axes[row, 0]
        ax.annotate(
            rf"$k = {k}$",
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-0.28, 0.5),
            textcoords="axes fraction",
            fontsize=11,
            ha="center",
            va="center",
            rotation=90,
        )

    # Column labels: L = … above each column's top axes.
    for col, length in enumerate(lengths):
        ax = axes[0, col]
        ax.annotate(
            rf"$L = {length}$",
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0.5, 1.02),
            textcoords="axes fraction",
            fontsize=11,
            ha="center",
            va="bottom",
        )

    # Set x-limits after all plotting so autoscaling from bar() cannot override them.
    left_limit = -exact_bar_width / 2.0 - 0.02
    for col in range(n_cols):
        axes[0, col].set_xlim(left_limit, col_max_x[col] + 0.5)

    legend_handles = [
        Patch(facecolor=exact_integer_color, edgecolor=exact_edge_color, linewidth=0.4,
              label="exact"),
        Line2D([], [], color=smooth_color, marker="o", markerfacecolor=smooth_color,
               markeredgewidth=0.5, markersize=4.5, linewidth=1.5,
               label="asymptotic"),
    ]
    legend = axes[-1, -1].legend(handles=legend_handles, loc="lower right", frameon=True,
                                fontsize=8, handlelength=2.2)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#737373")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_alpha(0.95)

    # Resolve constrained layout, then size nodes by the displayed bin width in each column.
    fig.canvas.draw()
    pixels_to_points = 72.0 / fig.dpi
    for scatter, ax, bin_width, diameter_fraction in prediction_scatters:
        x0 = ax.transData.transform((0.0, 1.0))[0]
        x1 = ax.transData.transform((bin_width, 1.0))[0]
        diameter_points = abs(x1 - x0) * pixels_to_points * diameter_fraction
        scatter.set_sizes([diameter_points**2])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # No CreationDate in the PDF, so a re-run reproduces the committed file byte for byte.
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                metadata={"CreationDate": None} if output_path.suffix == ".pdf" else {})
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot exact hyperbolicity distributions for uniform Hamming cubes using "
            "FFT-based convolution of the two-coordinate difference law."
        )
    )
    parser.add_argument(
        "--alphabet-sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Alphabet sizes k to plot.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[2, 4, 16, 64],
        help="Word lengths L to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "uniform_hamming_hyperbolicity.pdf",
        help="Output plot path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster DPI when saving bitmap outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_distribution_grid(
        alphabet_sizes=args.alphabet_sizes,
        lengths=args.lengths,
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
