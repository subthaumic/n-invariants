#!/usr/bin/env python3
"""Figure 7: metric n-point invariants of the SARS-CoV-2 data (Section 8.4).

The same layout as Figure 6, for the cumulative SARS-CoV-2 data at five collection-date
cutoffs, earliest at the top. Columns are a 2D metric-MDS embedding of a 50,000-sequence
subsample, then the distributions of the pairwise distance, the four-point hyperbolicity
deficit and the (4,1)-persistence. The rows use their own green colour family, distinct
from the model rows of Figure 6.

Each distribution panel shows the histogram sampled by
``manuscript/sars_cov2/step2_compute_invariants.py``, at accuracy ``epsilon = 0.1`` for the
four-point invariants and ``epsilon = 0.001`` for the distance, both at confidence 0.95.
The script reads these precomputed results and does no sampling itself.

The rows are fixed by EXPECTED_CUTOFFS, which matches CUTOFF_DATES in
``manuscript/sars_cov2/step2_compute_invariants.py``. A cutoff without results is drawn as
an empty panel.

Reads:  manuscript/sars_cov2/results/invariants/invariants_*.npz  (step 2)
        manuscript/sars_cov2/results/mds/mds_embedding_max*.npz   (step 3)
Writes: output/sars_cov2_invariants.pdf. Only the MDS point clouds are rasterized, which
        keeps the file small.

Run from the repository root::

    uv run python manuscript/figures/figure_7_sars_geometry.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    FormatStrFormatter,
    LogFormatterMathtext,
    NullLocator,
)
from scipy.interpolate import RectBivariateSpline  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

# The shared visual language of Figures 6 and 7.
from plotting import (  # noqa: E402
    HYPERBOLICITY_XMIN,
    PERSISTENCE_YTICKS,
    Y_FLOOR,
    family_palette,
    lighten_color,
    padded_persistence_ymax,
    plot_model_contours,
    plot_observed_persistence_support,
    shade_hamming_exclusion,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
SARS_RESULTS = MANUSCRIPT_DIR / "sars_cov2" / "results"

INVARIANTS_DIR = SARS_RESULTS / "invariants"
MDS_DIR = SARS_RESULTS / "mds"
FIGURES_DIR = SCRIPT_DIR / "output"
DEFAULT_OUTPUT_STEM = "sars_cov2_invariants"
DEFAULT_DPI = 800
MDS_POINT_CLOUD_GID = "sars-mds-point-cloud"
# A sub-bin kernel keeps the integer-distance structure visible: at this width,
# a curve value is determined mainly by its nearest bin and two neighbours.
DEFAULT_DISTANCE_KDE_BANDWIDTH = 0.65
DISTANCE_KDE_RESOLUTION = 8

# The date slices, in order. Must match CUTOFF_DATES in step2_compute_invariants.py.
# The figure always has one row per entry; a slice with no invariants_<cutoff>.npz on disk
# yet renders as an empty panel.
EXPECTED_CUTOFFS = [
    "2020-06-30",
    "2021-04-30",
    "2021-10-31",
    "2022-03-01",
    # "2022-08-01",  # ALT Omicron (replaces 2022-03-01); keep in sync with CUTOFF_DATES.
    "2024-03-01",
]

CUTOFF_LABELS = {
    "2020-06-30": ("30 Jun 2020", "pre-VOC"),
    "2021-04-30": ("30 Apr 2021", "pre-Delta"),
    "2021-10-31": ("31 Oct 2021", "Delta"),
    "2022-03-01": ("1 Mar 2022", "Omicron"),
    # "2022-08-01": ("1 Aug 2022", "Omicron"),  # ALT cutoff above
    "2024-03-01": ("1 Mar 2024", "XBB"),
}

# The real data gets its own green cubehelix family, defined here (the JC blue and HB warm
# families live in plotting.py, built the same way via family_palette). start/rot/hue
# pick the hue; the shared light/dark endpoints come from family_palette. Slices are spaced evenly
# light (earliest) -> dark (latest); MDS markers are outlined (see plot_mds_panel) to keep the
# pale early slices visible on white.
DATA_CUBEHELIX = dict(start=2.00, rot=-0.15, hue=1.80)  # green
BAR_EDGE = "#4d4d4d"

PROBABILITY_YMAX = 1.0
DEFAULT_SMOOTHING_SIGMA = 1.0
DEFAULT_CONTOUR_RESOLUTION = 6

# Geometry matched to one family block of figure_6_substitution_models: identical figure width
# and width_ratios, and the same ~1 inch per row so the stacked panels (and the MDS square
# that spans them, with its swatch key) are the same size as in the model figure.
FIG_WIDTH = 18.6
ROW_HEIGHT = 1.025          # inches per date-slice row (model grid: 2.05 per two-family row)
VERTICAL_MARGIN_IN = 0.85   # inches reserved for column titles (top) + x labels (bottom)

# Fallback axis limits used for rows whose slice is not on disk yet (and there is no other
# slice to borrow limits from). Once any slice is present the limits are data-driven.
DIST_YLIM_FALLBACK = (1.0e-4, 1.0)
HYP_YLIM_FALLBACK = (1.0e-9, 1.0)
DIST_XMAX_FALLBACK = 130
DELTA_XMAX_FALLBACK = 6.0
PERS_WIDTH_FALLBACK = 26
LENGTH_FALLBACK = 28207


# ---------------------------------------------------------------------------
# Colour ramp: the green "data" family, one evenly-spaced stage per slice.
# ---------------------------------------------------------------------------
def layer_color(index: int, n_slices: int):
    palette = family_palette(max(n_slices, 1), **DATA_CUBEHELIX)
    return tuple(palette[min(index, len(palette) - 1)])


def color_luminance(color) -> float:
    r, g, b = mcolors.to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ---------------------------------------------------------------------------
# Log-probability axis helpers (verbatim from the model grid).
# ---------------------------------------------------------------------------
def probability_ylim(pmfs: list[np.ndarray]) -> tuple[float, float]:
    positive = [pmf[pmf > 0.0] for pmf in pmfs if np.any(pmf > 0.0)]
    if not positive:
        return (Y_FLOOR, PROBABILITY_YMAX)
    values = np.concatenate(positive)
    return (max(Y_FLOOR, float(np.min(values)) * 0.7), PROBABILITY_YMAX)


def probability_major_ticks(ymin: float) -> list[float]:
    lower_power = math.floor(math.log10(max(ymin, np.nextafter(0.0, 1.0))))
    return [10.0**power for power in range(0, lower_power - 1, -3) if 10.0**power >= ymin]


def format_probability_axis(ax: plt.Axes, ylim: tuple[float, float]) -> None:
    ymin = ylim[0]
    ax.set_yscale("log")
    ax.set_ylim(ymin, PROBABILITY_YMAX)
    ax.yaxis.set_major_locator(FixedLocator(probability_major_ticks(ymin)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # Major (decade) gridlines only. The model grid uses which="both", but its ~9-decade panels
    # make matplotlib drop the minor log ticks; our narrower panels would otherwise show minor
    # gridlines between the decades, which the JC/HB panels don't have.
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)


def format_distance_axis(ax: plt.Axes, ylim: tuple[float, float]) -> None:
    """Like format_probability_axis, but for the pairwise-distance column: a labelled major
    tick with gridline at every decade, and no minor ticks. The distance panel spans only a
    few decades, so one tick per decade is not crowded; the hyperbolicity panel spans about
    nine decades and keeps a tick every third decade."""
    ymin = ylim[0]
    lower_power = math.floor(math.log10(max(ymin, np.nextafter(0.0, 1.0))))
    decades = [10.0**power for power in range(0, lower_power - 1, -1) if 10.0**power >= ymin]
    ax.set_yscale("log")
    ax.set_ylim(ymin, PROBABILITY_YMAX)
    ax.yaxis.set_major_locator(FixedLocator(decades))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)


def format_persistence_axis(ax: plt.Axes, ymax: int) -> None:
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


def format_stack_axis(ax: plt.Axes, *, show_x: bool, show_y: bool) -> None:
    ax.tick_params(axis="both", which="major", labelsize=6.5, pad=1.0)
    ax.tick_params(axis="both", which="minor", labelsize=0, pad=1.0)
    if not show_x:
        ax.tick_params(labelbottom=False)
    if not show_y:
        ax.tick_params(labelleft=False)


# ---------------------------------------------------------------------------
# Loading + converting the sampled histograms into panel arrays.
# ---------------------------------------------------------------------------
class Slice:
    """One date slice's invariant histograms, converted to plotting arrays."""

    def __init__(self, path: Path):
        data = dict(np.load(path, allow_pickle=True))
        self.cutoff = str(data["cutoff_date"])
        self.n_sequences = int(data["n_sequences"])
        self.n_haplotypes = int(data["n_haplotypes"])
        self.n_sites = int(data["n_sites"])
        self.diam = int(data["diam"])

        # Pairwise distance pmf indexed by integer Hamming distance.
        dm = data["dm_dict"].item()
        dist_total = sum(dm.values())
        dmax = int(max(dm)) if dm else 0
        self.distance_pmf = np.zeros(dmax + 1, dtype=float)
        for dist, count in dm.items():
            self.distance_pmf[int(round(dist))] = count / dist_total

        # Hyperbolicity pmf indexed so that index i -> deficit i/2 (half-integer grid).
        hy = data["hyp_hist"].item()
        hyp_total = sum(hy.values())
        imax = int(round(2 * max(hy))) if hy else 0
        self.hyperbolicity_pmf = np.zeros(imax + 1, dtype=float)
        for deficit, count in hy.items():
            self.hyperbolicity_pmf[int(round(2 * deficit))] = count / hyp_total

        # Persistence (birth, lifetime) histogram over the nonzero-lifetime samples.
        bd = data["bd_dict"].item()
        self.persistence_samples = int(data["n_samples_persistence"])
        births, lifetimes, counts = [], [], []
        for (birth, death), count in bd.items():
            life = death - birth
            if life > 0:
                births.append(int(round(birth)))
                lifetimes.append(int(round(life)))
                counts.append(int(count))
        self.pers_birth = np.asarray(births, dtype=np.int64)
        self.pers_life = np.asarray(lifetimes, dtype=np.int64)
        self.pers_count = np.asarray(counts, dtype=np.int64)


def load_present_slices() -> dict[str, Slice]:
    """Load every invariants_<cutoff>.npz on disk, keyed by cutoff date."""
    slices = {}
    for path in sorted(INVARIANTS_DIR.glob("invariants_*.npz")):
        sl = Slice(path)
        slices[sl.cutoff] = sl
    return slices


def load_mds():
    """Return (embedding, first_seen dates, requested_max_points, reference_row) for the largest run, or None.

    ``reference_row`` is the index of the reference genome within the embedding, or -1 if
    the embedding does not include it.
    """
    paths = list(MDS_DIR.glob("mds_embedding_max*.npz"))
    if not paths:
        return None
    path = max(paths, key=lambda p: int(np.load(p, allow_pickle=True)["requested_max_points"]))
    data = np.load(path, allow_pickle=True)
    reference_row = int(data["reference_row"]) if "reference_row" in data.files else -1
    return (
        np.asarray(data["mds_embedding"], dtype=float),
        np.asarray(data["first_seen_date"]),
        int(data["requested_max_points"]),
        reference_row,
    )


# ---------------------------------------------------------------------------
# Persistence density on a data-scaled birth window. Mirrors the model grid's
# density_grid / dense_contour_grid, but keeps the true genome length in the
# physical support so the Hamming-exclusion geometry stays correct while the
# rendered window stays small (births are tiny relative to L).
# ---------------------------------------------------------------------------
def _support_mask(x_grid, y_grid, width, length):
    return (
        (x_grid > 0)
        & (x_grid <= width)
        & (y_grid > 0)
        & (y_grid <= x_grid)
        & (y_grid <= length - x_grid)
    )


def persistence_density(sl: Slice, width: int, ymax: int, length: int, sigma: float):
    grid = np.zeros((ymax + 1, width + 1), dtype=float)
    if sl.pers_birth.size:
        visible = (sl.pers_life <= ymax) & (sl.pers_birth <= width)
        np.add.at(
            grid,
            (sl.pers_life[visible], sl.pers_birth[visible]),
            sl.pers_count[visible].astype(float) / float(sl.persistence_samples),
        )
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=float(sigma), mode="constant", cval=0.0)
    x = np.arange(width + 1, dtype=float)
    y = np.arange(ymax + 1, dtype=float)
    x_grid, y_grid = np.meshgrid(x, y)
    return np.where(_support_mask(x_grid, y_grid, width, length), grid, np.nan)


def dense_persistence_grid(density, width, ymax, length, contour_resolution):
    contour_resolution = max(int(contour_resolution), 1)
    x_native = np.arange(width + 1, dtype=float)
    y_native = np.arange(ymax + 1, dtype=float)
    values = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    x_degree = min(3, len(x_native) - 1)
    y_degree = min(3, len(y_native) - 1)
    spline = RectBivariateSpline(y_native, x_native, values, kx=y_degree, ky=x_degree, s=0.0)
    x_dense = np.linspace(0.0, float(width), width * contour_resolution + 1)
    y_dense = np.linspace(0.0, float(ymax), ymax * contour_resolution + 1)
    dense = np.maximum(spline(y_dense, x_dense), 0.0)
    x_grid, y_grid = np.meshgrid(x_dense, y_dense)
    return x_grid, y_grid, np.where(_support_mask(x_grid, y_grid, width, length), dense, np.nan)


# ---------------------------------------------------------------------------
# Individual panels.
# ---------------------------------------------------------------------------

def distance_kde_curve(
    pmf: np.ndarray,
    bandwidth: float,
    resolution: int = DISTANCE_KDE_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE of an integer-valued PMF, evaluated on a dense distance grid."""
    support = np.flatnonzero(pmf > 0.0).astype(float)
    if support.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    weights = np.asarray(pmf[support.astype(int)], dtype=float)
    weights /= float(np.sum(weights))
    xmax = max(float(pmf.size - 1), float(support[-1]))
    n_grid = max(int(math.ceil(xmax * resolution)) + 1, 2)
    x_dense = np.linspace(0.0, xmax, n_grid)
    standardized = (x_dense[:, None] - support[None, :]) / bandwidth
    density = np.exp(-0.5 * standardized * standardized) @ weights
    density /= bandwidth * math.sqrt(2.0 * math.pi)
    area = float(np.trapezoid(density, x_dense))
    if area > 0.0:
        density /= area
    return x_dense, density


def plot_distance_layer(ax, pmf, color, ylim, xmax, kde_bandwidth) -> None:
    x, density = distance_kde_curve(pmf, kde_bandwidth)
    if x.size:
        visible = density >= ylim[0]
        ax.fill_between(
            x,
            ylim[0],
            density,
            where=visible,
            interpolate=True,
            facecolor=color,
            edgecolor="none",
            alpha=1.0,
            zorder=2,
        )
        ax.plot(
            x,
            np.where(visible, density, np.nan),
            color=color,
            linewidth=0.82,
            alpha=1.0,
            zorder=3,
        )
    ax.set_xlim(-0.5, xmax + 0.5)
    format_distance_axis(ax, ylim)


def plot_hyperbolicity_layer(ax, pmf, color, delta_xmax, ylim) -> None:
    x = np.arange(pmf.size, dtype=float) / 2.0
    ticks = np.arange(pmf.size)
    visible = (x <= delta_xmax) & (pmf > 0.0)
    integer = visible & (ticks % 2 == 0)
    half = visible & (ticks % 2 == 1)
    width = bar_width(x)
    if np.any(integer):
        ax.bar(x[integer], pmf[integer], width=width, color=color,
               edgecolor=BAR_EDGE, linewidth=0.20, alpha=1.0, zorder=3)
    if np.any(half):
        ax.bar(x[half], pmf[half], width=width, color=lighten_color(color, 0.48),
               edgecolor=BAR_EDGE, linewidth=0.20, alpha=1.0, zorder=4)
    ax.set_xlim(HYPERBOLICITY_XMIN, delta_xmax)
    format_probability_axis(ax, ylim)


def plot_persistence_layer(ax, sl, color, width, ymax, length, sigma, contour_resolution) -> None:
    xlim = (0.0, float(width))
    ylim = (0.0, float(ymax))
    shade_hamming_exclusion(ax, length, xlim, ylim)
    density = persistence_density(sl, width, ymax, length, sigma)
    x_grid, y_grid, dense = dense_persistence_grid(density, width, ymax, length, contour_resolution)
    plot_model_contours(ax, x_grid, y_grid, dense, color, fill=True,
                        linestyle="-", linewidth=0.72, zorder=3)
    ax.set_xlim(xlim)
    format_persistence_axis(ax, ymax)
    plot_observed_persistence_support(
        ax, sl.pers_birth, sl.pers_life, sl.pers_count, color, ymax,
    )


def mark_pending(ax) -> None:
    ax.text(0.5, 0.5, "awaiting\ninvariants", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5, color="0.6", style="italic")


def plot_mds_panel(ax, mds, cutoffs) -> None:
    """Metric-MDS panel, each point coloured by the first cutoff slice that contains it."""
    n = len(cutoffs)
    if mds is None:
        ax.text(0.5, 0.5, "Metric-MDS embedding\npending\n(run step3_compute_mds.py)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
                color="#636363")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#bdbdbd")
            spine.set_linestyle((0, (4, 4)))
        return

    coords, first_seen, _, reference_row = mds
    cutoff64 = np.array([np.datetime64(c, "D") for c in cutoffs])
    # Each point joins X in the first slice whose cutoff is on/after its first-seen date;
    # points later than the last cutoff fall into the final (full-dataset) slice.
    point_slice = np.searchsorted(cutoff64, first_seen.astype("datetime64[D]"), side="left")
    point_slice = np.clip(point_slice, 0, n - 1)

    # The reference genome is drawn on its own below, not as a date-bucket point.
    is_reference = np.zeros(coords.shape[0], dtype=bool)
    if 0 <= reference_row < coords.shape[0]:
        is_reference[reference_row] = True

    # Opaque markers: overlapping translucent points blend into false intermediate colours and
    # muddy the five distinct slices. Draw lightest (earliest) last so the pale founder points
    # stay visible on top of the darker, more numerous later slices.
    draw_order = sorted(range(n), key=lambda i: color_luminance(layer_color(i, n)))
    for i in draw_order:
        mask = (point_slice == i) & ~is_reference
        if np.any(mask):
            points = ax.scatter(
                coords[mask, 0], coords[mask, 1], color=layer_color(i, n),
                s=13.5, edgecolors="#4d4d4d", linewidths=0.28,
            )
            points.set_gid(MDS_POINT_CLOUD_GID)
    if is_reference.any():
        rx, ry = coords[reference_row]
        ax.scatter([rx], [ry], marker="*", s=190, color="#f0f0f0", edgecolors="#111111",
                   linewidths=0.9, zorder=6, label="reference (Wuhan/WIV04)")
        ax.annotate("Wuhan/WIV04", (rx, ry), textcoords="offset points", xytext=(5, 4),
                    fontsize=6.5, color="#111111", zorder=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def plot_date_key(ax, cutoffs) -> None:
    """Legend keyed by human-readable collection-date cutoff and variant era."""
    n = len(cutoffs)
    ax.axis("off")
    handles = []
    for i, cutoff in enumerate(cutoffs):
        display_date, era = CUTOFF_LABELS[cutoff]
        handles.append(
            Patch(
                facecolor=layer_color(i, n), edgecolor=BAR_EDGE, linewidth=0.35,
                label=rf"$t \leq$ {display_date} ({era})",
            )
        )
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
              mode="expand", ncol=min(len(handles), 2), frameon=False, fontsize=7.2,
              handlelength=1.0, handletextpad=0.35, columnspacing=0.75, borderaxespad=0.0)


def rasterize_mds_point_cloud(fig: plt.Figure) -> int:
    """Rasterize only the dense MDS point clouds, leaving all other artists as vectors."""
    point_clouds = fig.findobj(
        lambda artist: artist.get_gid() == MDS_POINT_CLOUD_GID
    )
    for point_cloud in point_clouds:
        point_cloud.set_rasterized(True)
    return len(point_clouds)


# ---------------------------------------------------------------------------
# Figure assembly.
# ---------------------------------------------------------------------------
def make_figure(
    cutoffs, by_cutoff, mds, smoothing_sigma, contour_resolution,
    distance_kde_bandwidth,
):
    n = len(cutoffs)
    present = [by_cutoff[c] for c in cutoffs if c in by_cutoff]

    # Shared limits, data-driven from whichever slices are present (fallbacks otherwise), so
    # empty rows use the same axes as filled ones and the scaffold is stable as slices land.
    distance_ylim = probability_ylim([s.distance_pmf for s in present]) if present else DIST_YLIM_FALLBACK
    dist_xmax = max((s.distance_pmf.size - 1 for s in present), default=DIST_XMAX_FALLBACK)

    hyp_ylim = probability_ylim([s.hyperbolicity_pmf for s in present]) if present else HYP_YLIM_FALLBACK
    max_deficit = max(((s.hyperbolicity_pmf.size - 1) / 2.0 for s in present), default=DELTA_XMAX_FALLBACK)
    delta_xmax = max(max_deficit, 3.0)

    length = max((s.n_sites for s in present), default=LENGTH_FALLBACK)
    max_birth = max(
        (int(s.pers_birth.max()) for s in present if s.pers_birth.size),
        default=PERS_WIDTH_FALLBACK,
    )
    max_life = max((int(s.pers_life.max()) for s in present if s.pers_life.size), default=0)
    pers_width = max(int(math.ceil(max_birth * 1.15)), 8)
    pers_ymax = padded_persistence_ymax(max_life)

    fig_height = ROW_HEIGHT * n + VERTICAL_MARGIN_IN
    fig = plt.figure(figsize=(FIG_WIDTH, fig_height))
    grid = fig.add_gridspec(
        n, 4,
        width_ratios=[1.95, 1.08, 1.08, 1.22],
        hspace=0.08, wspace=0.16,
    )

    # MDS block spans all rows in column 0, with a date-key strip beneath it.
    mds_block = grid[0:n, 0].subgridspec(2, 1, height_ratios=[1.0, 0.10], hspace=0.06)
    mds_ax = fig.add_subplot(mds_block[0])
    key_ax = fig.add_subplot(mds_block[1])
    plot_mds_panel(mds_ax, mds, cutoffs)
    plot_date_key(key_ax, cutoffs)
    mds_ax.text(-0.16, 0.5, "SARS-CoV-2", transform=mds_ax.transAxes,
                rotation=90, ha="center", va="center", fontsize=11)

    distance_axes, hyp_axes, pers_axes = [], [], []
    for row in range(n):
        distance_axes.append(fig.add_subplot(grid[row, 1]))
        hyp_axes.append(fig.add_subplot(grid[row, 2]))
        pers_axes.append(fig.add_subplot(grid[row, 3]))

    for row, cutoff in enumerate(cutoffs):
        color = layer_color(row, n)
        show_x = row == n - 1
        show_y = row == n // 2
        sl = by_cutoff.get(cutoff)
        distance_ax, hyp_ax, pers_ax = distance_axes[row], hyp_axes[row], pers_axes[row]

        if sl is not None:
            plot_distance_layer(
                distance_ax, sl.distance_pmf, color, distance_ylim, dist_xmax,
                distance_kde_bandwidth,
            )
            plot_hyperbolicity_layer(hyp_ax, sl.hyperbolicity_pmf, color, delta_xmax, hyp_ylim)
            plot_persistence_layer(pers_ax, sl, color, pers_width, pers_ymax,
                                   length, smoothing_sigma, contour_resolution)
        else:
            distance_ax.set_xlim(-0.5, dist_xmax + 0.5)
            format_distance_axis(distance_ax, distance_ylim)
            mark_pending(distance_ax)
            hyp_ax.set_xlim(HYPERBOLICITY_XMIN, delta_xmax)
            format_probability_axis(hyp_ax, hyp_ylim)
            mark_pending(hyp_ax)
            shade_hamming_exclusion(pers_ax, length, (0.0, float(pers_width)), (0.0, float(pers_ymax)))
            pers_ax.set_xlim(0.0, float(pers_width))
            format_persistence_axis(pers_ax, pers_ymax)
            mark_pending(pers_ax)

        format_stack_axis(distance_ax, show_x=show_x, show_y=show_y)
        format_stack_axis(hyp_ax, show_x=show_x, show_y=show_y)
        format_stack_axis(pers_ax, show_x=show_x, show_y=show_y)

        if show_y:
            distance_ax.set_ylabel("Density", fontsize=8.0)
            hyp_ax.set_ylabel("Probability", fontsize=8.0)
            pers_ax.set_ylabel("Persistence", fontsize=8.0)
        if show_x:
            distance_ax.set_xlabel("Hamming distance", fontsize=8.5)
            hyp_ax.set_xlabel("Hyperbolicity deficit", fontsize=8.5)
            pers_ax.set_xlabel("Birth", fontsize=8.5)

    # Inch-based top/bottom margins so row height stays fixed as the row count changes.
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.45 / fig_height,
                        top=1.0 - 0.32 / fig_height, wspace=0.16, hspace=0.08)
    fig.canvas.draw()

    # set_box_aspect shrinks the MDS panel to a centred square inside its (wider) column, so its
    # post-draw position is narrower than the gridspec cell. Match the date-key strip to that
    # square (x0 + width) so the swatch legend never extends past the MDS boundaries.
    mds_pos = mds_ax.get_position()
    key_pos = key_ax.get_position()
    key_ax.set_position([mds_pos.x0, key_pos.y0, mds_pos.width, key_pos.height])

    # Column titles centred over each column.
    title_axes = [
        (mds_ax, "Metric MDS"),
        (distance_axes[0], "Pairwise distance"),
        (hyp_axes[0], "Four-point hyperbolicity"),
        (pers_axes[0], "Four-point persistence"),
    ]
    title_y = max(ax.get_position().y1 for ax, _ in title_axes) + 0.018
    for ax, title in title_axes:
        pos = ax.get_position()
        fig.text(0.5 * (pos.x0 + pos.x1), title_y, title, ha="center", va="bottom", fontsize=13.5)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Figure 7 from the SARS-CoV-2 results.")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Output directory.")
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output stem of the PDF, in which only the dense MDS point clouds are rasterized.",
    )
    parser.add_argument(
        "--distance-kde-bandwidth",
        type=float,
        default=DEFAULT_DISTANCE_KDE_BANDWIDTH,
        help=(
            "Gaussian KDE bandwidth in Hamming-distance units for the pairwise-distance "
            "column (default: 0.65, blending mainly each integer bin with its neighbours)."
        ),
    )
    parser.add_argument("--smoothing-sigma", type=float, default=DEFAULT_SMOOTHING_SIGMA)
    parser.add_argument("--contour-resolution", type=int, default=DEFAULT_CONTOUR_RESOLUTION)
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Resolution of the rasterized MDS point clouds; everything else stays vector.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.distance_kde_bandwidth <= 0:
        raise ValueError("distance-kde-bandwidth must be positive.")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")
    by_cutoff = load_present_slices()
    present = [c for c in EXPECTED_CUTOFFS if c in by_cutoff]
    pending = [c for c in EXPECTED_CUTOFFS if c not in by_cutoff]
    extra = [c for c in by_cutoff if c not in EXPECTED_CUTOFFS]
    print(f"Expected {len(EXPECTED_CUTOFFS)} slice(s); present: "
          f"{', '.join(present) or 'none'}; pending: {', '.join(pending) or 'none'}")
    if extra:
        print(f"  note: invariants on disk not in EXPECTED_CUTOFFS (ignored): {', '.join(extra)}")

    mds = load_mds()
    print("MDS embedding: " + ("loaded" if mds is not None else "not found (placeholder panel)"))

    fig = make_figure(
        EXPECTED_CUTOFFS, by_cutoff, mds,
        args.smoothing_sigma, args.contour_resolution,
        args.distance_kde_bandwidth,
    )

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = (figures_dir / args.output_stem).with_suffix(".pdf")
    n_rasterized = rasterize_mds_point_cloud(fig)
    # No CreationDate in the PDF, so a re-run reproduces the committed file byte for byte.
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", metadata={"CreationDate": None})
    print(f"Wrote {out} with {n_rasterized} MDS point-cloud layer(s) rasterized at {args.dpi} dpi")
    plt.close(fig)


if __name__ == "__main__":
    main()
