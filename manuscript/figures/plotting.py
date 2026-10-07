"""Plotting style shared by Figures 6 and 7.

Figures 6 and 7 use the same layout for different point sets (the substitution models and
the SARS-CoV-2 haplotypes), so they share their colour scheme, the rendering of the
persistence contours, and the shading of the region excluded by Hamming geometry.

**Colour.** Each row family (JC, HB, and the SARS-CoV-2 data) is one cubehelix ramp, from
light at the earliest time layer to dark at the last. Sharing the lightness endpoints
across families keeps a given stage equally prominent whatever its hue.

**Persistence contours.** A birth-lifetime histogram on the integer lattice is smoothed
with a Gaussian kernel, restricted to the region allowed by Hamming geometry, interpolated
by a bivariate spline, and contoured at highest-density levels, i.e. at thresholds chosen
so that each contour encloses a given fraction of the total mass.

**The excluded region.** A degree-one class of a quadruple in a Hamming space of length L
born at ``b`` cannot persist beyond ``min(b, L - b)``, since its lifetime is bounded by
the sides of the underlying rectangle. :func:`shade_hamming_exclusion` shades the region
above this bound, so that it reads as excluded rather than unobserved.
"""

from __future__ import annotations

import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgba
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

__all__ = [
    "Y_FLOOR",
    "CONTOUR_MASS_LEVELS",
    "PERSISTENCE_YTICKS",
    "HYPERBOLICITY_XMIN",
    "FAMILY_CUBEHELIX",
    "CUBEHELIX_LIGHT",
    "CUBEHELIX_DARK",
    "family_palette",
    "lighten_color",
    "color_luminance",
    "support_mask",
    "density_grid",
    "dense_contour_grid",
    "shade_hamming_exclusion",
    "opaque_tint_ramp",
    "highest_density_thresholds",
    "plot_model_contours",
    "plot_observed_persistence_support",
    "padded_persistence_ymax",
]

# Probability floor for the log-scale panels: mass below this is not drawn.
Y_FLOOR = 1.0e-10
# Highest-density regions of the displayed nonzero persistence mass.
CONTOUR_MASS_LEVELS = (0.95, 0.80, 0.50)
PERSISTENCE_YTICKS = (0, 5, 10, 15)
HYPERBOLICITY_XMIN = -0.25

# JC blue, HB warm red. The SARS grid adds a green "data" family the same way.
FAMILY_CUBEHELIX = {
    "JC": dict(start=2.75, rot=0.05, hue=1.90),  # blue
    "HB": dict(start=0.80, rot=0.10, hue=1.60),  # red
}
CUBEHELIX_LIGHT = 0.86  # lightness of stage 0 (earliest / lightest)
CUBEHELIX_DARK = 0.25  # lightness of the last stage (t=inf limit / darkest)


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
def family_palette(
    n_stages: int,
    *,
    start: float,
    rot: float,
    hue: float,
    light: float = CUBEHELIX_LIGHT,
    dark: float = CUBEHELIX_DARK,
) -> list:
    """``n_stages`` evenly spaced colours, light (stage 0) to dark (last), for one hue family."""
    return sns.cubehelix_palette(
        max(n_stages, 1), start=start, rot=rot, hue=hue, light=light, dark=dark, reverse=False
    )


def lighten_color(color, amount: float = 0.55) -> tuple[float, float, float, float]:
    """Blend ``color`` toward white by ``amount``, preserving its alpha."""
    r, g, b, a = to_rgba(color)
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount, a)


def color_luminance(color) -> float:
    """Relative luminance, for choosing readable text over a coloured background."""
    rgb = mcolors.to_rgb(color)
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


# ---------------------------------------------------------------------------
# Persistence density on the birth-lifetime lattice
# ---------------------------------------------------------------------------
def support_mask(x_grid: np.ndarray, y_grid: np.ndarray, length: int) -> np.ndarray:
    """Where a degree-one class of a Hamming quadruple of length ``length`` can live.

    Birth ``x`` and lifetime ``y`` must satisfy ``0 < y <= min(x, length - x)``.
    """
    return (
        (x_grid > 0)
        & (x_grid <= length - 1)
        & (y_grid > 0)
        & (y_grid <= x_grid)
        & (y_grid <= length - x_grid)
    )


def density_grid(
    result: dict[str, object], length: int, ymax: int, samples: int, smoothing_sigma: float
) -> np.ndarray:
    """Smoothed birth-lifetime density on the integer lattice, masked to the support.

    ``result`` carries the ``birth``, ``lifetime`` and ``count`` arrays of one row.
    Counts are normalized by ``samples``, so the grid holds probability mass per cell.
    """
    grid = np.zeros((ymax + 1, length + 1), dtype=float)
    birth = result["birth"]
    lifetime = result["lifetime"]
    count = result["count"]
    assert isinstance(birth, np.ndarray)
    assert isinstance(lifetime, np.ndarray)
    assert isinstance(count, np.ndarray)
    if birth.size:
        visible = lifetime <= ymax
        grid[lifetime[visible], birth[visible]] = count[visible].astype(float) / float(samples)

    if smoothing_sigma > 0:
        grid = gaussian_filter(grid, sigma=float(smoothing_sigma), mode="constant", cval=0.0)

    x = np.arange(length + 1, dtype=float)
    y = np.arange(ymax + 1, dtype=float)
    x_grid, y_grid = np.meshgrid(x, y)
    return np.where(support_mask(x_grid, y_grid, length), grid, np.nan)


def dense_contour_grid(
    density: np.ndarray, length: int, ymax: int, contour_resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a smoothed lattice density for visually smoother contours."""
    contour_resolution = max(int(contour_resolution), 1)
    x_native = np.arange(length + 1, dtype=float)
    y_native = np.arange(ymax + 1, dtype=float)
    values = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)

    y_degree = min(3, len(y_native) - 1)
    x_degree = min(3, len(x_native) - 1)
    spline = RectBivariateSpline(y_native, x_native, values, kx=y_degree, ky=x_degree, s=0.0)

    x_dense = np.linspace(0.0, float(length), length * contour_resolution + 1)
    y_dense = np.linspace(0.0, float(ymax), ymax * contour_resolution + 1)
    dense = np.maximum(spline(y_dense, x_dense), 0.0)
    x_grid, y_grid = np.meshgrid(x_dense, y_dense)
    dense = np.where(support_mask(x_grid, y_grid, length), dense, np.nan)
    return x_grid, y_grid, dense


def highest_density_thresholds(density: np.ndarray, masses: tuple[float, ...]) -> np.ndarray:
    """Density levels enclosing the given fractions of the total mass.

    Sorts the cells by density and walks down until the cumulative mass reaches each
    target, so a contour drawn at the returned level encloses that share of the mass.
    """
    values = density[np.isfinite(density) & (density > 0.0)]
    if values.size == 0:
        return np.empty(0, dtype=float)

    total = float(np.sum(values))
    if total <= 0.0:
        return np.empty(0, dtype=float)

    sorted_values = np.sort(values)[::-1]
    cumulative_mass = np.cumsum(sorted_values) / total
    thresholds = []
    for mass in masses:
        if not 0.0 < mass < 1.0:
            raise ValueError(f"Contour mass levels must lie in (0, 1), got {mass}.")
        index = int(np.searchsorted(cumulative_mass, mass, side="left"))
        index = min(index, sorted_values.size - 1)
        thresholds.append(float(sorted_values[index]))
    return np.asarray(thresholds, dtype=float)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def shade_hamming_exclusion(
    ax: plt.Axes, length: int, xlim: tuple[float, float], ylim: tuple[float, float]
) -> None:
    """Shade the birth-lifetime region excluded by Hamming geometry, and draw its boundary."""
    birth = np.linspace(xlim[0], xlim[1], 600)
    raw_max_persistence = np.minimum(birth, length - birth)
    raw_max_persistence = np.where(birth <= length - 1, raw_max_persistence, 0.0)
    clipped_max_persistence = np.clip(raw_max_persistence, 0.0, ylim[1])
    ax.fill_between(
        birth,
        clipped_max_persistence,
        ylim[1],
        color="#969696",
        alpha=0.18,
        linewidth=0,
        zorder=0,
    )
    boundary_mask = (raw_max_persistence <= ylim[1]) & (birth <= length - 1)
    boundary_y = np.where(boundary_mask, raw_max_persistence, np.nan)
    ax.plot(birth, boundary_y, color="#636363", linestyle=":", linewidth=1.2, zorder=2)


def opaque_tint_ramp(color, lighten_amounts: list[float]) -> list[tuple[float, float, float, float]]:
    """Nested fill colours for the contour bands, without adding transparency."""
    return [lighten_color(color, amount) for amount in lighten_amounts]


def plot_model_contours(
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    density: np.ndarray,
    color,
    *,
    fill: bool = True,
    linestyle: str = "-",
    linewidth: float = 0.9,
    zorder: float = 4,
):
    """Draw filled highest-density contours at :data:`CONTOUR_MASS_LEVELS`."""
    finite_values = density[np.isfinite(density)]
    zmax = float(np.max(finite_values)) if finite_values.size else 0.0
    if zmax <= 0.0:
        return None

    contour_levels = np.unique(highest_density_thresholds(density, CONTOUR_MASS_LEVELS))
    contour_levels = contour_levels[(contour_levels > 0.0) & (contour_levels <= zmax)]
    if contour_levels.size == 0:
        return None

    if fill:
        fill_levels = np.concatenate([contour_levels, [zmax * 1.0001]])
        ax.contourf(
            x_grid,
            y_grid,
            density,
            levels=fill_levels,
            colors=opaque_tint_ramp(color, [0.55, 0.35, 0.15][: fill_levels.size - 1]),
            antialiased=True,
            zorder=zorder - 1,
        )
    return ax.contour(
        x_grid,
        y_grid,
        density,
        levels=contour_levels,
        colors=[color],
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=1.0,
        zorder=zorder,
    )


def plot_observed_persistence_support(
    ax: plt.Axes,
    birth: np.ndarray,
    lifetime: np.ndarray,
    count: np.ndarray,
    color,
    ymax: float,
) -> None:
    """Mark every occupied persistence cell with equal visual weight.

    Contours show where the mass is; these markers show where anything was observed at
    all, including cells far out in the tail whose mass is too small to contour.
    """
    birth = np.asarray(birth)
    lifetime = np.asarray(lifetime)
    count = np.asarray(count)
    occupied = (count > 0) & (lifetime > 0) & (lifetime <= ymax)
    if not np.any(occupied):
        return

    r, g, b, _ = to_rgba(color)
    edge_color = (0.68 * r, 0.68 * g, 0.68 * b)
    ax.scatter(
        birth[occupied],
        lifetime[occupied],
        marker="o",
        s=2.8,
        facecolors="none",
        edgecolors=[edge_color],
        linewidths=0.24,
        alpha=1.0,
        zorder=5,
    )


def padded_persistence_ymax(
    max_lifetime: float,
    *,
    margin_factor: float = 1.10,
    minimum: int = 10,
    tick_step: int = 2,
) -> int:
    """Pad an observed persistence maximum and round up to a clean tick."""
    if max_lifetime <= 0:
        return minimum
    padded = float(max_lifetime) * margin_factor
    rounded = tick_step * math.ceil(padded / tick_step)
    return max(minimum, int(rounded))
