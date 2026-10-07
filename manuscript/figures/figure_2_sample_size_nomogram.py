"""Figure 2: sample sizes for estimating invariant distributions (Section 6).

The sample count N_{lambda=1} of Corollary 6.5 measures the accuracy in units of diam(X),
so the diameter drops out. Given the invariant (n, p and the Lipschitz constant L), it
depends only on the relative accuracy epsilon and the confidence alpha. The range constant
lambda = 1 holds for all four invariants shown.

The figure shows log10(N) over the (confidence, relative accuracy) plane, one panel per
invariant. Relative accuracy, which determines most of the cost, is on the y-axis.
Confidence is on the x-axis; it has little effect on N, since the approximation term
dominates the confidence term for n >= 3. The colours run from blue and green for
feasible sample counts through yellow to red for prohibitive ones.

Sample counts agree with ``n_invariants.sample_size`` (method 'WB', lam=1). They are solved
in log10(N), so they remain valid beyond the int64 range of ``sample_size``.

Run from the repository root::

    uv run python manuscript/figures/figure_2_sample_size_nomogram.py
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from n_invariants import bounds

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
P_VALUE = 1.0
DOABLE_SAMPLE_LIMIT = 1e10         # one-day sampling budget; onset of the yellow warning band
EPS_MIN, EPS_MAX = 1.0e-3, 1.0     # relative-accuracy (y) axis; interesting region: small eps
OMA_MIN, OMA_MAX = 1.0e-3, 1.0      # 1 - alpha (x) axis, i.e. alpha in [0, 0.999]

# Colour ramp with operational breakpoints in log10(N): violet/blue on [0, 4], green on [4, 8],
# yellow on [8, 10], orange on [10, 14], and increasingly dark red on [14, 20]. Positions below
# are in the output coordinates of PiecewiseNorm (0--10 maps to 0--0.75; 10--20 to 0.75--1).
GREEN_EDGE = 0.75
LOG_N_CEIL = 20.0
TRAFFIC_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "sample_cost",
    [(0.00, "#280044"),   # log10(N) = 0
     (0.15, "#243b7a"),   # log10(N) = 2
     (0.30, "#16858c"),   # log10(N) = 4
     (0.40, "#269f67"),   # settle smoothly into the green band
     (0.54, "#58b957"),   # green through most of the 1e4--1e8 interval
     (0.60, "#e6d72f"),   # log10(N) = 8: yellow begins
     (0.75, "#f1b52c"),   # log10(N) = 10: orange begins
     (0.85, "#d7301f"),   # log10(N) = 14: red begins
     (1.00, "#67000d")],  # log10(N) = 20: dark red
    N=512,
)

# The invariants, each with its curvature index n and Lipschitz constant L.
INVARIANTS = (
    {"title": "Pairwise distance",          "n": 2, "L": 1.0},
    {"title": "Three-point Gromov product", "n": 3, "L": 1.0},
    {"title": "Four-point persistence",     "n": 4, "L": 1.0},
    {"title": "Four-point hyperbolicity",   "n": 4, "L": 2.0},
)
LABEL_N = (1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18)


class PiecewiseNorm(mcolors.Normalize):
    """Linear on [vmin, x0] -> [0, y0], then linear on [x0, vmax] -> [y0, 1]."""

    def __init__(self, x0: float, y0: float, vmin: float, vmax: float):
        super().__init__(vmin=vmin, vmax=vmax, clip=False)
        self.x0 = x0
        self.y0 = y0

    def __call__(self, value, clip=None):
        v = np.ma.asarray(value, dtype=float)
        lower = self.y0 * (v - self.vmin) / (self.x0 - self.vmin)
        upper = self.y0 + (1.0 - self.y0) * (v - self.x0) / (self.vmax - self.x0)
        out = np.where(v <= self.x0, lower, upper)
        return np.ma.masked_array(np.clip(out, 0.0, 1.0), np.ma.getmaskarray(v))


# ---------------------------------------------------------------------------
# Weed--Bach sample bound, solved in log10(N) space (overflow-safe).
# ---------------------------------------------------------------------------
def log10_n_min_wb(epsilon: float, alpha: float, n: int, p: float = P_VALUE, L: float = 1.0) -> float:
    """log10 of the smallest N with W_p(nu_{F,N}, nu_F) < epsilon diam(X) at confidence alpha.

    Here F is an L-Lipschitz n-invariant with diam F(K_n(X)) <= diam(X).

    Solves  L^p C N^{-e} + sqrt(-0.5 ln(1-alpha)) N^{-1/2} = epsilon^p  for N, the defining
    equation of Corollary 6.5 with lambda = 1 (n_invariants.sample_size with lam=1),
    in u = log10(N).
    """
    d = n * (n - 1) / 2.0
    C = float(bounds.fournier_constant(n, p))          # unit-diameter curvature constant
    e = p / d if p < d / 2.0 else 0.5
    s = math.sqrt(-0.5 * math.log1p(-alpha))
    rhs = epsilon ** p

    def f(u: float) -> float:
        return (L ** p) * C * 10.0 ** (-e * u) + s * 10.0 ** (-0.5 * u) - rhs

    if f(0.0) <= 0.0:                                  # a single sample already suffices
        return 0.0
    return brentq(f, 0.0, 400.0, xtol=1e-10, rtol=1e-12, maxiter=200)


# ---------------------------------------------------------------------------
# Nomogram
# ---------------------------------------------------------------------------
def make_nomogram(out_path: Path) -> None:
    eps = np.logspace(math.log10(EPS_MIN), math.log10(EPS_MAX), 220)              # y: accuracy
    one_minus_alpha = np.logspace(math.log10(OMA_MIN), math.log10(OMA_MAX), 120)  # x: 1 - alpha

    norm = PiecewiseNorm(math.log10(DOABLE_SAMPLE_LIMIT), GREEN_EDGE, 0.0, LOG_N_CEIL)
    levels = np.linspace(0.0, LOG_N_CEIL, 321)
    label_levels = [math.log10(v) for v in LABEL_N]
    label_col = int(np.argmin(np.abs(one_minus_alpha - 3.0e-3)))   # anchor labels at alpha ~ 0.997
    stroke = [pe.withStroke(linewidth=1.6, foreground="#333333")]

    fig, axes = plt.subplots(1, len(INVARIANTS), figsize=(16.5, 4.6),
                             sharey=True, layout="constrained")
    fig.set_constrained_layout_pads(wspace=0.045)
    X, Y = np.meshgrid(one_minus_alpha, eps)

    for panel, (ax, inv) in enumerate(zip(axes, INVARIANTS)):
        n, L = inv["n"], inv["L"]
        logN = np.array([[log10_n_min_wb(float(e), float(1.0 - oma), n, P_VALUE, L)
                          for oma in one_minus_alpha] for e in eps])
        panel_label_levels = label_levels + ([LOG_N_CEIL] if panel == len(INVARIANTS) - 1 else [])

        # Rasterize only the dense fill in PDF output to avoid hairline seams between polygons;
        # contours, labels, and axes remain vector graphics above the rasterization threshold.
        ax.set_rasterization_zorder(0)
        mesh = ax.contourf(X, Y, logN, levels=levels, norm=norm, cmap=TRAFFIC_CMAP,
                           extend="max", zorder=-1)
        ax.contour(X, Y, logN, levels=panel_label_levels,
                   colors="white", linewidths=0.8, alpha=0.85)

        # Solid contours + manually placed upright labels (clabel's inline gap/angle misbehave
        # on the inverted log axes). One label per contour, stacked at a fixed confidence column.
        profile = logN[:, label_col]
        for lv in panel_label_levels:
            if not (profile.min() < lv < profile.max()):
                continue
            eps_lv = float(np.interp(lv, profile[::-1], eps[::-1]))
            ax.text(one_minus_alpha[label_col], eps_lv, rf"$10^{{{int(round(lv))}}}$",
                    color="white", fontsize=7.5, ha="center", va="center",
                    path_effects=stroke, zorder=5)

        ax.set_title(rf"$n={n},\ L={L:g}$", fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(OMA_MAX, OMA_MIN)                  # confidence increases to the right
        ax.set_ylim(EPS_MAX, EPS_MIN)                  # finer accuracy (small eps) at the top
        ax.set_xticks([1.0, 1e-1, 1e-2, 1e-3])
        ax.set_xticklabels([r"$0$", r"$0.9$", r"$0.99$", r"$0.999$"])
        ax.get_xticklabels()[-1].set_ha("right")
        ax.set_xlabel(r"confidence $\alpha$")

    axes[0].set_ylabel(r"relative accuracy $\varepsilon$")
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.02, pad=0.01,
                        ticks=list(range(0, int(LOG_N_CEIL) + 1, 2)))
    cbar.set_label(r"required samples $\log_{10} N$")

    # Constrained layout settles over two draws. The manuscript figure was saved after a PNG
    # render at this dpi, so draw one into memory first to reproduce it exactly.
    fig.savefig(io.BytesIO(), format="png", dpi=220)
    # No CreationDate in the PDF, so a re-run reproduces the committed file byte for byte.
    fig.savefig(out_path.with_suffix(".pdf"), dpi=220, metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    # Correctness guard: the log-space solver must reproduce n_invariants.bounds.sample_size.
    for eps, alpha, n, L in [(0.1, 0.95, 4, 1.0), (0.1, 0.95, 2, 1.0), (0.3, 0.99, 4, 2.0)]:
        mine = 10.0 ** log10_n_min_wb(eps, alpha, n, P_VALUE, L)
        ref = float(bounds.sample_size(eps, alpha, n, P_VALUE, lip=L, lam=1))
        assert abs(mine - ref) / ref < 1e-3, (eps, alpha, n, L, mine, ref)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_size_nomogram"
    make_nomogram(out_path)
    print(f"Wrote {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
