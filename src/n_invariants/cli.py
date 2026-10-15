"""Command line entry point: ``n-invariants``.

``bound``
    Sample sizes for the curvature measure or one Lipschitz n-invariant, with
    :func:`n_invariants.sample_size` (Cor 6.3, Cor 6.5).
``bound-real-valued``
    Sample sizes for one real-valued n-invariant, with
    :func:`n_invariants.sample_size_real_valued` (Prop 6.6).
``accuracy``
    The accuracy a given number of samples guarantees for the article's n-invariants, with
    :func:`n_invariants.guaranteed_accuracy`.
"""

from __future__ import annotations

import argparse

from .bounds import guaranteed_accuracy, sample_size, sample_size_real_valued
from .invariants import HYPERBOLICITY_DEFICIT, PAIRWISE_DISTANCE, VR_PERSISTENCE_4_1

ARTICLE_INVARIANTS = {F.name: F for F in (PAIRWISE_DISTANCE, HYPERBOLICITY_DEFICIT, VR_PERSISTENCE_4_1)}

ARTICLE_CONSTANTS = """\
the article's n-invariants (Prop 4.9 and Section 8.4):
  pairwise distance                  -n 2  --lipschitz 1  --lambda 1  real-valued
  four-point hyperbolicity deficit   -n 4  --lipschitz 2  --lambda 1  real-valued
  (4,1)-persistence                  -n 4  --lipschitz 1  --lambda 1
"""

UNION_BOUND = """\
Each accuracy holds with probability at least alpha on its own. For m of them to hold
together with probability beta, use alpha = 1 - (1 - beta) / m (union bound).
"""


def _print_rows(title, rows, n_samples, guarantee, alpha):
    print()
    print(title)
    print()
    for name, value in rows:
        print(f"  {name:<6} = {value}")
    print()
    print(f"  {'N':<6} = {n_samples:,}")
    print()
    print(f"  With probability >= {alpha:g}, {guarantee}")
    print()


def _bound(args: argparse.Namespace) -> int:
    lam = args.lip if args.lam is None else args.lam

    if args.lip == 1 and lam == 1:
        n_samples = int(sample_size(args.epsilon, args.alpha, args.n, args.p,
                                    method=args.method))
        title = "Sample size for the curvature measure mu_n (Corollary 6.3)"
        rows = [("n", f"{args.n}"), ("p", f"{args.p:g}")]
        guarantee = (f"W_p(mu_(N,n), mu_n) <= {args.epsilon:g} diam(X), and so\n"
                     f"  W_p(nu_(F,N), nu_F) <= {args.epsilon:g} Lip(F) diam(X) "
                     "for every Lipschitz n-invariant F.")
    else:
        n_samples = int(sample_size(args.epsilon, args.alpha, args.n, args.p,
                                    lip=args.lip, lam=lam, method=args.method))
        title = "Sample size for one Lipschitz n-invariant F (Corollary 6.5)"
        rows = [("n", f"{args.n}"), ("p", f"{args.p:g}"),
                ("Lip(F)", f"{args.lip:g}"), ("lambda", f"{lam:g}")]
        guarantee = f"W_p(nu_(F,N), nu_F) <= {args.epsilon:g} diam(X)."

    rows += [("eps", f"{args.epsilon:g}  (in units of diam(X))"),
             ("alpha", f"{args.alpha:g}"),
             ("method", args.method)]
    _print_rows(title, rows, n_samples, guarantee, args.alpha)
    return 0


def _bound_real_valued(args: argparse.Namespace) -> int:
    n_samples = sample_size_real_valued(args.epsilon, args.alpha, args.p, lam=args.lam)
    rows = [("p", f"{args.p:g}"), ("lambda", f"{args.lam:g}"),
            ("eps", f"{args.epsilon:g}  (in units of diam(X))"),
            ("alpha", f"{args.alpha:g}")]
    guarantee = f"W_p(nu_(F,N), nu_F) <= {args.epsilon:g} diam(X)."
    _print_rows("Sample size for one real-valued n-invariant F (Proposition 6.6)",
                rows, n_samples, guarantee, args.alpha)
    return 0


def _accuracy(args: argparse.Namespace) -> int:
    names = args.invariants or list(ARTICLE_INVARIANTS)
    result = guaranteed_accuracy(args.n_samples, [ARTICLE_INVARIANTS[name] for name in names],
                     args.p, alpha=args.alpha)

    print()
    print(f"Accuracy guaranteed by N = {args.n_samples:,} sampled tuples")
    print()
    print(f"  {'p':<6} = {args.p:g}")
    print(f"  {'alpha':<6} = {args.alpha:g}  (for each n-invariant on its own)")
    print()
    print(f"  {'n-invariant':<22} {'n':>2}  {'Lip':>3}  {'lambda':>6}  {'values':<8}  "
          f"{'accuracy':>9}  by")
    for name in names:
        F, c = ARTICLE_INVARIANTS[name], result[name]
        values = "real" if F.real_valued else "diagrams"
        print(f"  {name:<22} {F.n:>2}  {F.lip:>3g}  {F.lam:>6g}  {values:<8}  "
              f"{c.accuracy:>9.3g}  {c.statement}")
    print()
    print(f"  With probability >= {args.alpha:g}, W_p(nu_(F,N), nu_F) <= accuracy * diam(X),")
    print("  for each n-invariant F on its own.")
    print()
    return 0


def _add_order_and_confidence(parser):
    parser.add_argument("-p", "--p", type=float, default=2.0, dest="p",
                        help="Order of the Wasserstein distance (default: 2).")
    parser.add_argument("-a", "--alpha", type=float, default=0.999,
                        help="Confidence (default: 0.999).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="n-invariants",
        description="Metric n-point invariants of large finite metric measure spaces.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bound = subparsers.add_parser(
        "bound",
        help="Number of n-tuples needed for the curvature measure or one Lipschitz n-invariant.",
        description=(
            "Smallest N such that, with probability at least alpha, the empirical measure "
            "of N sampled n-tuples is within eps diam(X) of its target in W_p. By default "
            "the target is the curvature measure mu_n, which covers every Lipschitz "
            "n-invariant at once. Pass --lipschitz (and optionally --lambda) for one "
            "n-invariant. The result does not depend on the number of points of X."
        ),
        epilog=ARTICLE_CONSTANTS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bound.add_argument("-n", "--n", type=int, default=4, dest="n",
                       help="Points per sampled tuple (default: 4).")
    _add_order_and_confidence(bound)
    bound.add_argument("-e", "--epsilon", type=float, default=0.1,
                       help="Target accuracy, in units of diam(X) (default: 0.1).")
    bound.add_argument("-L", "--lipschitz", type=float, default=1.0, dest="lip",
                       help="Lipschitz constant of the n-invariant, with respect to the "
                            "supremum norm on distance matrices (default: 1).")
    bound.add_argument("--lambda", type=float, default=None, dest="lam",
                       help="Range constant: diam F(K_n(X)) <= lambda diam(X) "
                            "(default: the Lipschitz constant, which is always valid).")
    bound.add_argument("--method", choices=("WB", "MD", "Markov"), default="WB",
                       help="Concentration inequality (default: WB, the sharpest).")
    bound.set_defaults(func=_bound)

    bound_real = subparsers.add_parser(
        "bound-real-valued",
        help="Number of tuples needed for one real-valued n-invariant.",
        description=(
            "Smallest N such that, with probability at least alpha, the empirical "
            "distribution of a real-valued n-invariant F on N sampled tuples is within "
            "eps diam(X) of nu_F in W_p (Proposition 6.6). It depends on F only through "
            "its range constant, and not on n."
        ),
        epilog=ARTICLE_CONSTANTS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_order_and_confidence(bound_real)
    bound_real.add_argument("-e", "--epsilon", type=float, default=0.1,
                            help="Target accuracy, in units of diam(X) (default: 0.1).")
    bound_real.add_argument("--lambda", type=float, default=1.0, dest="lam",
                            help="Range constant: diam F(K_n(X)) <= lambda diam(X) (default: 1).")
    bound_real.set_defaults(func=_bound_real_valued)

    accuracy_parser = subparsers.add_parser(
        "accuracy",
        help="Accuracy that N sampled tuples guarantee for the article's n-invariants.",
        description=(
            "For each n-invariant, the sharpest accuracy that N sampled tuples guarantee "
            "with probability at least alpha: through the curvature measure "
            "(Corollaries 6.3 and 6.5) or, for real-valued ones, Proposition 6.6."
        ),
        epilog=UNION_BOUND,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    accuracy_parser.add_argument("-N", "--n-samples", type=lambda s: int(float(s)), required=True,
                                dest="n_samples", help="Number of sampled tuples, e.g. 3.4e9.")
    _add_order_and_confidence(accuracy_parser)
    accuracy_parser.add_argument("invariants", nargs="*", choices=list(ARTICLE_INVARIANTS),
                                metavar="INVARIANT",
                                help=f"Any of {', '.join(ARTICLE_INVARIANTS)} (default: all).")
    accuracy_parser.set_defaults(func=_accuracy)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
