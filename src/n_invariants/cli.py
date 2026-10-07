"""Command line entry point: ``n-invariants``.

The subcommand ``bound`` computes sample sizes with :func:`n_invariants.sample_size`.
"""

from __future__ import annotations

import argparse

from .bounds import sample_size

ARTICLE_INVARIANTS = """\
the article's invariants (Prop 4.8 and Section 8.4):
  pairwise distance                  -n 2  --lipschitz 1  --lambda 1
  four-point hyperbolicity deficit   -n 4  --lipschitz 2  --lambda 1
  (4,1)-persistence                  -n 4  --lipschitz 1  --lambda 1
"""


def _bound(args: argparse.Namespace) -> int:
    lam = args.lip if args.lam is None else args.lam

    if args.lip == 1 and lam == 1:
        n_samples = int(sample_size(args.epsilon, args.alpha, args.n, args.p,
                                 method=args.method))
        title = "the curvature measure mu_n (Corollary 6.3)"
        rows = [("n", f"{args.n}"), ("p", f"{args.p:g}")]
        guarantee = (f"W_p(mu_(N,n), mu_n) <= {args.epsilon:g} diam(X), and so\n"
                     f"  W_p(nu_(F,N), nu_F) <= {args.epsilon:g} Lip(F) diam(X) "
                     "for every Lipschitz n-invariant F.")
    else:
        n_samples = int(sample_size(args.epsilon, args.alpha, args.n, args.p,
                                 lip=args.lip, lam=lam, method=args.method))
        title = "one Lipschitz n-invariant F (Corollary 6.5)"
        rows = [("n", f"{args.n}"), ("p", f"{args.p:g}"),
                ("Lip(F)", f"{args.lip:g}"), ("lambda", f"{lam:g}")]
        guarantee = f"W_p(nu_(F,N), nu_F) <= {args.epsilon:g} diam(X)."

    rows += [("eps", f"{args.epsilon:g}  (in units of diam(X))"),
             ("alpha", f"{args.alpha:g}"),
             ("method", args.method)]

    print()
    print(f"Sample size for {title}")
    print()
    for name, value in rows:
        print(f"  {name:<6} = {value}")
    print()
    print(f"  {'N':<6} = {n_samples:,}")
    print()
    print(f"  With probability >= {args.alpha:g}, {guarantee}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="n-invariants",
        description="Metric n-point invariants of large finite metric measure spaces.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bound = subparsers.add_parser(
        "bound",
        help="Number of n-tuples needed to estimate an n-point invariant distribution.",
        description=(
            "Smallest N such that, with probability at least alpha, the empirical measure "
            "of N sampled n-tuples is within eps diam(X) of its target in W_p. By default "
            "the target is the curvature measure mu_n, which covers every Lipschitz "
            "invariant at once. Pass --lipschitz (and optionally --lambda) for one invariant. "
            "The result does not depend on the number of points of X."
        ),
        epilog=ARTICLE_INVARIANTS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bound.add_argument("-n", "--n", type=int, default=4, dest="n",
                       help="Points per sampled tuple (default: 4).")
    bound.add_argument("-p", "--p", type=float, default=2.0, dest="p",
                       help="Order of the Wasserstein distance (default: 2).")
    bound.add_argument("-e", "--epsilon", type=float, default=0.1,
                       help="Target accuracy, in units of diam(X) (default: 0.1).")
    bound.add_argument("-a", "--alpha", type=float, default=0.95,
                       help="Confidence (default: 0.95).")
    bound.add_argument("-L", "--lipschitz", type=float, default=1.0, dest="lip",
                       help="Lipschitz constant of the invariant, with respect to the "
                            "supremum norm on distance matrices (default: 1).")
    bound.add_argument("--lambda", type=float, default=None, dest="lam",
                       help="Range constant: diam F(K_n(X)) <= lambda diam(X) "
                            "(default: the Lipschitz constant, which is always valid).")
    bound.add_argument("--method", choices=("WB", "MD", "Markov"), default="WB",
                       help="Concentration inequality (default: WB, the sharpest).")
    bound.set_defaults(func=_bound)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
