"""The command line: sample sizes and the accuracy of a given sample."""

import pytest

from n_invariants.cli import main


def test_bound_for_the_curvature_measure(capsys):
    main(["bound", "-n", "4", "-p", "1", "-e", "0.1", "-a", str(1 - 0.05 / 15)])
    assert "53,433,249" in capsys.readouterr().out


def test_bound_for_a_real_valued_n_invariant(capsys):
    main(["bound-real-valued", "-p", "1", "-e", "0.1", "-a", "0.95"])
    out = capsys.readouterr().out
    assert "Proposition 6.6" in out
    assert "1,324" in out


def test_accuracy_of_the_sars_sample(capsys):
    main(["accuracy", "-N", "53433249", "-p", "1", "-a", str(1 - 0.05 / 15)])
    lines = capsys.readouterr().out.splitlines()
    rows = {line.split()[0]: line.split()[-3:] for line in lines if line.strip().endswith(("6.6", "6.3"))}
    assert rows == {
        "pairwise_distance": ["0.000561", "Prop", "6.6"],
        "hyperbolicity_deficit": ["0.000561", "Prop", "6.6"],
        "vr_persistence_4_1": ["0.1", "Cor", "6.3"],
    }


def test_accuracy_accepts_a_subset_and_scientific_notation(capsys):
    main(["accuracy", "-N", "1e6", "hyperbolicity_deficit"])
    out = capsys.readouterr().out
    assert "N = 1,000,000" in out
    assert "hyperbolicity_deficit" in out and "pairwise_distance" not in out


def test_accuracy_rejects_unknown_n_invariants():
    with pytest.raises(SystemExit):
        main(["accuracy", "-N", "100", "curvature"])
