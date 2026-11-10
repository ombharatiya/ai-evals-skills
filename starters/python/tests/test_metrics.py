import math

from evals.metrics import cohen_kappa, confusion, judge_report, wilson


def test_confusion_counts_with_pass_as_positive():
    human = ["PASS", "PASS", "FAIL", "FAIL", "FAIL"]
    judge = ["PASS", "FAIL", "PASS", "FAIL", "FAIL"]
    c = confusion(human, judge)
    assert (c.tp, c.fn, c.fp, c.tn) == (1, 1, 1, 2)


def test_report_names_rates_by_class():
    human = ["PASS"] * 8 + ["FAIL"] * 2
    judge = ["PASS"] * 8 + ["PASS", "FAIL"]
    r = judge_report(human, judge)
    assert r.pass_recall == 1.0
    assert r.fail_recall == 0.5
    assert r.agreement == 0.9


def test_always_pass_judge_has_high_agreement_and_zero_kappa():
    # The trap the guide warns about: 95% agreement from a judge that never says FAIL.
    human = ["PASS"] * 95 + ["FAIL"] * 5
    judge = ["PASS"] * 100
    r = judge_report(human, judge)
    assert r.agreement == 0.95
    assert r.fail_recall == 0.0
    assert r.kappa == 0.0


def test_kappa_matches_hand_computation():
    # observed 0.8; human PASS 0.6, judge PASS 0.6 -> expected 0.52 -> kappa 0.2833
    human = ["PASS"] * 6 + ["FAIL"] * 4
    judge = ["PASS"] * 5 + ["FAIL"] + ["PASS"] + ["FAIL"] * 3
    assert math.isclose(cohen_kappa(confusion(human, judge)), (0.8 - 0.52) / 0.48, rel_tol=1e-9)


def test_wilson_matches_the_guide_example():
    # The guide's confidence-interval chapter: Wilson on 7/10 is about [0.40, 0.89].
    lo, hi = wilson(7, 10)
    assert round(lo, 2) == 0.40 and round(hi, 2) == 0.89


def test_wilson_stays_inside_zero_one_at_the_edges():
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0.6 < lo < 0.8
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.4
    assert all(math.isnan(x) for x in wilson(0, 0))
