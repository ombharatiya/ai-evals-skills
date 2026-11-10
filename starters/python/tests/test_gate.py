from evals.gate import Delta, band, newcombe_paired, paired_delta


def d(delta, lo, hi):
    return Delta(n=100, baseline_rate=0.8, candidate_rate=0.8 + delta, delta=delta, ci=(lo, hi),
                 both_pass=0, baseline_only=0, candidate_only=0, both_fail=0)


def test_bands_follow_the_guide_gate():
    # Cases from the release-gates chapter's gate(): 1-point indifference, 2-point minor regression.
    assert band(d(0.03, 0.01, 0.05)) == "ship"              # sure improvement
    assert band(d(0.0, -0.008, 0.009)) == "ship"            # whole interval inside +-1 point
    assert band(d(0.0, -0.018, 0.018)) == "hold"            # contains zero, reaches past 1 point
    assert band(d(-0.012, -0.025, -0.001)) == "investigate"  # sure drop, point delta within 2 points
    assert band(d(-0.05, -0.08, -0.03)) == "block"           # sure drop past 2 points


def test_interval_stays_wide_when_the_runs_agree_on_every_row():
    # The review's case: one shared id used to give [0, 0] and ship.
    delta, lo, hi = newcombe_paired(1, 0, 0, 0)
    assert delta == 0 and lo < -0.5 and hi > 0.5
    assert band(Delta(1, 1, 1, delta, (lo, hi), 1, 0, 0, 0)) == "hold"
    _, lo, hi = newcombe_paired(12, 0, 0, 0)
    assert lo < -0.2 and hi > 0.2


def test_paired_delta_detects_a_clear_regression():
    base = {str(i): True for i in range(200)}
    cand = {str(i): i >= 40 for i in range(200)}  # 40 rows newly failing
    r = paired_delta(base, cand)
    assert round(r.delta, 2) == -0.20 and r.baseline_only == 40
    assert r.ci[1] < 0
    assert band(r) == "block"


def test_paired_delta_uses_only_shared_ids():
    base = {"a": True, "b": False}
    cand = {"a": True, "b": True, "c": False}
    r = paired_delta(base, cand)
    assert r.n == 2 and r.candidate_only == 1


def test_resolution_says_how_many_rows_a_window_needs():
    from evals.gate import no_change_halfwidth, rows_for_indifference

    # The site review's numbers: at a 76% pass rate, 50 identical rows give about +-5.3 points.
    assert round(no_change_halfwidth(50, 0.76), 4) == 0.0529
    assert rows_for_indifference(0.76, 0.01) == 278
    assert rows_for_indifference(0.97, 0.01) == 408
    assert no_change_halfwidth(278, 0.76) <= 0.01 < no_change_halfwidth(277, 0.76)
