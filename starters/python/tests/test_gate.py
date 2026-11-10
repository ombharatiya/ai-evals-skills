from evals.gate import Delta, band, paired_delta


def d(lo, hi):
    return Delta(n=100, baseline_rate=0.8, candidate_rate=0.8, delta=(lo + hi) / 2, ci=(lo, hi))


def test_bands_follow_the_release_gate_chapter():
    assert band(d(0.01, 0.05)) == "ship"           # sure improvement
    assert band(d(-0.01, 0.015)) == "ship"         # inside the indifference window
    assert band(d(-0.04, 0.03)) == "hold"          # contains zero, reaches outside
    assert band(d(-0.015, -0.005)) == "investigate"  # sure but small regression
    assert band(d(-0.08, -0.03)) == "block"        # sure and past the window


def test_paired_delta_on_identical_runs_is_zero():
    runs = {str(i): i % 3 != 0 for i in range(90)}
    r = paired_delta(runs, dict(runs))
    assert r.delta == 0 and r.ci == (0.0, 0.0)


def test_paired_delta_detects_a_clear_regression():
    base = {str(i): True for i in range(200)}
    cand = {str(i): i >= 40 for i in range(200)}  # 40 rows newly failing
    r = paired_delta(base, cand)
    assert round(r.delta, 2) == -0.20
    assert r.ci[1] < 0
    assert band(r) == "block"
