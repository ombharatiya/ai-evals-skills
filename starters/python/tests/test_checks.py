from evals.checks import run_checks


def row(**kw):
    base = {"id": "x", "query": "q", "response": "Refunds take 5 to 7 business days."}
    base.update(kw)
    return base


def by_name(results):
    return {r.name: r for r in results}


def test_only_not_empty_runs_without_expectations():
    assert [r.name for r in run_checks(row())] == ["not_empty"]


def test_checks_read_expectations_from_the_row():
    results = by_name(run_checks(row(must_include=["5 to 7"], must_not_include=["instant"], max_words=10)))
    assert all(r.passed for r in results.values())
    results = by_name(run_checks(row(must_include=["30 days"], must_not_include=["business"], max_words=3)))
    assert not results["must_include"].passed and "30 days" in results["must_include"].detail
    assert not results["must_not_include"].passed
    assert not results["max_words"].passed


def test_empty_response_fails():
    assert not by_name(run_checks(row(response="   ")))["not_empty"].passed
