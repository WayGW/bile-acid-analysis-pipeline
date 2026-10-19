"""Regression tests for S2 (Games-Howell), S3 (post-hoc follows main test),
S15 (zero-variance group) and S5 (two-group significance brackets).

Reference values for Games-Howell were produced with
pingouin 0.7.0 ``pairwise_gameshowell`` on GH_DATA; if pingouin is installed
the test also compares live.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from modules import statistical_tests as st
from modules.statistical_tests import StatisticalAnalyzer, significant_pairs

TT = st.TestType


def _frame(groups):
    return pd.DataFrame([(g, v) for g, vs in groups.items() for v in vs], columns=["g", "y"])


# ---------------------------------------------------------------- S2
GH_DATA = {
    "A": [4.1, 5.3, 3.8, 4.9, 5.6, 4.4, 5.0],
    "B": [6.2, 9.8, 4.1, 11.5, 7.7, 8.9],
    "C": [12.4, 3.1, 18.9, 9.6, 15.2, 6.8, 21.7, 10.3],
    "D": [5.9, 6.4, 6.1, 5.2, 6.8, 6.0],
}
# (T, df, p) from pingouin.pairwise_gameshowell
GH_REF = {
    ("A", "B"): (-2.98963016852, 5.52486322594, 0.093841304606),
    ("A", "C"): (-3.42600750007, 7.17868724181, 0.0417571292134),
    ("A", "D"): (-4.06054101297, 10.9905782396, 0.00869195524202),
    ("B", "C"): (-1.73303351614, 9.99916735782, 0.357437059853),
    ("B", "D"): (1.78870023464, 5.41078771412, 0.369593740634),
    ("C", "D"): (2.82032309461, 7.14025249959, 0.0933655232306),
}


def test_games_howell_matches_pingouin_reference_values():
    res = StatisticalAnalyzer()._games_howell(_frame(GH_DATA), "y", "g").pairwise_results
    assert len(res) == len(GH_REF)
    for _, row in res.iterrows():
        T, df, p = GH_REF[(row.group1, row.group2)]
        assert row.t_stat == pytest.approx(T, rel=1e-9)
        assert row.df == pytest.approx(df, rel=1e-9)
        assert row.pvalue_adj == pytest.approx(p, rel=1e-6)
        # no extra FDR on top of the studentized range
        assert row.significant == (p < 0.05)
        # 'pvalue' keeps the unadjusted Welch t p for reference
        assert row.pvalue == pytest.approx(2 * stats.t.sf(abs(T), df), rel=1e-9)


def test_games_howell_matches_pingouin_live():
    pg = pytest.importorskip("pingouin")
    d = _frame(GH_DATA)
    ref = pg.pairwise_gameshowell(data=d, dv="y", between="g")
    pcol = "pval" if "pval" in ref.columns else "p-tukey"
    ref_p = {(a, b): p for a, b, p in zip(ref["A"], ref["B"], ref[pcol])}
    res = StatisticalAnalyzer()._games_howell(d, "y", "g").pairwise_results
    for _, row in res.iterrows():
        assert row.pvalue_adj == pytest.approx(ref_p[(row.group1, row.group2)], rel=1e-6)


def test_analyze_welch_route_uses_games_howell():
    r = StatisticalAnalyzer().analyze(_frame(GH_DATA), "y", "g")
    assert r.main_test.test_type == TT.ANOVA_WELCH
    assert r.posthoc_test is not None and r.posthoc_test.test_type == TT.GAMES_HOWELL


# ---------------------------------------------------------------- S3
S3_DATA = {  # each group passes Shapiro, but n = 4 < MIN_PARAMETRIC_N
    "X": [1.0, 2.1, 2.9, 4.2],
    "Y": [11.0, 12.2, 12.8, 14.1],
    "Z": [21.0, 22.1, 23.0, 23.9],
}


def test_small_n_kruskal_is_followed_by_dunn():
    r = StatisticalAnalyzer().analyze(_frame(S3_DATA), "y", "g")
    assert r.assumptions.overall_normality
    assert r.main_test.test_type == TT.KRUSKAL_WALLIS
    assert r.main_test.significant
    # before S3 this was TUKEY_HSD
    assert r.posthoc_test.test_type == TT.DUNN


@pytest.mark.parametrize("rec, expected", [
    (TT.KRUSKAL_WALLIS, TT.DUNN),
    (TT.ANOVA_ONEWAY, TT.TUKEY_HSD),
    (TT.ANOVA_WELCH, TT.GAMES_HOWELL),
])
def test_posthoc_follows_recommended_test(rec, expected):
    a = StatisticalAnalyzer()
    assumptions = a.check_assumptions(_frame(GH_DATA), "y", "g")
    assumptions.recommended_test = rec
    assert a.run_posthoc(_frame(GH_DATA), "y", "g", assumptions).test_type == expected


# ---------------------------------------------------------------- S15
S15_DATA = {  # Ctrl/Trt normal with unequal spread; LOD group fully LOD-replaced
    "Ctrl": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    "Trt": [4.5, 5.0, 5.5, 6.0, 6.5, 7.0],
    "LOD": [0.3] * 6,
}


@pytest.mark.parametrize("log10", [False, True])
def test_zero_variance_group_routes_to_kruskal(log10):
    d = _frame(S15_DATA)
    if log10:
        d["y"] = np.log10(d["y"])  # previously: spurious Welch F ~ 1570, p ~ 1e-9
    r = StatisticalAnalyzer().analyze(d, "y", "g")
    assert r.assumptions.constant_groups == ["LOD"]
    assert r.main_test.test_type == TT.KRUSKAL_WALLIS
    ref = stats.kruskal(*[d.loc[d.g == g, "y"].values for g in S15_DATA])
    assert r.main_test.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.main_test.pvalue == pytest.approx(ref.pvalue, rel=1e-12)
    assert r.posthoc_test.test_type == TT.DUNN


@pytest.mark.parametrize("log10", [False, True])
def test_zero_variance_group_two_groups_routes_to_mann_whitney(log10):
    # Ctrl is normal with n >= 5; previously this went to Welch t, which
    # effectively compares Ctrl's mean with the constant LOD value.
    groups = {"Ctrl": S15_DATA["Ctrl"], "LOD": S15_DATA["LOD"]}
    d = _frame(groups)
    if log10:
        d["y"] = np.log10(d["y"])
    r = StatisticalAnalyzer().analyze(d, "y", "g")
    assert r.assumptions.constant_groups == ["LOD"]
    assert r.main_test.test_type == TT.MANN_WHITNEY
    ref = stats.mannwhitneyu(d.loc[d.g == "Ctrl", "y"], d.loc[d.g == "LOD", "y"], alternative="two-sided")
    assert r.main_test.pvalue == pytest.approx(ref.pvalue, rel=1e-12)
    assert r.main_test.effect_size_type == "rank_biserial"


def test_no_constant_groups_recorded_for_ordinary_data():
    a = StatisticalAnalyzer().check_assumptions(_frame(GH_DATA), "y", "g")
    assert a.constant_groups == []


# ---------------------------------------------------------------- S5
def test_two_group_significant_main_test_yields_bracket():
    d = _frame({"Ctrl": [1.0, 1.3, 0.8, 1.1, 0.9, 1.2], "Trt": [2.4, 2.9, 2.2, 2.7, 3.1, 2.5]})
    r = StatisticalAnalyzer().analyze(d, "y", "g")
    assert r.posthoc_test is None
    assert r.main_test.pvalue < 0.001
    assert significant_pairs(r) == [("Ctrl", "Trt", "***")]
    assert significant_pairs(r, max_pairs=0) == []


def test_two_group_nonsignificant_main_test_yields_no_bracket():
    d = _frame({"Ctrl": [1.0, 1.3, 0.8, 1.1, 0.9, 1.2], "Trt": [1.1, 0.9, 1.2, 1.0, 1.3, 0.8]})
    r = StatisticalAnalyzer().analyze(d, "y", "g")
    assert not r.main_test.significant
    assert significant_pairs(r) == []


def test_multigroup_pairs_come_from_posthoc_table():
    r = StatisticalAnalyzer().analyze(_frame(GH_DATA), "y", "g")
    tab = r.posthoc_test.pairwise_results
    expected = tab[tab.significant].sort_values("pvalue_adj")
    got = significant_pairs(r)
    assert [(a, b) for a, b, _ in got] == list(zip(expected.group1, expected.group2))
    assert significant_pairs(r, max_pairs=1) == got[:1]


def test_significant_pairs_handles_none():
    assert significant_pairs(None) == []
