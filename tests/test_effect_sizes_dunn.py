"""S7 regression: effect sizes that match the test, and Dunn's test with ties.

Reference values were produced in R 4.5.3 with
effectsize 1.0.3 (rank_biserial, rank_epsilon_squared) and
dunn.test 1.4.1 (dunn.test(v, g, method = "none", altp = TRUE)).
The data contain repeated 0.3 values to mimic LOD replacement, so the
tie correction matters.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from modules import statistical_tests as st
from modules.statistical_tests import StatisticalAnalyzer

TT = st.TestType

MW_X = [0.3, 0.3, 0.3, 1.2, 2.5, 3.1, 4.0]
MW_Y = [0.3, 2.2, 3.5, 4.1, 5.8, 6.6]
R_RANK_BISERIAL = -0.547619047619          # effectsize::rank_biserial(x, y)

KW = {
    "A": [0.3, 0.3, 0.3, 0.3, 1.1, 2.0],
    "B": [0.3, 1.5, 2.2, 3.0, 3.3, 4.1],
    "C": [2.5, 3.8, 4.4, 5.0, 6.1, 7.2, 0.3],
}
R_RANK_EPSILON_SQ = 0.482191338074          # effectsize::rank_epsilon_squared(v ~ g)
R_DUNN = {                                  # dunn.test(..., method="none", altp=TRUE)
    ("A", "B"): (-1.64130783301, 0.100733530933),
    ("A", "C"): (-2.94305762525, 0.00324987919043),
    ("B", "C"): (-1.23979199174, 0.215052341035),
}


def _frame(groups):
    return pd.DataFrame([(g, v) for g, vs in groups.items() for v in vs], columns=["g", "y"])


def test_mann_whitney_reports_rank_biserial():
    d = _frame({"X": MW_X, "Y": MW_Y})
    es, kind, interp = StatisticalAnalyzer().calculate_effect_size(d, "y", "g", TT.MANN_WHITNEY)
    assert kind == "rank_biserial"
    assert es == pytest.approx(R_RANK_BISERIAL, rel=1e-10)
    assert interp == "large"


def test_rank_biserial_matches_pingouin_live():
    pg = pytest.importorskip("pingouin")
    d = _frame({"X": MW_X, "Y": MW_Y})
    es, _, _ = StatisticalAnalyzer().calculate_effect_size(d, "y", "g", TT.MANN_WHITNEY)
    assert es == pytest.approx(float(pg.mwu(MW_X, MW_Y)["RBC"].iloc[0]), rel=1e-10)


def test_kruskal_reports_rank_epsilon_squared():
    es, kind, interp = StatisticalAnalyzer().calculate_effect_size(_frame(KW), "y", "g", TT.KRUSKAL_WALLIS)
    assert kind == "epsilon_squared"
    assert es == pytest.approx(R_RANK_EPSILON_SQ, rel=1e-10)
    assert interp == "large"


def test_parametric_tests_keep_their_effect_sizes():
    a = StatisticalAnalyzer()
    d2 = _frame({"X": MW_X, "Y": MW_Y})
    assert a.calculate_effect_size(d2, "y", "g", TT.TTEST_INDEPENDENT)[1] == "cohens_d"
    assert a.calculate_effect_size(d2, "y", "g", TT.TTEST_WELCH)[1] == "cohens_d"
    assert a.calculate_effect_size(_frame(KW), "y", "g", TT.ANOVA_ONEWAY)[1] == "eta_squared"
    assert a.calculate_effect_size(_frame(KW), "y", "g", TT.ANOVA_WELCH)[1] == "eta_squared"


def test_dunn_matches_r_dunn_test_with_ties():
    res = StatisticalAnalyzer()._dunn_test(_frame(KW), "y", "g").pairwise_results
    assert len(res) == len(R_DUNN)
    for _, row in res.iterrows():
        z, p = R_DUNN[(row.group1, row.group2)]
        assert row.z_stat == pytest.approx(z, rel=1e-9)
        assert row.pvalue == pytest.approx(p, rel=1e-8)
    # BH within the table is unchanged policy
    expected_adj = stats.false_discovery_control(res.pvalue.values, method="bh")
    assert np.allclose(res.pvalue_adj.values, expected_adj)


def test_dunn_ignores_missing_values():
    d = _frame(KW)
    d_nan = pd.concat([d, pd.DataFrame({"g": ["A", "C", "C"], "y": [np.nan] * 3})], ignore_index=True)
    a = StatisticalAnalyzer()
    p_clean = a._dunn_test(d, "y", "g").pairwise_results.pvalue.values
    p_nan = a._dunn_test(d_nan, "y", "g").pairwise_results.pvalue.values
    assert np.allclose(p_clean, p_nan, rtol=1e-12)


def test_dunn_without_ties_reduces_to_classic_formula():
    d = _frame({"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0], "C": [9.0, 10.0, 11.0, 12.0]})
    res = StatisticalAnalyzer()._dunn_test(d, "y", "g").pairwise_results
    n = 12
    se = np.sqrt(n * (n + 1) / 12 * (1 / 4 + 1 / 4))
    row = res[(res.group1 == "A") & (res.group2 == "B")].iloc[0]
    assert row.z_stat == pytest.approx((2.5 - 6.5) / se, rel=1e-12)
