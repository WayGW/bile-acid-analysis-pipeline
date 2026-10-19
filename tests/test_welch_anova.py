"""S1 regression: Welch's one-way ANOVA must match statsmodels.

Reference: statsmodels.stats.oneway.anova_oneway(use_var="unequal"),
which implements Welch (1951). Before the S1 fix the pipeline's F was
inflated by a factor that grows with k (k > 2).
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from modules import statistical_tests as st
from modules.statistical_tests import StatisticalAnalyzer


def _groups(seed, sizes, sds, shifts):
    rng = np.random.default_rng(seed)
    return [rng.normal(loc=m, scale=s, size=n) for n, s, m in zip(sizes, sds, shifts)]


CASES = {
    "k3_equal_n": _groups(1, [8, 8, 8], [1.0, 3.0, 9.0], [0.0, 1.0, 4.0]),
    "k4_unequal_n": _groups(2, [6, 9, 12, 7], [0.5, 2.0, 4.0, 8.0], [0.0, 0.5, 2.0, 5.0]),
    "k5_small_n": _groups(3, [5, 5, 6, 6, 5], [1.0, 1.5, 3.0, 6.0, 10.0], [0.0, 0.0, 1.0, 3.0, 6.0]),
    "k3_null": _groups(4, [10, 10, 10], [1.0, 4.0, 10.0], [0.0, 0.0, 0.0]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_welch_anova_matches_statsmodels(name):
    groups = CASES[name]
    F, p = StatisticalAnalyzer()._welch_anova(groups)
    ref = anova_oneway(groups, use_var="unequal")
    assert F == pytest.approx(ref.statistic, rel=1e-10)
    assert p == pytest.approx(ref.pvalue, rel=1e-8, abs=1e-300)


def test_welch_anova_k2_equals_welch_t_squared():
    """Independent identity check: for k = 2, Welch F == Welch t^2 and p agrees."""
    g1, g2 = _groups(5, [7, 11], [1.0, 5.0], [0.0, 3.0])
    F, p = StatisticalAnalyzer()._welch_anova([g1, g2])
    t, p_t = stats.ttest_ind(g1, g2, equal_var=False)
    assert F == pytest.approx(t ** 2, rel=1e-10)
    assert p == pytest.approx(p_t, rel=1e-8)


def test_run_test_welch_path_reports_reference_values():
    """The ANOVA_WELCH branch of run_test (used by analyze) must carry the corrected F/p."""
    groups = CASES["k4_unequal_n"]
    df = pd.DataFrame({
        "y": np.concatenate(groups),
        "g": np.repeat([f"G{i}" for i in range(len(groups))], [len(x) for x in groups]),
    })
    res = StatisticalAnalyzer().run_test(df, "y", "g", st.TestType.ANOVA_WELCH)
    ref = anova_oneway(groups, use_var="unequal")
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-8)


def test_welch_anova_tiny_pvalue_not_truncated_to_zero():
    """p comes from the F survival function, so very small p stay > 0."""
    groups = _groups(6, [12, 12, 12], [0.2, 0.3, 0.4], [0.0, 3.0, 6.0])
    F, p = StatisticalAnalyzer()._welch_anova(groups)
    ref = anova_oneway(groups, use_var="unequal")
    assert p > 0
    assert p == pytest.approx(ref.pvalue, rel=1e-6)
