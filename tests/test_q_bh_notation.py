"""Regression tests for explicit BH q-value notation in reports and APA text."""
from types import SimpleNamespace

from modules.report_generation import (
    get_threeway_differences_summary,
    get_twoway_differences_summary,
)
from modules.statistical_tests import (
    TestType as StatisticalTestType,
    format_threeway_apa,
    format_twoway_apa,
)


def _twoway_result():
    return SimpleNamespace(
        test_type=StatisticalTestType.ANOVA_TWOWAY,
        factor_a_name="Genotype",
        factor_b_name="Diet",
        factor_a_stat=1.20,
        factor_a_pvalue=0.040,
        factor_a_qvalue=0.200,
        factor_a_df=(1, 20),
        factor_b_stat=1.10,
        factor_b_pvalue=0.300,
        factor_b_qvalue=0.400,
        factor_b_df=(1, 20),
        interaction_stat=0.50,
        interaction_pvalue=0.600,
        interaction_qvalue=0.700,
        interaction_df=(1, 20),
        interaction_significant=False,
        effect_sizes={},
        posthoc_results=None,
        posthoc_type="",
        alpha=0.05,
    )


def _threeway_result():
    return SimpleNamespace(
        test_type=StatisticalTestType.ANOVA_THREEWAY,
        factor_a_name="Genotype",
        factor_b_name="Diet",
        factor_c_name="Treatment",
        factor_a_stat=1.20,
        factor_a_pvalue=0.040,
        factor_a_qvalue=0.200,
        factor_a_df=(1, 20),
        factor_b_stat=1.10,
        factor_b_pvalue=0.300,
        factor_b_qvalue=0.400,
        factor_b_df=(1, 20),
        factor_c_stat=1.00,
        factor_c_pvalue=0.500,
        factor_c_qvalue=0.600,
        factor_c_df=(1, 20),
        interaction_ab_stat=0.90,
        interaction_ab_pvalue=0.030,
        interaction_ab_qvalue=0.300,
        interaction_ab_df=(1, 20),
        interaction_ac_stat=0.80,
        interaction_ac_pvalue=0.400,
        interaction_ac_qvalue=0.500,
        interaction_ac_df=(1, 20),
        interaction_bc_stat=0.70,
        interaction_bc_pvalue=0.500,
        interaction_bc_qvalue=0.600,
        interaction_bc_df=(1, 20),
        interaction_abc_stat=0.60,
        interaction_abc_pvalue=0.600,
        interaction_abc_qvalue=0.700,
        interaction_abc_df=(1, 20),
        effect_sizes={},
        posthoc_results=None,
        posthoc_type="",
        alpha=0.05,
    )


def test_twoway_apa_uses_q_bh_notation():
    result = SimpleNamespace(
        variable_name="CA",
        twoway_result=_twoway_result(),
    )
    text = format_twoway_apa(result)

    assert "q(BH) = 0.200" in text
    assert "q(BH) = 0.700" in text
    assert ", q = " not in text


def test_threeway_apa_uses_q_bh_notation():
    result = SimpleNamespace(
        variable_name="CA",
        threeway_result=_threeway_result(),
    )
    text = format_threeway_apa(result)

    assert "q(BH) = 0.200" in text
    assert "q(BH) = 0.300" in text
    assert ", q = " not in text


def test_factorial_summary_headers_identify_bh():
    twoway = get_twoway_differences_summary({
        "CA": SimpleNamespace(twoway_result=_twoway_result()),
    })
    assert "Genotype q (BH)" in twoway.columns
    assert "Diet q (BH)" in twoway.columns
    assert "Interaction q (BH)" in twoway.columns

    threeway = get_threeway_differences_summary({
        "CA": SimpleNamespace(threeway_result=_threeway_result()),
    })
    assert "Genotype q (BH)" in threeway.columns
    assert "Genotype×Diet q (BH)" in threeway.columns
    assert "Genotype×Diet×Treatment q (BH)" in threeway.columns
