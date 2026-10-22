"""Regression tests for factorial ANOVA annotation boxes.

The boxes must show raw p-values alongside BH q-values when FDR is enabled,
and their background must follow the q-value decision rather than the raw
p-value.  Synthetic data only.
"""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
import pytest

from modules.visualization import (
    BileAcidVisualizer,
    Q_BH_NOTE,
    Q_BH_NOTE_SHORT,
)


GOLD = to_rgb("#fff3cd")
GREY = to_rgb("#e9ecef")


@pytest.fixture
def twoway_data():
    return pd.DataFrame({
        "A": np.repeat(["WT", "KO"], 4),
        "B": np.tile(np.repeat(["Chow", "WD"], 2), 2),
        "CA": [1.0, 1.2, 2.0, 2.2, 3.0, 3.2, 4.0, 4.2],
    })


@pytest.fixture
def threeway_data():
    rows = [
        (a, b, c)
        for a in ("WT", "KO")
        for b in ("Chow", "WD")
        for c in ("Veh", "Drug")
        for _ in range(2)
    ]
    df = pd.DataFrame(rows, columns=["A", "B", "C"])
    df["CA"] = np.arange(1.0, len(df) + 1.0)
    return df


def _twoway_result(**overrides):
    values = dict(
        factor_a_name="Genotype",
        factor_b_name="Diet",
        factor_a_pvalue=0.5,
        factor_a_qvalue=0.5,
        factor_b_pvalue=0.5,
        factor_b_qvalue=0.5,
        interaction_pvalue=0.5,
        interaction_qvalue=0.5,
        posthoc_results=None,
        alpha=0.05,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _threeway_result(**overrides):
    values = dict(
        factor_a_name="Genotype",
        factor_b_name="Diet",
        factor_c_name="Treatment",
        alpha=0.05,
    )
    for prefix in (
        "factor_a",
        "factor_b",
        "factor_c",
        "interaction_ab",
        "interaction_ac",
        "interaction_bc",
        "interaction_abc",
    ):
        values[f"{prefix}_pvalue"] = 0.5
        values[f"{prefix}_qvalue"] = 0.5
    values.update(overrides)
    return SimpleNamespace(**values)


def _match_text(texts, needle):
    matches = [text for text in texts if needle in text.get_text()]
    assert len(matches) == 1
    return matches[0]


def _facecolor(text):
    return text.get_bbox_patch().get_facecolor()[:3]


def test_twoway_annotation_uses_q_for_background_and_shows_both_values(twoway_data):
    # Raw p-values are significant, but the corrected q-values are not.
    result = _twoway_result(
        factor_a_pvalue=0.001,
        factor_a_qvalue=0.200,
        interaction_pvalue=0.010,
        interaction_qvalue=0.300,
    )
    ax = BileAcidVisualizer().plot_twoway_bar(
        twoway_data, "CA", "A", "B", result, show_points=False
    )
    try:
        text = _match_text(ax.texts, "Genotype:")
        assert "p=0.001, q(BH)=0.200 n.s." in text.get_text()
        assert "p=0.010, q(BH)=0.300 n.s." in text.get_text()
        assert np.allclose(_facecolor(text), GREY)
    finally:
        plt.close(ax.figure)


def test_twoway_annotation_is_gold_only_for_q_significant_effect(twoway_data):
    result = _twoway_result(
        factor_a_pvalue=0.020,
        factor_a_qvalue=0.010,
    )
    ax = BileAcidVisualizer().plot_twoway_bar(
        twoway_data, "CA", "A", "B", result, show_points=False
    )
    try:
        text = _match_text(ax.texts, "Genotype:")
        assert "p=0.020, q(BH)=0.010 *" in text.get_text()
        assert np.allclose(_facecolor(text), GOLD)
    finally:
        plt.close(ax.figure)


def test_threeway_faceted_summary_shows_both_values_and_uses_q_background(
    threeway_data,
):
    result = _threeway_result(
        factor_a_pvalue=0.001,
        factor_a_qvalue=0.200,
    )
    fig = BileAcidVisualizer().plot_threeway_faceted(
        threeway_data, "CA", "A", "B", "C", result
    )
    try:
        text = _match_text(fig.texts, "Main effects:")
        assert "Genotype: p=0.001, q(BH)=0.200 ns" in text.get_text()
        assert Q_BH_NOTE_SHORT in text.get_text()
        assert np.allclose(_facecolor(text), GREY)
    finally:
        plt.close(fig)

    result = _threeway_result(
        factor_a_pvalue=0.020,
        factor_a_qvalue=0.010,
    )
    fig = BileAcidVisualizer().plot_threeway_faceted(
        threeway_data, "CA", "A", "B", "C", result
    )
    try:
        text = _match_text(fig.texts, "Main effects:")
        assert "Genotype: p=0.020, q(BH)=0.010 *" in text.get_text()
        assert Q_BH_NOTE_SHORT in text.get_text()
        assert np.allclose(_facecolor(text), GOLD)
    finally:
        plt.close(fig)


def test_threeway_multi_panel_annotation_shows_both_values_and_uses_q_background(
    threeway_data,
):
    result = _threeway_result(
        factor_a_pvalue=0.001,
        factor_a_qvalue=0.200,
        interaction_ab_pvalue=0.010,
        interaction_ab_qvalue=0.300,
    )
    fig = BileAcidVisualizer().plot_threeway_multi_panel(
        threeway_data, ["CA"], "A", "B", "C", {"CA": result},
        factor_a_name="Genotype", factor_b_name="Diet",
        factor_c_name="Treatment",
    )
    try:
        text = _match_text(
            [text for ax in fig.axes for text in ax.texts], "Genotype:"
        )
        assert "p=0.001, q(BH)=0.200 ns" in text.get_text()
        assert "p=0.010, q(BH)=0.300 ns" in text.get_text()
        assert np.allclose(_facecolor(text), GREY)
        assert any(Q_BH_NOTE in text.get_text() for text in fig.texts)
    finally:
        plt.close(fig)

    result = _threeway_result(
        factor_a_pvalue=0.020,
        factor_a_qvalue=0.010,
    )
    fig = BileAcidVisualizer().plot_threeway_multi_panel(
        threeway_data, ["CA"], "A", "B", "C", {"CA": result},
        factor_a_name="Genotype", factor_b_name="Diet",
        factor_c_name="Treatment",
    )
    try:
        text = _match_text(
            [text for ax in fig.axes for text in ax.texts], "Genotype:"
        )
        assert "p=0.020, q(BH)=0.010 *" in text.get_text()
        assert np.allclose(_facecolor(text), GOLD)
        assert any(Q_BH_NOTE in text.get_text() for text in fig.texts)
    finally:
        plt.close(fig)


def test_twoway_multi_panel_uses_q_bh_labels_and_adds_footnote(twoway_data):
    result = _twoway_result(
        factor_a_pvalue=0.020,
        factor_a_qvalue=0.010,
    )
    fig = BileAcidVisualizer().plot_twoway_multi_panel(
        twoway_data, ["CA"], "A", "B", {"CA": result},
        factor_a_name="Genotype", factor_b_name="Diet",
    )
    try:
        text = _match_text(
            [text for ax in fig.axes for text in ax.texts], "Genotype:"
        )
        assert "p=0.020, q(BH)=0.010 *" in text.get_text()
        assert any(Q_BH_NOTE in text.get_text() for text in fig.texts)
    finally:
        plt.close(fig)


def test_oneway_multi_panel_uses_q_bh_label_and_adds_footnote(twoway_data):
    result = SimpleNamespace(
        main_test=SimpleNamespace(
            pvalue=0.020,
            qvalue=0.010,
            significant=True,
            groups_compared=[],
        ),
        posthoc_test=None,
    )
    fig = BileAcidVisualizer().plot_multi_panel_groups_with_stats(
        twoway_data, ["CA"], "A", {"CA": result},
        ncols=1, show_points=False,
    )
    try:
        assert "q(BH)=0.010" in fig.axes[0].get_title()
        assert any(Q_BH_NOTE in text.get_text() for text in fig.texts)
    finally:
        plt.close(fig)
