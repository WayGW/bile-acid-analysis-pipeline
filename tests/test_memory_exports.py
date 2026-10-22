import inspect
import zipfile
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import app
from modules.data_processing import ProcessedData


def _fake_processed():
    sample = pd.DataFrame({
        "Sample_ID": ["S1", "S2", "S3"],
        "Group": ["A", "A", "B"],
        "CA": [1.0, 2.0, 3.0],
    })
    structure = SimpleNamespace(
        sample_id_col="Sample_ID",
        group_col="Group",
        factors={},
        bile_acid_cols=["CA"],
        analyte_lods={"CA": 1.0},
        analyte_lod_counts={"CA": 0},
        analyte_lod_rows={"CA": []},
        lod_source="default",
    )
    return SimpleNamespace(
        sample_data=sample,
        concentrations=sample[["CA"]],
        percentages=pd.DataFrame({"CA_pct": [50.0, 60.0, 70.0]}),
        totals=pd.DataFrame({"total_all": [1.0, 2.0, 3.0]}),
        ratios=pd.DataFrame({"ratio": [0.5, 0.6, 0.7]}),
        structure=structure,
    )


def test_processed_data_does_not_retain_raw_or_group_summaries():
    names = {f.name for f in fields(ProcessedData)}
    assert "raw_data" not in names
    assert "group_summaries" not in names


def test_export_figures_are_lazy_and_zip_is_disk_backed():
    assert inspect.isgeneratorfunction(app.iter_all_export_figures)

    processed = _fake_processed()
    zip_path, count, errors = app.create_results_zip(
        processed, None, iter(()), None, {"lod_handling": "lod"}
    )
    try:
        path = Path(zip_path)
        assert path.exists()
        assert count == 0
        assert errors == []
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        assert "data/concentrations.xlsx" in names
        assert "data/full_data_lod_highlighted.xlsx" in names
        assert not any(name.startswith("figures/") for name in names)
    finally:
        Path(zip_path).unlink(missing_ok=True)


def _fake_settings(**overrides):
    settings = {
        "lod_handling": "lod",
        "lod_threshold": 50,
        "alpha": 0.05,
        "plot_type": "violin",
        "show_points": False,
        "color_palette": "Dark2",
        "plot_style": "whitegrid",
        "dilution_factor": 1.0,
        "units": "uM",
        "tissue": "blood",
        "level_orders": {"Group": ["B", "A"]},
    }
    settings.update(overrides)
    return settings


def _fake_results():
    return SimpleNamespace(
        is_twoway=False,
        is_threeway=False,
        individual_ba_results={},
        log_individual_ba_results={},
        totals_results={},
        log_totals_results={},
        percentages_results={},
        ratios_results={},
        log_ratios_results={},
        hydrophobicity_results={},
        category_results={},
    )


def _fake_processed_four_samples():
    sample = pd.DataFrame({
        "Sample_ID": ["S1", "S2", "S3", "S4"],
        "Group": ["A", "A", "B", "B"],
        "CA": [1.0, 2.0, 3.0, 4.0],
    })
    structure = SimpleNamespace(
        sample_id_col="Sample_ID",
        group_col="Group",
        factors={},
        n_factors=0,
        bile_acid_cols=["CA"],
        analyte_lods={"CA": 1.0},
        analyte_lod_counts={"CA": 0},
        analyte_lod_rows={"CA": []},
        lod_source="default",
    )
    return SimpleNamespace(
        sample_data=sample,
        concentrations=sample[["CA"]],
        percentages=pd.DataFrame({"CA_pct": [40.0, 50.0, 60.0, 70.0]}),
        totals=pd.DataFrame({"total_all": [1.0, 2.0, 3.0, 4.0]}),
        ratios=pd.DataFrame({"ratio": [0.5, 0.6, 0.7, 0.8]}),
        structure=structure,
    )


def _oneway_spec_params():
    return {
        "design": "one",
        "data_source": "sample",
        "value_cols": ["CA"],
        "source_cols": ["CA"],
        "factor_cols": [],
        "group_col": "Group",
        "stats_source_raw": "individual_ba_results",
        "stats_source_log": "log_individual_ba_results",
        "stats_columns": ["CA"],
        "ylabel": "Concentration (uM)",
        "ncols": 1,
        "log_scale": False,
    }


def test_figure_specs_snapshot_style_and_clear_cached_zip(tmp_path):
    app.st.session_state.clear()
    app.st.session_state.figure_specs = {}
    settings = _fake_settings()
    params = _oneway_spec_params()
    app.register_figure_spec("concentrations", "oneway_multi", settings, params)

    spec = app.st.session_state.figure_specs["concentrations"]
    settings["plot_type"] = "bar"
    params["value_cols"].append("DCA")
    assert spec["settings"]["plot_type"] == "violin"
    assert spec["params"]["value_cols"] == ["CA"]

    old_zip = tmp_path / "old.zip"
    old_zip.write_bytes(b"old")
    app.st.session_state.zip_path = str(old_zip)
    app.st.session_state.zip_figure_count = 4

    changed = _oneway_spec_params()
    changed["value_cols"] = ["CA", "DCA"]
    app.register_figure_spec("concentrations", "oneway_multi", settings, changed)
    assert not old_zip.exists()
    assert app.st.session_state.zip_path is None

    # Re-registering the exact same recipe must not discard a fresh package.
    fresh_zip = tmp_path / "fresh.zip"
    fresh_zip.write_bytes(b"fresh")
    app.st.session_state.zip_path = str(fresh_zip)
    app.register_figure_spec(
        "concentrations", "oneway_multi",
        app.st.session_state.figure_specs["concentrations"]["settings"],
        app.st.session_state.figure_specs["concentrations"]["params"],
    )
    assert fresh_zip.exists()


def test_scale_pair_rebuilds_viewed_style_order_and_log_form():
    processed = _fake_processed_four_samples()
    results = _fake_results()
    app.st.session_state.clear()
    app.st.session_state.figure_specs = {}
    app.register_scale_pair(
        "concentrations", "oneway_multi", _fake_settings(), _oneway_spec_params()
    )

    specs = app.st.session_state.figure_specs
    assert set(specs) == {"concentrations", "concentrations_log"}
    linear = app.build_figure_from_spec(specs["concentrations"], processed, results)
    try:
        ax = linear.axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["B", "A"]
        assert ax.get_ylabel() == "Concentration (uM)"
        assert ax.get_yscale() == "linear"
    finally:
        plt.close(linear)

    logged = app.build_figure_from_spec(specs["concentrations_log"], processed, results)
    try:
        # The plotting layer expresses log variants as log10-transformed values.
        assert logged.axes[0].get_ylabel() == "log10(Concentration (uM))"
    finally:
        plt.close(logged)


def test_export_prefers_viewed_recipe_before_default_variants():
    processed = _fake_processed_four_samples()
    results = _fake_results()
    app.st.session_state.clear()
    app.st.session_state.figure_specs = {}
    app.register_figure_spec(
        "concentrations", "oneway_multi", _fake_settings(), _oneway_spec_params()
    )

    figures = app.iter_export_figures(processed, results, _fake_settings())
    try:
        name, fig = next(figures)
        assert name == "concentrations"
        plt.close(fig)
    finally:
        figures.close()


def test_zip_includes_viewed_recipe_and_always_on_figure_types():
    processed = _fake_processed_four_samples()
    results = _fake_results()
    app.st.session_state.clear()
    app.st.session_state.figure_specs = {}
    app.st.session_state.hydrophobicity = pd.DataFrame({
        "HI_pool": [0.1, 0.2, 0.3, 0.4],
        "hydrophobic_load": [1.0, 1.1, 1.2, 1.3],
    })
    app.register_figure_spec(
        "concentrations", "oneway_multi", _fake_settings(), _oneway_spec_params()
    )

    zip_path, count, errors = app.create_results_zip(
        processed, results,
        app.iter_export_figures(processed, results, _fake_settings()),
        None, _fake_settings(),
    )
    try:
        assert errors == []
        assert count >= 3
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
        assert "figures/concentrations.pdf" in names
        assert "figures/percentages_pie.pdf" in names
        assert "figures/hydrophobicity.pdf" in names
    finally:
        Path(zip_path).unlink(missing_ok=True)
