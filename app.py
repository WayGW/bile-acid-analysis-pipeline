"""
Bile Acid Analysis Pipeline - Streamlit Application
====================================================

Run with: streamlit run app.py
"""

import copy
import gc
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import zipfile
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config.bile_acid_species import (
    BILE_ACID_PANEL, ANALYSIS_GROUPS, ANALYSIS_GROUP_DISPLAY_NAMES,
    get_glycine_conjugated, get_taurine_conjugated,
    get_primary, get_secondary, get_conjugated, get_unconjugated,
    get_keto_derivatives, get_iso_forms, get_nor_bile_acids,
    get_12alpha_hydroxylated, get_non12alpha_hydroxylated, get_sulfated
)
from modules.data_processing import BileAcidDataProcessor, ProcessedData, validate_data_quality
from modules.statistical_tests import (
    StatisticalAnalyzer, format_analysis_report, format_twoway_apa,
    format_threeway_apa, effect_pvalue,
)
from modules.visualization import BileAcidVisualizer
from modules.group_order import apply_order, default_levels, order_table, positions_to_order, combined_order
from modules.hydrophobicity import HydrophobicityCalculator
from modules.report_generation import (
    ExcelReportGenerator, SignificancePlotter,
    ComprehensiveAnalysisResults, format_apa_statistics,
    get_significant_differences_summary, get_twoway_differences_summary,
    get_threeway_differences_summary
)

st.set_page_config(page_title="Bile Acid Analysis Pipeline", page_icon="🧬", layout="wide")


def df_to_excel_bytes(df, number_format='0.00'):
    """Convert a DataFrame to Excel bytes with numeric cell formatting."""
    from openpyxl.utils import get_column_letter
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        ws = writer.sheets['Sheet1']
        for col_idx, col_name in enumerate(df.columns, 1):
            if pd.api.types.is_numeric_dtype(df[col_name]):
                col_letter = get_column_letter(col_idx)
                for row_idx in range(2, len(df) + 2):
                    ws[f'{col_letter}{row_idx}'].number_format = number_format
    buf.seek(0)
    return buf.getvalue()


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'processed_data': None,
        'analysis_results': ComprehensiveAnalysisResults(),
        'report_generator': None,
        'stats_computed': False,
        'stats_sections_done': set(),  # Track which stat sections are computed
        'last_settings': None,  # Track settings that affect data/stats
        'zip_path': None,  # On-disk ZIP package (not held in memory)
        'zip_figure_count': 0,  # Number of figures in cached ZIP
        'selected_sheet': None,  # User-selected data sheet
        'hydrophobicity': None,  # HI_pool and hydrophobic_load DataFrame
        'hydro_tissue': 'blood',  # Current tissue for HI calculation
        'level_orders': {},  # User display order {column: [levels]} (figures only)
        'figure_specs': {},  # Lightweight recipes for figures viewed in tabs
        'uploaded_file_id': None,  # Detect same-name file replacements
        'uploaded_tmp_path': None,  # Stable temp file for the current upload
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _remove_file(path):
    """Delete a temporary file, ignoring missing files and filesystem errors."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def clear_zip_package():
    """Remove the on-disk results ZIP and forget its path."""
    _remove_file(st.session_state.get('zip_path'))
    st.session_state.zip_path = None
    st.session_state.zip_figure_count = 0


def clear_export_artifacts():
    """Drop the on-disk ZIP package after a data/display change."""
    clear_zip_package()


def clear_figure_specs(predicate=None):
    """Forget viewed-figure recipes, all of them or only matching names."""
    specs = st.session_state.get('figure_specs') or {}
    if predicate is None:
        st.session_state.figure_specs = {}
    else:
        st.session_state.figure_specs = {
            name: spec for name, spec in specs.items() if not predicate(name)
        }
    clear_zip_package()


def reset_analysis_state(clear_upload=False):
    """Release per-analysis objects while keeping sidebar settings intact."""
    clear_export_artifacts()
    st.session_state.processed_data = None
    st.session_state.analysis_results = ComprehensiveAnalysisResults()
    st.session_state.report_generator = None
    st.session_state.stats_computed = False
    st.session_state.stats_sections_done = set()
    st.session_state.hydrophobicity = None
    st.session_state.selected_sheet = None
    st.session_state.level_orders = {}
    st.session_state.figure_specs = {}
    st.session_state.last_settings = None
    st.session_state.pop('heatmap_fig', None)
    if clear_upload:
        _remove_file(st.session_state.get('uploaded_tmp_path'))
        st.session_state.uploaded_tmp_path = None
        st.session_state.uploaded_file_id = None


def get_data_affecting_settings(settings):
    """Extract settings that affect data processing and statistics."""
    return {
        'lod_handling': settings['lod_handling'],
        'lod_threshold': settings.get('lod_threshold', 50),
        'aggregate_lod_threshold': settings.get('aggregate_lod_threshold', 5),
        'fdr_enabled': settings.get('fdr_enabled', True),
        'alpha': settings['alpha'],
        'dilution_factor': settings.get('dilution_factor', 1.0),
        'units': settings.get('units', 'nmol/L'),
    }


def check_settings_changed(settings):
    """Check if data-affecting settings have changed, reset caches if so."""
    current = get_data_affecting_settings(settings)
    
    if st.session_state.last_settings != current:
        # Settings changed - need to reprocess data and recompute stats
        st.session_state.stats_computed = False
        st.session_state.stats_sections_done = set()
        st.session_state.analysis_results = ComprehensiveAnalysisResults()
        st.session_state.report_generator = None
        st.session_state.hydrophobicity = None
        st.session_state.figure_specs = {}
        st.session_state.pop('heatmap_fig', None)
        clear_export_artifacts()
        st.session_state.last_settings = current
        return True
    return False


# ---------------------------------------------------------------------------
# Group order for figures (display only; never affects statistics)
# ---------------------------------------------------------------------------

def make_visualizer(settings):
    """BileAcidVisualizer with the sidebar style and the user's group order."""
    return BileAcidVisualizer(color_palette=settings['color_palette'],
                              style=settings['plot_style'],
                              level_orders=settings.get('level_orders'))


def get_order_columns(processed):
    """Columns whose display order can be set: {column: (label, default mode)}.

    Factors (two-/three-way designs) default to alphabetical, as in the factorial
    plots; a plain group column defaults to order of first appearance.
    """
    s = processed.structure
    data = processed.sample_data
    cols = {}
    n_factors = getattr(s, 'n_factors', 0) or 0
    if n_factors >= 2:
        for name, col in list((getattr(s, 'factors', {}) or {}).items())[:n_factors]:
            if col in data.columns:
                cols[col] = (name, 'sorted')
    gc = s.group_col
    if gc and gc in data.columns and gc not in cols and gc != '_group_':
        cols[gc] = (gc, 'appearance')
    return cols


def build_level_orders(processed, stored):
    """User orders plus derived orders for combined factor labels ('A - B[ - C]')."""
    orders = {k: list(v) for k, v in (stored or {}).items() if v}
    s = processed.structure
    data = processed.sample_data
    factor_cols = [c for c in (getattr(s, 'factors', {}) or {}).values() if c in data.columns]
    if orders and len(factor_cols) >= 2 and any(c in orders for c in factor_cols):
        def levels(c):
            return apply_order(default_levels(data[c], 'sorted'), orders.get(c))
        n = min(getattr(s, 'n_factors', 0) or 0, 3)
        if n >= 2:   # labels built in the tabs from factors A, B[, C]
            orders['_factorial_group_'] = combined_order([levels(c) for c in factor_cols[:n]])
        if s.group_col == '_group_':   # synthetic group built from all factors
            orders['_group_'] = combined_order([levels(c) for c in factor_cols])
    return orders


def report_display_orders(settings):
    """User-set orders to record in the report Overview (derived labels omitted)."""
    orders = {k: v for k, v in (settings.get('level_orders') or {}).items()
              if v and not str(k).startswith('_')}
    return orders or None


def order_pivot_columns(pivot_df, settings, col):
    """Reorder a summary pivot's group columns by the display order (unchanged if none)."""
    order = (settings.get('level_orders') or {}).get(col)
    if not order:
        return pivot_df
    return pivot_df[apply_order(list(pivot_df.columns), order)]


def render_group_order_controls(processed):
    """Expander with one Group/Position table per group column or factor."""
    cols = get_order_columns(processed)
    if not cols:
        return
    stored = st.session_state.level_orders
    with st.expander("\U0001F500 Group order for figures", expanded=False):
        st.caption("Position 1 is furthest left. Applies to every figure, the ZIP package and the "
                   "report figures. Statistics are not recomputed or changed.")
        with st.form("group_order_form", border=False):
            tables = {}
            for (col, (label, mode)), ui in zip(cols.items(), st.columns(len(cols))):
                levels = apply_order(default_levels(processed.sample_data[col], mode), stored.get(col))
                with ui:
                    st.markdown(f"**{label}**")
                    tables[col] = st.data_editor(
                        order_table(levels), key=f"group_order_editor_{col}",
                        hide_index=True, disabled=["Group"], width="stretch",
                        column_config={"Position": st.column_config.NumberColumn(
                            "Position", min_value=1, step=1, required=True)})
            b1, b2 = st.columns(2)
            apply_clicked = b1.form_submit_button("Apply order", type="primary")
            reset_clicked = b2.form_submit_button("Reset to default")
    if not (apply_clicked or reset_clicked):
        return
    if reset_clicked:
        new_orders = {}
    else:
        new_orders, errors = {}, []
        for col, table in tables.items():
            order, err = positions_to_order(table)
            if err:
                errors.append(f"{cols[col][0]}: {err}")
            else:
                new_orders[col] = order
        if errors:
            st.warning("Group order not applied. " + " ".join(errors))
            return
    for col in cols:
        st.session_state.pop(f"group_order_editor_{col}", None)
    if new_orders != stored:
        st.session_state.level_orders = new_orders
        # Display-only change: drop saved figure recipes, keep all statistics.
        st.session_state.pop('heatmap_fig', None)
        clear_figure_specs()
    st.rerun()


def fig_to_bytes(fig, format='png', dpi=300):
    """Convert matplotlib figure to bytes."""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def _ensure_report_generator(processed, settings):
    """Create ExcelReportGenerator if not already cached."""
    if st.session_state.report_generator is not None:
        return st.session_state.report_generator

    group_col = processed.structure.group_col
    factors = getattr(processed.structure, 'factors', {})
    n_factors = getattr(processed.structure, 'n_factors', 0)

    report_gen = ExcelReportGenerator(
        data=processed.sample_data,
        group_col=group_col,
        sample_id_col=getattr(processed.structure, 'sample_id_col', None),  # S16: keep Sample_ID in report sheets
        bile_acid_cols=processed.structure.bile_acid_cols,
        totals=processed.totals,
        ratios=processed.ratios,
        percentages=processed.percentages,
        hydrophobicity=st.session_state.hydrophobicity,
        hydro_ph=HydrophobicityCalculator.PH_ENVIRONMENTS[settings.get('tissue', 'blood')],
        hydro_tissue=settings.get('tissue', 'blood'),
        alpha=settings['alpha'],
        factors=factors,
        n_factors=n_factors,
        analyte_lod_counts=getattr(processed.structure, 'analyte_lod_counts', {}),
        analyte_lod_rows=getattr(processed.structure, 'analyte_lod_rows', {}),
        analyte_lod_values=getattr(processed.structure, 'analyte_lod_values', {}),
        n_samples=len(processed.sample_data),
        lod_threshold=settings.get('lod_threshold', 50),
        aggregate_lod_threshold=settings.get('aggregate_lod_threshold', 5),
        dilution_factor=settings.get('dilution_factor', 1.0),
        fdr_enabled=settings.get('fdr_enabled', True),
        units=settings.get('units', 'nmol/L'),
    )
    st.session_state.report_generator = report_gen
    return report_gen


def ensure_section_computed(processed, settings, section):
    """Compute stats for a single section if not already done."""
    if section in st.session_state.stats_sections_done:
        return st.session_state.analysis_results

    group_col = processed.structure.group_col
    if not group_col:
        return st.session_state.analysis_results

    report_gen = _ensure_report_generator(processed, settings)
    results = report_gen.run_section(section)
    st.session_state.analysis_results = results
    st.session_state.stats_sections_done.add(section)
    return results


def compute_all_statistics(processed, settings):
    """Compute all statistics at once (used for one-way and two-way designs)."""
    if st.session_state.stats_computed:
        return st.session_state.analysis_results

    group_col = processed.structure.group_col
    if not group_col:
        return None

    report_gen = _ensure_report_generator(processed, settings)
    results = report_gen.run_all_statistics()
    st.session_state.analysis_results = results
    st.session_state.stats_computed = True
    st.session_state.stats_sections_done = {'individual_ba', 'totals', 'ratios', 'percentages', 'categories', 'hydrophobicity'}
    return results


def get_sig_pairs(result, max_pairs=5):
    """Extract significant pairs from result (post-hoc, or main test when k = 2; S5)."""
    from modules.statistical_tests import significant_pairs
    return significant_pairs(result, max_pairs=max_pairs, sort=True)


# ---------------------------------------------------------------------------
# Viewed-figure recipes
# ---------------------------------------------------------------------------
# The old implementation stored every viewed figure as PDF bytes.  That made
# the export exact but kept tens of megabytes alive for the whole session.  A
# recipe stores only the small choices that produced the figure (columns,
# log scale, style snapshot, factor names, etc.); the figure is rebuilt one at
# a time when the ZIP is requested.

def register_figure_spec(name, kind, settings, params):
    """Remember how a figure was viewed, without retaining its pixels."""
    spec = {
        'kind': kind,
        'settings': copy.deepcopy(settings),
        'params': copy.deepcopy(params),
    }
    specs = st.session_state.setdefault('figure_specs', {})
    if specs.get(name) == spec:
        return
    specs[name] = spec
    # An already-generated ZIP no longer matches what the user has viewed.
    clear_zip_package()


def register_scale_pair(base_name, kind, settings, params):
    """Register both linear and log forms of a multi-panel figure."""
    viewed_log = bool(params.get('log_scale', False))
    for log_scale in (viewed_log, not viewed_log):
        p = dict(params)
        p['log_scale'] = log_scale
        register_figure_spec(
            f"{base_name}{'_log' if log_scale else ''}", kind, settings, p
        )


def _result_source_name(results, section, log=False):
    """Attribute name holding the result dict for a section and design."""
    is_threeway = bool(getattr(results, 'is_threeway', False))
    is_twoway = bool(getattr(results, 'is_twoway', False))
    if section == 'individual_ba':
        base = ('threeway_individual_ba' if is_threeway
                else 'twoway_individual_ba' if is_twoway
                else 'individual_ba_results')
    elif section == 'totals':
        base = ('threeway_totals' if is_threeway
                else 'twoway_totals' if is_twoway
                else 'totals_results')
    elif section == 'ratios':
        base = ('threeway_ratios' if is_threeway
                else 'twoway_ratios' if is_twoway
                else 'ratios_results')
    elif section == 'percentages':
        base = ('threeway_percentages' if is_threeway
                else 'twoway_percentages' if is_twoway
                else 'percentages_results')
    elif section == 'hydrophobicity':
        base = ('threeway_hydrophobicity' if is_threeway
                else 'twoway_hydrophobicity' if is_twoway
                else 'hydrophobicity_results')
    elif section == 'categories':
        base = 'category_results'
    else:
        return ''
    if log and section in ('individual_ba', 'totals', 'ratios'):
        return f'log_{base}'
    return base


def _filtered_sample_data(processed):
    """Sample rows used by every plot tab (drop missing/NaN groups)."""
    group_col = processed.structure.group_col
    if not group_col or group_col not in processed.sample_data.columns:
        return processed.sample_data.iloc[0:0].copy()
    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    return data[data[group_col].astype(str).str.lower() != 'nan']


def _spec_meta_columns(params, data, processed):
    factor_cols = [c for c in (params.get('factor_cols') or []) if c in data.columns]
    if factor_cols:
        return factor_cols
    group_col = params.get('group_col') or processed.structure.group_col
    return [group_col] if group_col in data.columns else []


def _prepare_spec_data(processed, params):
    """Rebuild the DataFrame used by a saved multi-panel/interaction recipe."""
    data = _filtered_sample_data(processed)
    source = params.get('data_source', 'sample')
    value_cols = list(params.get('value_cols') or [])
    source_cols = list(params.get('source_cols') or value_cols)

    if source == 'sample':
        return data, params.get('group_col') or processed.structure.group_col

    if source == 'totals':
        source_df = processed.totals
    elif source == 'ratios':
        source_df = processed.ratios
    elif source == 'percentages':
        source_df = processed.percentages
    elif source == 'hydro':
        source_df = st.session_state.get('hydrophobicity')
    else:
        raise ValueError(f"Unknown figure data source: {source}")

    if source_df is None:
        raise ValueError(f"No data available for {source}")
    missing = [c for c in source_cols if c not in source_df.columns]
    if missing:
        raise ValueError(f"Missing columns for saved figure: {missing}")

    values = source_df.loc[data.index, source_cols].reset_index(drop=True)
    values.columns = value_cols
    meta_cols = _spec_meta_columns(params, data, processed)
    if meta_cols:
        meta = data[meta_cols].reset_index(drop=True)
        return pd.concat([meta, values], axis=1), params.get('group_col') or processed.structure.group_col
    return values, params.get('group_col') or processed.structure.group_col


def _spec_stats(results, params):
    """Get the result objects for the exact log/raw form of a saved figure."""
    key = (params.get('stats_source_log') if params.get('log_scale')
           else params.get('stats_source_raw'))
    if not key:
        return {}
    all_stats = getattr(results, key, None)
    if not isinstance(all_stats, dict):
        return {}
    value_cols = list(params.get('value_cols') or [])
    stat_cols = params.get('stats_columns')
    if isinstance(stat_cols, dict):
        pairs = [(col, stat_cols.get(col)) for col in value_cols]
    elif stat_cols:
        pairs = list(zip(value_cols, stat_cols))
    else:
        pairs = [(col, col) for col in value_cols]
    return {plot_col: all_stats[res_col]
            for plot_col, res_col in pairs
            if res_col and res_col in all_stats}


def _build_multi_figure(spec, processed, results):
    params = spec['params']
    settings = spec['settings']
    viz = make_visualizer(settings)
    value_cols = list(params.get('value_cols') or [])
    if not value_cols:
        raise ValueError("Saved figure has no selected columns")
    data, group_col = _prepare_spec_data(processed, params)
    stats = _spec_stats(results, params)
    log_scale = bool(params.get('log_scale', False))
    design = params.get('design', 'one')

    if design == 'three':
        return viz.plot_threeway_multi_panel(
            data=data, value_cols=value_cols,
            factor_a_col=params['factor_a_col'],
            factor_b_col=params['factor_b_col'],
            factor_c_col=params['factor_c_col'],
            threeway_results=stats, ncols=params.get('ncols', 6),
            factor_a_name=params.get('factor_a_name'),
            factor_b_name=params.get('factor_b_name'),
            factor_c_name=params.get('factor_c_name'),
            ylabel=params.get('ylabel', 'Concentration'),
            show_points=settings.get('show_points', True),
            plot_type=settings.get('plot_type', 'box'),
            log_scale=log_scale,
        )
    if design == 'two':
        return viz.plot_twoway_multi_panel(
            data=data, value_cols=value_cols,
            factor_a_col=params['factor_a_col'],
            factor_b_col=params['factor_b_col'],
            twoway_results=stats, ncols=params.get('ncols', 6),
            factor_a_name=params.get('factor_a_name'),
            factor_b_name=params.get('factor_b_name'),
            ylabel=params.get('ylabel', 'Concentration'),
            show_points=settings.get('show_points', True),
            plot_type=settings.get('plot_type', 'box'),
            log_scale=log_scale,
        )
    return viz.plot_multi_panel_groups_with_stats(
        data, value_cols, group_col, stats,
        ncols=params.get('ncols', 6),
        plot_type=settings.get('plot_type', 'box'),
        log_scale=log_scale,
        show_points=settings.get('show_points', True),
        ylabel=params.get('ylabel', 'Concentration'),
    )


def _build_interaction_figure(spec, processed, results):
    params = spec['params']
    settings = spec['settings']
    viz = make_visualizer(settings)
    value_cols = list(params.get('value_cols') or [])
    data, _ = _prepare_spec_data(processed, params)
    stats = _spec_stats(results, params)
    if params.get('design') == 'three':
        return viz.plot_threeway_interaction_multi_panel(
            data=data, value_cols=value_cols,
            factor_a_col=params['factor_a_col'],
            factor_b_col=params['factor_b_col'],
            factor_c_col=params['factor_c_col'],
            threeway_results=stats, ncols=params.get('ncols', 6),
            factor_a_name=params.get('factor_a_name'),
            factor_b_name=params.get('factor_b_name'),
            factor_c_name=params.get('factor_c_name'),
            ylabel=params.get('ylabel', 'Concentration'),
        )
    return viz.plot_twoway_interaction_multi_panel(
        data=data, value_cols=value_cols,
        factor_a_col=params['factor_a_col'],
        factor_b_col=params['factor_b_col'],
        twoway_results=stats, ncols=params.get('ncols', 6),
        factor_a_name=params.get('factor_a_name'),
        factor_b_name=params.get('factor_b_name'),
        ylabel=params.get('ylabel', 'Concentration'),
    )


def _build_composition_data(processed, params):
    data = _filtered_sample_data(processed)
    pct_cols = list(params.get('pct_cols') or [])
    factor_cols = [c for c in (params.get('factor_cols') or []) if c in data.columns]
    if factor_cols:
        group_series = data[factor_cols[0]].astype(str)
        for col in factor_cols[1:]:
            group_series = group_series + ' - ' + data[col].astype(str)
        group_col = '_factorial_group_'
    else:
        group_col = params.get('group_col') or processed.structure.group_col
        if group_col not in data.columns:
            raise ValueError("No grouping column for composition figure")
        group_series = data[group_col]
    missing = [c for c in pct_cols if c not in processed.percentages.columns]
    if missing:
        raise ValueError(f"Missing percentage columns: {missing}")
    values = processed.percentages.loc[data.index, pct_cols].reset_index(drop=True)
    meta = group_series.reset_index(drop=True).rename(group_col)
    return pd.concat([meta, values], axis=1), group_col


def _build_category_data(processed, params):
    data = _filtered_sample_data(processed)
    factor_cols = [c for c in (params.get('factor_cols') or []) if c in data.columns]
    if factor_cols:
        group_series = data[factor_cols[0]].astype(str)
        for col in factor_cols[1:]:
            group_series = group_series + ' - ' + data[col].astype(str)
        group_col = '_factorial_group_'
        data = data.copy()
        data[group_col] = group_series
        return data, group_col
    group_col = params.get('group_col') or processed.structure.group_col
    return data, group_col


def _build_composition_figure(spec, processed):
    params = spec['params']
    settings = spec['settings']
    viz = make_visualizer(settings)
    data, group_col = _build_composition_data(processed, params)
    pct_cols = list(params.get('pct_cols') or [])
    kind = spec['kind']
    if kind == 'composition_pie':
        return viz.plot_composition_pie_charts(
            data, group_col, pct_cols,
            title=params.get('title'),
            top_n=params.get('top_n', 7),
            other_threshold=params.get('other_threshold', 2.0),
        )
    if kind == 'composition_bars':
        return viz.plot_composition_horizontal_bars(
            data, group_col, pct_cols,
            title=params.get('title'),
            top_n=params.get('top_n', 15),
            show_values=params.get('show_values', True),
        )
    return viz.plot_composition_stacked_horizontal(
        data, group_col, pct_cols,
        title=params.get('title'),
        top_n=params.get('top_n', 12),
    )


def _build_category_figure(spec, processed):
    params = spec['params']
    viz = make_visualizer(spec['settings'])
    data, group_col = _build_category_data(processed, params)
    return viz.plot_category_composition_pie_charts(
        data=data,
        group_col=group_col,
        bile_acid_cols=list(params.get('bile_acid_cols') or processed.structure.bile_acid_cols),
        title=params.get('title', 'Bile Acid Category Composition by Group'),
    )


def _build_heatmap_figure(spec, processed):
    params = spec['params']
    viz = make_visualizer(spec['settings'])
    data = _filtered_sample_data(processed)
    group_col = params.get('group_col') or processed.structure.group_col
    factor_cols = params.get('factor_cols') or {group_col: group_col}
    return viz.plot_faceted_heatmap(
        sample_data=data,
        totals=processed.totals.loc[data.index],
        group_col=group_col,
        selected_categories=params.get('selected_categories') or {},
        category_display_names=ANALYSIS_GROUP_DISPLAY_NAMES,
        normalization=params.get('normalization', 'zscore'),
        cmap=params.get('cmap', 'RdBu_r'),
        sample_id_col=params.get('sample_id_col') or processed.structure.sample_id_col,
        factor_cols=factor_cols,
    )


def build_figure_from_spec(spec, processed, results):
    """Rebuild one saved figure.  Caller closes it after writing the PDF."""
    kind = spec.get('kind')
    if kind in ('oneway_multi', 'twoway_multi', 'threeway_multi'):
        return _build_multi_figure(spec, processed, results)
    if kind in ('twoway_interaction', 'threeway_interaction'):
        return _build_interaction_figure(spec, processed, results)
    if kind in ('composition_pie', 'composition_bars', 'composition_stacked'):
        return _build_composition_figure(spec, processed)
    if kind == 'category_pie':
        return _build_category_figure(spec, processed)
    if kind == 'heatmap':
        return _build_heatmap_figure(spec, processed)
    raise ValueError(f"Unknown saved figure kind: {kind}")


def _make_spec(kind, settings, params):
    return {
        'kind': kind,
        'settings': copy.deepcopy(settings),
        'params': copy.deepcopy(params),
    }


def iter_default_export_specs(processed, results, settings):
    """Recipes for figures whose tabs were never opened.

    These complement iter_all_export_figures(), which supplies the standard
    concentration/total/percentage/ratio variants.  Viewed recipes are merged
    first, so a user-customized version always wins over these defaults.
    """
    group_col = processed.structure.group_col
    if not group_col or not results:
        return
    is_threeway = bool(getattr(results, 'is_threeway', False))
    is_twoway = bool(getattr(results, 'is_twoway', False))
    valid_data = _filtered_sample_data(processed)
    valid_index = valid_data.index
    if is_threeway:
        design = 'three'
        factor_cols = [results.factor_a_col, results.factor_b_col, results.factor_c_col]
        factor_names = [results.factor_a_name, results.factor_b_name, results.factor_c_name]
    elif is_twoway:
        design = 'two'
        factor_cols = [results.factor_a_col, results.factor_b_col]
        factor_names = [results.factor_a_name, results.factor_b_name]
    else:
        design = 'one'
        factor_cols = []
        factor_names = []

    common = {
        'design': design,
        'group_col': group_col,
        'factor_cols': factor_cols,
        'factor_a_col': factor_cols[0] if factor_cols else '',
        'factor_b_col': factor_cols[1] if len(factor_cols) > 1 else '',
        'factor_c_col': factor_cols[2] if len(factor_cols) > 2 else '',
        'factor_a_name': factor_names[0] if factor_names else '',
        'factor_b_name': factor_names[1] if len(factor_names) > 1 else '',
        'factor_c_name': factor_names[2] if len(factor_names) > 2 else '',
        'show_points': settings.get('show_points', True),
        'plot_type': settings.get('plot_type', 'box'),
    }

    # Pool-composition figures (always shown in the Percentages tab).
    pct_cols = [c for c in processed.percentages.columns if c.endswith('_pct')]
    if pct_cols and (is_threeway or is_twoway):
        # The standard generator covers one-way percentage plots; factorial
        # designs need this default because that tab may never be opened.
        top_pcts = processed.percentages.loc[valid_index, pct_cols].mean().nlargest(10).index.tolist()
        display_names = [c.replace('_pct', '') for c in top_pcts]
        name = 'percentages_threeway' if is_threeway else 'percentages_twoway'
        yield name, _make_spec(
            'threeway_multi' if is_threeway else 'twoway_multi', settings,
            {
                **common, 'data_source': 'percentages',
                'value_cols': display_names, 'source_cols': top_pcts,
                'stats_source_raw': _result_source_name(results, 'percentages'),
                'stats_columns': top_pcts,
                'ylabel': '% of Total BA', 'ncols': 6, 'log_scale': False,
            },
        )
    if pct_cols:
        base = {**common, 'data_source': 'composition', 'pct_cols': pct_cols}
        yield 'percentages_pie', _make_spec('composition_pie', settings, {
            **base, 'title': 'Bile Acid Pool Composition by Group',
            'top_n': 7, 'other_threshold': 2.0,
        })
        yield 'percentages_bars', _make_spec('composition_bars', settings, {
            **base, 'title': 'Bile Acid Pool Composition Comparison',
            'top_n': 15, 'show_values': True,
        })
        yield 'percentages_stacked', _make_spec('composition_stacked', settings, {
            **base, 'title': 'Complete Bile Acid Pool Composition', 'top_n': 12,
        })

    # Category-composition pie grid (on-demand in the Totals tab).
    if processed.structure.bile_acid_cols:
        yield 'category_composition_pie', _make_spec('category_pie', settings, {
            **common,
            'bile_acid_cols': list(processed.structure.bile_acid_cols),
            'title': 'Bile Acid Category Composition by Group',
        })

    # Hydrophobicity figures.
    hydro_df = st.session_state.get('hydrophobicity')
    if hydro_df is not None and not hydro_df.empty:
        hydro_valid = hydro_df.loc[hydro_df.index.intersection(valid_index)]
        hydro_cols = [c for c in hydro_df.columns if not hydro_valid[c].isna().all()]
        if hydro_cols:
            hydro_name = ('hydrophobicity_threeway' if is_threeway
                          else 'hydrophobicity_twoway' if is_twoway
                          else 'hydrophobicity')
            yield hydro_name, _make_spec(
                'threeway_multi' if is_threeway else 'twoway_multi' if is_twoway else 'oneway_multi',
                settings,
                {
                    **common,
                    'data_source': 'hydro',
                    'value_cols': hydro_cols,
                    'source_cols': hydro_cols,
                    'stats_source_raw': _result_source_name(results, 'hydrophobicity'),
                    'stats_columns': hydro_cols,
                    'ylabel': 'HI Score',
                    'ncols': 2,
                    'log_scale': False,
                },
            )

    # Default heatmap, matching the Heatmap tab's initial choices.
    available_bas = set(processed.structure.bile_acid_cols)
    available_categories = {}
    for cat_key, ba_list in ANALYSIS_GROUPS.items():
        present = [ba for ba in ba_list if ba in available_bas]
        if present and cat_key in processed.totals.columns:
            available_categories[ANALYSIS_GROUP_DISPLAY_NAMES.get(cat_key, cat_key)] = cat_key
    if available_categories:
        names = sorted(available_categories)
        selected_display = [n for n in ["Primary", "Secondary", "Oxidized (Keto)", "Epimerized (Iso)"]
                            if n in names] or names[:3]
        selected_cats = {
            available_categories[name]: [ba for ba in ANALYSIS_GROUPS[available_categories[name]]
                                           if ba in available_bas]
            for name in selected_display
        }
        factor_dict = processed.structure.factors or {group_col: group_col}
        yield 'faceted_heatmap', _make_spec('heatmap', settings, {
            **common,
            'selected_categories': selected_cats,
            'normalization': 'zscore',
            'cmap': 'RdBu_r',
            'factor_cols': factor_dict,
            'sample_id_col': processed.structure.sample_id_col,
        })

    # Interaction figures for the default selections.
    if is_threeway:
        top_conc = processed.concentrations.mean().nlargest(10).index.tolist()
        if top_conc:
            yield 'concentrations_threeway_interaction', _make_spec('threeway_interaction', settings, {
                **common, 'data_source': 'sample',
                'value_cols': top_conc, 'source_cols': top_conc,
                'stats_source_raw': _result_source_name(results, 'individual_ba'),
                'stats_columns': top_conc,
                'ylabel': f'Concentration ({settings.get("units", "nmol/L")})',
                'ncols': 6,
            })
        ratio_valid = processed.ratios.loc[valid_index]
        ratio_cols = [c for c in processed.ratios.columns if not ratio_valid[c].isna().all()][:9]
        if ratio_cols:
            yield 'ratios_threeway_interaction', _make_spec('threeway_interaction', settings, {
                **common, 'data_source': 'ratios',
                'value_cols': ratio_cols, 'source_cols': ratio_cols,
                'stats_source_raw': _result_source_name(results, 'ratios'),
                'stats_columns': ratio_cols,
                'ylabel': 'Ratio', 'ncols': 6,
            })
    elif is_twoway:
        # Two-way concentration interactions are already yielded by
        # iter_all_export_figures() for its standard selections.  Ratio
        # interactions are not, so they need a default recipe here.
        ratio_valid = processed.ratios.loc[valid_index]
        ratio_cols = [c for c in processed.ratios.columns if not ratio_valid[c].isna().all()][:9]
        if ratio_cols:
            yield 'ratios_interaction', _make_spec('twoway_interaction', settings, {
                **common, 'data_source': 'ratios',
                'value_cols': ratio_cols, 'source_cols': ratio_cols,
                'stats_source_raw': _result_source_name(results, 'ratios'),
                'stats_columns': ratio_cols,
                'ylabel': 'Ratio', 'ncols': 6,
            })


def iter_export_figures(processed, results, settings, errors=None):
    """Yield all export figures, preferring the exact recipes users viewed.

    ``errors`` may be a list; recipe rebuild failures are appended there so the
    export tab can warn instead of silently omitting a viewed figure.
    """
    seen = set()
    specs = st.session_state.get('figure_specs') or {}

    for name, spec in specs.items():
        if name in seen:
            continue
        try:
            fig = build_figure_from_spec(spec, processed, results)
        except Exception as exc:
            if errors is not None:
                errors.append(f"{name} (viewed recipe): {exc}")
            continue
        if fig is None:
            continue
        seen.add(name)
        yield name, fig

    for name, spec in iter_default_export_specs(processed, results, settings):
        if name in seen:
            continue
        try:
            fig = build_figure_from_spec(spec, processed, results)
        except Exception as exc:
            if errors is not None:
                errors.append(f"{name} (default recipe): {exc}")
            continue
        if fig is None:
            continue
        seen.add(name)
        yield name, fig

    for name, fig in iter_all_export_figures(processed, results, settings):
        if name in seen:
            continue
        seen.add(name)
        yield name, fig


def iter_all_export_figures(processed, results, settings):
    """Yield (name, figure) one at a time so export never holds every figure."""
    from config.bile_acid_species import get_primary, get_secondary, get_glycine_conjugated, get_taurine_conjugated
    
    viz = make_visualizer(settings)
    group_col = processed.structure.group_col
    
    if not group_col or not results:
        return
    
    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    data = data[data[group_col].astype(str).str.lower() != 'nan']
    available_bas = processed.structure.bile_acid_cols

    is_threeway = getattr(results, 'is_threeway', False)
    is_twoway = results.is_twoway

    if is_threeway:
        # =====================================================================
        # THREE-WAY ANOVA EXPORT FIGURES
        # =====================================================================
        fa_col, fb_col, fc_col = results.factor_a_col, results.factor_b_col, results.factor_c_col
        fa_name, fb_name, fc_name = results.factor_a_name, results.factor_b_name, results.factor_c_name

        conc_selections = {
            'top10': processed.concentrations.mean().nlargest(10).index.tolist(),
            'significant': [b for b in available_bas if b in results.threeway_individual_ba
                           and _threeway_any_effect(results.threeway_individual_ba[b], settings['alpha'])],
            'primary': [b for b in get_primary() if b in available_bas],
            'secondary': [b for b in get_secondary() if b in available_bas],
        }

        for sel_name, selected in conc_selections.items():
            if not selected:
                continue
            tw_stats_raw = {b: results.threeway_individual_ba.get(b) for b in selected
                           if b in results.threeway_individual_ba}
            tw_stats_log = {b: results.log_threeway_individual_ba.get(b) for b in selected
                           if b in results.log_threeway_individual_ba}
            for log_scale in [False, True]:
                suffix = f"_{sel_name}{'_log' if log_scale else ''}"
                try:
                    fig = viz.plot_threeway_multi_panel(
                        data=data, value_cols=selected,
                        factor_a_col=fa_col, factor_b_col=fb_col, factor_c_col=fc_col,
                        threeway_results=tw_stats_log if log_scale else tw_stats_raw, ncols=6,
                        factor_a_name=fa_name, factor_b_name=fb_name, factor_c_name=fc_name,
                        ylabel=f'Concentration ({settings["units"]})', show_points=settings['show_points'],
                        plot_type=settings['plot_type'], log_scale=log_scale)
                    yield f'concentrations{suffix}', fig
                except Exception:
                    pass

        key_totals = ['total_all', 'total_primary', 'total_secondary', 'total_conjugated',
                      'total_unconjugated', 'glycine_conjugated', 'taurine_conjugated',
                      'sulfated', 'oxidized', 'epimerized', 'nor_bile_acids',
                      'total_12alpha_hydroxylated', 'total_non12alpha_hydroxylated',
                      'primary_conjugated', 'secondary_conjugated',
                      'primary_unconjugated', 'secondary_unconjugated']
        available_totals = [t for t in key_totals if t in processed.totals.columns]

        if available_totals:
            totals_data = pd.concat([data[[fa_col, fb_col, fc_col]].reset_index(drop=True),
                                     processed.totals[available_totals].loc[data.index].reset_index(drop=True)], axis=1)
            tw_stats_raw = {t: results.threeway_totals.get(t) for t in available_totals
                           if t in results.threeway_totals}
            tw_stats_log = {t: results.log_threeway_totals.get(t) for t in available_totals
                           if t in results.log_threeway_totals}
            for log_scale in [False, True]:
                suffix = '_log' if log_scale else ''
                try:
                    fig = viz.plot_threeway_multi_panel(
                        data=totals_data, value_cols=available_totals,
                        factor_a_col=fa_col, factor_b_col=fb_col, factor_c_col=fc_col,
                        threeway_results=tw_stats_log if log_scale else tw_stats_raw, ncols=6,
                        factor_a_name=fa_name, factor_b_name=fb_name, factor_c_name=fc_name,
                        ylabel=f'Concentration ({settings["units"]})', show_points=settings['show_points'],
                        plot_type=settings['plot_type'], log_scale=log_scale)
                    yield f'totals{suffix}', fig
                except Exception:
                    pass

        ratio_cols = [col for col in processed.ratios.columns if not processed.ratios[col].isna().all()]
        if ratio_cols:
            ratio_data = pd.concat([data[[fa_col, fb_col, fc_col]].reset_index(drop=True),
                                    processed.ratios[ratio_cols].loc[data.index].reset_index(drop=True)], axis=1)
            tw_stats_raw = {r: results.threeway_ratios.get(r) for r in ratio_cols[:9]
                           if r in results.threeway_ratios}
            tw_stats_log = {r: results.log_threeway_ratios.get(r) for r in ratio_cols[:9]
                           if r in results.log_threeway_ratios}
            for log_scale in [False, True]:
                suffix = '_log' if log_scale else ''
                try:
                    fig = viz.plot_threeway_multi_panel(
                        data=ratio_data, value_cols=ratio_cols[:9],
                        factor_a_col=fa_col, factor_b_col=fb_col, factor_c_col=fc_col,
                        threeway_results=tw_stats_log if log_scale else tw_stats_raw, ncols=6,
                        factor_a_name=fa_name, factor_b_name=fb_name, factor_c_name=fc_name,
                        ylabel='Ratio', show_points=settings['show_points'],
                        plot_type=settings['plot_type'], log_scale=log_scale)
                    yield f'ratios{suffix}', fig
                except Exception:
                    pass

    elif is_twoway:
        # =====================================================================
        # TWO-WAY ANOVA EXPORT FIGURES
        # =====================================================================
        fa_col = results.factor_a_col
        fb_col = results.factor_b_col
        fa_name = results.factor_a_name
        fb_name = results.factor_b_name

        # --- Concentrations ---
        conc_selections = {
            'top10': processed.concentrations.mean().nlargest(10).index.tolist(),
            'significant': [b for b in available_bas if b in results.twoway_individual_ba
                           and _twoway_any_effect(results.twoway_individual_ba[b], settings['alpha'])],
            'primary': [b for b in get_primary() if b in available_bas],
            'secondary': [b for b in get_secondary() if b in available_bas],
        }

        for sel_name, selected in conc_selections.items():
            if not selected:
                continue
            tw_stats_raw = {b: results.twoway_individual_ba.get(b) for b in selected
                           if b in results.twoway_individual_ba}
            tw_stats_log = {b: results.log_twoway_individual_ba.get(b) for b in selected
                           if b in results.log_twoway_individual_ba}
            for log_scale in [False, True]:
                suffix = f"_{sel_name}{'_log' if log_scale else ''}"
                try:
                    fig = viz.plot_twoway_multi_panel(
                        data=data, value_cols=selected,
                        factor_a_col=fa_col, factor_b_col=fb_col,
                        twoway_results=tw_stats_log if log_scale else tw_stats_raw, ncols=6,
                        factor_a_name=fa_name, factor_b_name=fb_name,
                        ylabel=f'Concentration ({settings["units"]})', show_points=settings['show_points'],
                        plot_type=settings['plot_type'], log_scale=log_scale)
                    yield f'concentrations{suffix}', fig
                except Exception:
                    pass
            try:
                fig_int = viz.plot_twoway_interaction_multi_panel(
                    data=data, value_cols=selected,
                    factor_a_col=fa_col, factor_b_col=fb_col,
                    twoway_results=tw_stats_raw, ncols=6,
                    factor_a_name=fa_name, factor_b_name=fb_name,
                    ylabel=f'Concentration ({settings["units"]})')
                yield f'concentrations_{sel_name}_interaction', fig_int
            except Exception:
                pass

        # --- Totals ---
        key_totals = ['total_all', 'total_primary', 'total_secondary', 'total_conjugated',
                      'total_unconjugated', 'glycine_conjugated', 'taurine_conjugated',
                      'sulfated', 'oxidized', 'epimerized', 'nor_bile_acids',
                      'total_12alpha_hydroxylated', 'total_non12alpha_hydroxylated',
                      'primary_conjugated', 'secondary_conjugated',
                      'primary_unconjugated', 'secondary_unconjugated']
        available_totals = [t for t in key_totals if t in processed.totals.columns]

        if available_totals:
            totals_data = pd.concat([data[[fa_col, fb_col]].reset_index(drop=True),
                                     processed.totals[available_totals].loc[data.index].reset_index(drop=True)], axis=1)
            tw_stats_raw = {t: results.twoway_totals.get(t) for t in available_totals
                           if t in results.twoway_totals}
            tw_stats_log = {t: results.log_twoway_totals.get(t) for t in available_totals
                           if t in results.log_twoway_totals}
            for log_scale in [False, True]:
                suffix = '_log' if log_scale else ''
                try:
                    fig = viz.plot_twoway_multi_panel(
                        data=totals_data, value_cols=available_totals,
                        factor_a_col=fa_col, factor_b_col=fb_col,
                        twoway_results=tw_stats_log if log_scale else tw_stats_raw, ncols=6,
                        factor_a_name=fa_name, factor_b_name=fb_name,
                        ylabel=f'Concentration ({settings["units"]})', show_points=settings['show_points'],
                        plot_type=settings['plot_type'], log_scale=log_scale)
                    yield f'totals{suffix}', fig
                except Exception:
                    pass
            try:
                fig_int = viz.plot_twoway_interaction_multi_panel(
                    data=totals_data, value_cols=available_totals,
                    factor_a_col=fa_col, factor_b_col=fb_col,
                    twoway_results=tw_stats_raw, ncols=6,
                    factor_a_name=fa_name, factor_b_name=fb_name,
                    ylabel=f'Concentration ({settings["units"]})')
                yield 'totals_interaction', fig_int
            except Exception:
                pass

        # --- Ratios ---
        ratio_cols = [col for col in processed.ratios.columns if not processed.ratios[col].isna().all()]
        if ratio_cols:
            ratio_data = pd.concat([data[[fa_col, fb_col]].reset_index(drop=True),
                                    processed.ratios[ratio_cols].loc[data.index].reset_index(drop=True)], axis=1)
            tw_stats_raw = {r: results.twoway_ratios.get(r) for r in ratio_cols[:9]
                           if r in results.twoway_ratios}
            tw_stats_log = {r: results.log_twoway_ratios.get(r) for r in ratio_cols[:9]
                           if r in results.log_twoway_ratios}
            for log_scale in [False, True]:
                suffix = '_log' if log_scale else ''
                try:
                    fig = viz.plot_twoway_multi_panel(
                        data=ratio_data, value_cols=ratio_cols[:9],
                        factor_a_col=fa_col, factor_b_col=fb_col,
                        twoway_results=tw_stats_log if log_scale else tw_stats_raw, ncols=6,
                        factor_a_name=fa_name, factor_b_name=fb_name,
                        ylabel='Ratio', show_points=settings['show_points'],
                        plot_type=settings['plot_type'], log_scale=log_scale)
                    yield f'ratios{suffix}', fig
                except Exception:
                    pass

    else:
        # =====================================================================
        # ONE-WAY EXPORT FIGURES (unchanged)
        # =====================================================================

        # === CONCENTRATIONS TAB FIGURES ===
        conc_selections = {
            'top10': processed.concentrations.mean().nlargest(10).index.tolist(),
            'significant': [b for b in available_bas if b in results.individual_ba_results
                           and results.individual_ba_results[b].main_test.significant],
            'primary': [b for b in get_primary() if b in available_bas],
            'secondary': [b for b in get_secondary() if b in available_bas],
        }

        for sel_name, selected in conc_selections.items():
            if not selected:
                continue
            stats_raw = {b: results.individual_ba_results.get(b) for b in selected if b in results.individual_ba_results}
            stats_log = {b: results.log_individual_ba_results.get(b) for b in selected if b in results.log_individual_ba_results}

            for log_scale in [False, True]:
                suffix = f"_{sel_name}{'_log' if log_scale else ''}"
                try:
                    fig = viz.plot_multi_panel_groups_with_stats(
                        data, selected, group_col, stats_log if log_scale else stats_raw, ncols=6,
                        plot_type=settings['plot_type'], log_scale=log_scale,
                        show_points=settings['show_points'], ylabel=f'Concentration ({settings["units"]})')
                    yield f'concentrations{suffix}', fig
                except Exception:
                    pass

        # === TOTALS TAB FIGURES ===
        totals_combined = pd.concat([data[[group_col]], processed.totals.loc[data.index]], axis=1)

        key_totals = ['total_all', 'total_primary', 'total_secondary', 'total_conjugated',
                      'total_unconjugated', 'glycine_conjugated', 'taurine_conjugated',
                      'sulfated', 'oxidized', 'epimerized', 'nor_bile_acids',
                      'total_12alpha_hydroxylated', 'total_non12alpha_hydroxylated',
                      'primary_conjugated', 'secondary_conjugated',
                      'primary_unconjugated', 'secondary_unconjugated']
        available_totals = [t for t in key_totals if t in totals_combined.columns]

        if available_totals:
            totals_stats_raw = {t: results.totals_results.get(t) for t in available_totals if t in results.totals_results}
            totals_stats_log = {t: results.log_totals_results.get(t) for t in available_totals if t in results.log_totals_results}
            for log_scale in [False, True]:
                suffix = '_log' if log_scale else ''
                try:
                    fig = viz.plot_multi_panel_groups_with_stats(
                        totals_combined, available_totals, group_col,
                        totals_stats_log if log_scale else totals_stats_raw, ncols=6,
                        plot_type=settings['plot_type'], log_scale=log_scale,
                        show_points=settings['show_points'], ylabel=f'Concentration ({settings["units"]})')
                    yield f'totals{suffix}', fig
                except Exception:
                    pass

        # === PERCENTAGES TAB FIGURES ===
        pct_cols = [col for col in processed.percentages.columns if col.endswith('_pct')]
        if pct_cols:
            top_pct = processed.percentages[pct_cols].mean().nlargest(10).index.tolist()
            pct_combined = pd.concat([data[[group_col]], processed.percentages.loc[data.index, top_pct]], axis=1)

            display_cols = [col.replace('_pct', '') for col in top_pct]
            pct_display = pct_combined.rename(columns={old: new for old, new in zip(top_pct, display_cols)})

            pct_stats = {}
            for pct_col, disp_col in zip(top_pct, display_cols):
                if pct_col in results.percentages_results:
                    pct_stats[disp_col] = results.percentages_results[pct_col]

            try:
                fig = viz.plot_multi_panel_groups_with_stats(
                    pct_display, display_cols, group_col, pct_stats, ncols=6,
                    plot_type=settings['plot_type'], log_scale=False,
                    show_points=settings['show_points'], ylabel='% of Total BA')
                yield 'percentages_top10', fig
            except Exception:
                pass

        # === RATIOS TAB FIGURES ===
        ratio_cols = [col for col in processed.ratios.columns if not processed.ratios[col].isna().all()]
        if ratio_cols:
            ratios_combined = pd.concat([data[[group_col]], processed.ratios.loc[data.index, ratio_cols]], axis=1)
            ratios_stats_raw = {r: results.ratios_results.get(r) for r in ratio_cols if r in results.ratios_results}
            ratios_stats_log = {r: results.log_ratios_results.get(r) for r in ratio_cols if r in results.log_ratios_results}

            for log_scale in [False, True]:
                suffix = '_log' if log_scale else ''
                try:
                    fig = viz.plot_multi_panel_groups_with_stats(
                        ratios_combined, ratio_cols[:9], group_col,
                        ratios_stats_log if log_scale else ratios_stats_raw, ncols=6,
                        plot_type=settings['plot_type'], log_scale=log_scale,
                        show_points=settings['show_points'], ylabel='Ratio')
                    yield f'ratios{suffix}', fig
                except Exception:
                    pass

    return


def create_results_zip(processed, results, figures, report_gen, settings=None):
    """Create a ZIP on disk, consuming figures one at a time.

    Returns ``(zip_path, figure_count, errors)``.  Writing to a temporary file and
    closing each figure immediately keeps peak memory bounded to one workbook or
    one figure at a time.
    """
    # Get metadata columns
    id_cols = []
    if processed.structure.sample_id_col and processed.structure.sample_id_col in processed.sample_data.columns:
        id_cols.append(processed.structure.sample_id_col)
    if processed.structure.group_col and processed.structure.group_col in processed.sample_data.columns:
        id_cols.append(processed.structure.group_col)
    # Include Factor_ columns
    for factor_col in processed.structure.factors.values():
        if factor_col in processed.sample_data.columns and factor_col not in id_cols:
            id_cols.append(factor_col)

    metadata = processed.sample_data[id_cols] if id_cols else pd.DataFrame(index=processed.sample_data.index)

    tmp = tempfile.NamedTemporaryFile(prefix='bile_acid_results_', suffix='.zip', delete=False)
    zip_path = tmp.name
    tmp.close()

    errors = []
    figure_count = 0
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Data files with metadata
            for name, df in [('concentrations', processed.concentrations),
                             ('percentages', processed.percentages),
                             ('totals', processed.totals),
                             ('ratios', processed.ratios)]:
                try:
                    export_df = pd.concat([metadata, df], axis=1)
                    zf.writestr(f'data/{name}.xlsx', df_to_excel_bytes(export_df))
                    del export_df
                except Exception as exc:
                    errors.append(f'{name} workbook: {exc}')

            # Full data
            try:
                full = pd.concat([processed.sample_data, processed.totals, processed.ratios, processed.percentages], axis=1)
                zf.writestr('data/full_analysis.xlsx', df_to_excel_bytes(full))
                del full
            except Exception as exc:
                errors.append(f'full analysis workbook: {exc}')

            # LOD-highlighted Excel
            try:
                lod_handling = settings.get('lod_handling', 'lod') if settings else 'lod'
                lod_excel_bytes = create_full_data_excel_with_highlighting(processed, lod_handling)
                zf.writestr('data/full_data_lod_highlighted.xlsx', lod_excel_bytes)
                del lod_excel_bytes
            except Exception as exc:
                errors.append(f'LOD-highlighted workbook: {exc}')

            # Excel report. Write it to a temporary file rather than holding a
            # second full workbook in memory.
            if report_gen:
                report_tmp = tempfile.NamedTemporaryFile(prefix='bile_acid_report_', suffix='.xlsx', delete=False)
                report_path = report_tmp.name
                report_tmp.close()
                try:
                    report_gen.display_orders = report_display_orders(settings or {})
                    report_gen.save_excel_report(report_path)
                    zf.write(report_path, 'reports/statistical_report.xlsx')
                except Exception as exc:
                    errors.append(f'statistical report: {exc}')
                finally:
                    _remove_file(report_path)
                    # The workbook is already in the ZIP; drop the sheet copies
                    # so they are not retained while figures are generated.
                    report_gen.analysis_sheets.clear()

            # Figures are yielded by a generator; close each one before the next.
            figure_iter = figures.items() if isinstance(figures, dict) else figures
            try:
                for name, fig in figure_iter:
                    if not fig:
                        continue
                    try:
                        zf.writestr(f'figures/{name}.pdf', fig_to_bytes(fig, 'pdf'))
                        figure_count += 1
                    except Exception as exc:
                        errors.append(f'{name} figure: {exc}')
                    finally:
                        plt.close(fig)
            except Exception as exc:
                errors.append(f'figure generation: {exc}')
    except Exception:
        _remove_file(zip_path)
        raise

    return zip_path, figure_count, errors


def create_full_data_excel_with_highlighting(processed, lod_handling='lod') -> bytes:
    """Create Excel workbook with yellow highlighting on cells where LOD values were replaced."""
    from openpyxl.styles import PatternFill
    
    lod_display = { "lod": "LOD value","half_lod": "LOD/2", "zero": "Zero", "drop": "NaN (excluded)"}
    
    buf = BytesIO()
    
    # Get metadata columns
    id_cols = []
    if processed.structure.sample_id_col and processed.structure.sample_id_col in processed.sample_data.columns:
        id_cols.append(processed.structure.sample_id_col)
    if processed.structure.group_col and processed.structure.group_col in processed.sample_data.columns:
        id_cols.append(processed.structure.group_col)
    # Include Factor_ columns
    for factor_col in processed.structure.factors.values():
        if factor_col in processed.sample_data.columns and factor_col not in id_cols:
            id_cols.append(factor_col)

    metadata = processed.sample_data[id_cols] if id_cols else pd.DataFrame(index=processed.sample_data.index)

    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        # Concentrations sheet with highlighting
        conc_df = pd.concat([metadata, processed.concentrations], axis=1)
        conc_df.to_excel(writer, sheet_name='Concentrations', index=False)
        
        # Totals
        totals_df = pd.concat([metadata, processed.totals], axis=1)
        totals_df.to_excel(writer, sheet_name='Totals', index=False)
        
        # Ratios
        ratios_df = pd.concat([metadata, processed.ratios], axis=1)
        ratios_df.to_excel(writer, sheet_name='Ratios', index=False)
        
        # Percentages
        pct_df = pd.concat([metadata, processed.percentages], axis=1)
        pct_df.to_excel(writer, sheet_name='Percentages', index=False)
        
        # LOD Summary sheet
        lod_summary_rows = []
        for ba in processed.structure.bile_acid_cols:
            lod_val = processed.structure.analyte_lods.get(ba, 'N/A')
            count = processed.structure.analyte_lod_counts.get(ba, 0)
            n = len(processed.sample_data)
            
            # Compute replacement value based on handling method
            if isinstance(lod_val, (int, float)):
                if lod_handling == 'half_lod':
                    repl = lod_val / 2
                elif lod_handling == 'lod':
                    repl = lod_val
                elif lod_handling == 'zero':
                    repl = 0
                else:
                    repl = 'NaN'
            else:
                repl = 'N/A'
            
            lod_summary_rows.append({
                'Analyte': ba,
                'LOD (nM)': lod_val,
                'Replacement Value': repl,
                'Samples Below LOD': count,
                '% Below LOD': f"{count/n*100:.1f}" if n > 0 else "0",
                'Total Samples': n
            })
        lod_df = pd.DataFrame(lod_summary_rows)
        lod_df.to_excel(writer, sheet_name='LOD Summary', index=False)
        
        # Legend sheet
        legend_data = pd.DataFrame({
            'Item': ['Yellow cells', 'LOD Source', 'LOD Handling'],
            'Description': [
                f'Value was below LOD and replaced with {lod_display.get(lod_handling, lod_handling)}',
                processed.structure.lod_source,
                f'{lod_handling} ({lod_display.get(lod_handling, lod_handling)})'
            ]
        })
        legend_data.to_excel(writer, sheet_name='Legend', index=False)
        
        # Apply yellow highlighting to LOD-replaced cells on Concentrations sheet
        yellow_fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')
        ws = writer.sheets['Concentrations']
        
        # Map column names to Excel column indices (1-based, after metadata columns)
        col_headers = list(conc_df.columns)
        for ba, row_indices in processed.structure.analyte_lod_rows.items():
            if ba in col_headers:
                col_idx = col_headers.index(ba) + 1  # 1-based for openpyxl
                for row_idx in row_indices:
                    # +2 because Excel is 1-based and row 1 is the header
                    ws.cell(row=row_idx + 2, column=col_idx).fill = yellow_fill
    
    buf.seek(0)
    return buf.getvalue()


def render_sidebar():
    """Render sidebar settings."""
    st.sidebar.markdown("## ⚙️ Settings")
    st.sidebar.markdown("### 🔬 Sample Preparation")
    dilution_input = st.sidebar.number_input(
        "Dilution factor",
        min_value=0.0,
        value=float(st.session_state.get('dilution_factor', 1.0)),
        step=0.1,
        format="%.4g",
        help="Applied only to LOD replacement values (measured values are already dilution-corrected)."
    )
    if st.sidebar.button("Apply dilution factor"):
        st.session_state.dilution_factor = dilution_input
    dilution_factor = float(st.session_state.get('dilution_factor', 1.0))

    units_input = st.sidebar.text_input(
        "Concentration units",
        value=st.session_state.get('units', 'nmol/L'),
        help="Units label shown on concentration and total bile acid plots."
    )
    if st.sidebar.button("Apply units"):
        st.session_state.units = units_input
    units = st.session_state.get('units', 'nmol/L')

    st.sidebar.markdown("### 📊 Detection Limits")
    lod_handling = st.sidebar.selectbox(
        "Below-LOD handling",
        [ "lod", "half_lod", "zero", "drop"],
        format_func=lambda x: { "lod": "LOD value (recommended)", "half_lod": "LOD/2",
                               "zero": "Zero", "drop": "NaN (exclude)"}[x],
        help="LOD values are auto-detected per analyte from the LC-MS standards sheet"
    )
    lod_threshold = st.sidebar.slider(
        "LOD exclusion threshold (%)",
        min_value=0, max_value=100, value=50, step=5,
        help="Exclude individual analytes from statistical testing if ≥ this % of values "
             "are below LOD. Set to 0 to disable. Uses modified rule: analyte is "
             "kept if ANY group has ≥ (100 − threshold)% detected values."
    )
    aggregate_lod_threshold = st.sidebar.slider(
        "Aggregate LOD burden threshold (%)",
        min_value=0, max_value=100, value=5, step=1,
        help="Exclude aggregate totals from statistical testing when the maximum "
             "group LOD-bound replacement burden reaches this percentage. "
             "Burden = LOD-bound contribution / (observed + LOD-bound contribution). "
             "total_all is always tested; it only receives a warning. Set to 0 to disable."
    )
    
    st.sidebar.markdown("### 🧪 Hydrophobicity")
    tissue_options = {
        "blood": "Blood / Serum (pH 7.4)",
        "duodenum": "Duodenum (pH 6.0)",
        "jejunum": "Jejunum (pH 6.5)",
        "ileum": "Ileum (pH 7.0)",
        "cecum": "Cecum (pH 5.7)",
        "colon": "Colon (pH 6.5)",
        "stomach": "Stomach (pH 2.0)",
    }
    tissue = st.sidebar.selectbox(
        "Sample tissue",
        list(tissue_options.keys()),
        format_func=lambda x: tissue_options[x],
        help="Sets pH for hydrophobicity index calculation. At blood pH (7.4), all bile acids are >99% ionized."
    )

    st.sidebar.markdown("### 📈 Analysis")
    alpha = st.sidebar.slider("Significance (α)", 0.01, 0.10, 0.05, 0.01)
    fdr_enabled = st.sidebar.checkbox(
        "FDR-adjust primary tests (BH)",
        value=True,
        help="Apply Benjamini-Hochberg q-values across analytes within each "
             "section and scale (raw/log₁₀). q-significant omnibus results gate "
             "post-hoc comparisons; pairwise tests remain within-endpoint BH-adjusted."
    )
    plot_type = st.sidebar.selectbox("Plot type", ["box", "violin", "bar", "strip"])
    show_points = st.sidebar.checkbox("Show data points", True)
    
    # Color palette selection
    st.sidebar.markdown("### 🎨 Appearance")
    palette_options = {
        "Set2": "Set2 (Default - Soft pastels)",
        "Set1": "Set1 (Bold primary)",
        "Paired": "Paired (Light/dark pairs)",
        "Dark2": "Dark2 (Darker pastels)",
        "colorblind": "Colorblind-friendly",
        "tab10": "Tab10 (Matplotlib default)",
        "Pastel1": "Pastel1 (Very soft)",
        "Accent": "Accent (High contrast)"
    }
    color_palette = st.sidebar.selectbox("Color palette", list(palette_options.keys()),
                                         format_func=lambda x: palette_options[x])
    
    # Background style selection
    style_options = {
        "whitegrid": "White with grid",
        "white": "Clean white",
        "darkgrid": "Gray with grid",
        "ticks": "White with ticks"
    }
    plot_style = st.sidebar.selectbox("Plot background", list(style_options.keys()),
                                      format_func=lambda x: style_options[x])
    
    return {'lod_handling': lod_handling, 'lod_threshold': lod_threshold,
            'aggregate_lod_threshold': aggregate_lod_threshold,
            'fdr_enabled': fdr_enabled,
            'alpha': alpha, 'plot_type': plot_type,
            'show_points': show_points,
            'color_palette': color_palette, 'plot_style': plot_style,
            'dilution_factor': dilution_factor,
            'units': units,
            'tissue': tissue}


def render_concentrations_tab(processed, settings):
    """Render concentrations tab."""
    st.markdown("### Individual Bile Acid Concentrations")

    viz = make_visualizer(settings)
    group_col = processed.structure.group_col
    results = st.session_state.analysis_results

    if not group_col:
        st.warning("No group column detected.")
        return

    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    data = data[data[group_col].astype(str).str.lower() != 'nan']
    available_bas = processed.structure.bile_acid_cols
    is_threeway = results is not None and getattr(results, 'is_threeway', False)
    is_twoway = results is not None and results.is_twoway

    quick = st.segmented_control("Quick select",
        ["Top 10", "Significant", "Primary", "Primary Conjugated", "Primary Unconjugated", 
         "Secondary", "Conjugated", "Unconjugated", "Secondary Conjugated", "Secondary Unconjugated",
         "Glycine", "Taurine", "Sulfated", "Oxidized", "Epimerized",
         "12α-OH", "Non-12α-OH", "Nor", "Show All", "Custom"],
        default="Top 10")

    if quick == "Top 10":
        selected = processed.concentrations.mean().nlargest(10).index.tolist()
    elif quick == "Significant":
        if is_threeway:
            selected = [b for b in available_bas if b in results.threeway_individual_ba
                       and _threeway_any_effect(results.threeway_individual_ba[b], settings['alpha'])]
        elif is_twoway:
            selected = [b for b in available_bas if b in results.twoway_individual_ba
                       and _twoway_any_effect(results.twoway_individual_ba[b], settings['alpha'])]
        else:
            selected = [b for b in available_bas if b in results.individual_ba_results
                       and results.individual_ba_results[b].main_test.significant]
        if not selected:
            st.info("No significant individual BAs found.")
            selected = processed.concentrations.mean().nlargest(5).index.tolist()
    elif quick == "Primary":
        selected = [b for b in get_primary() if b in available_bas]
    elif quick == "Primary Conjugated":
        primary_list = get_primary()
        selected = [b for b in get_conjugated() if b in available_bas and b in primary_list]
    elif quick == "Primary Unconjugated":
        primary_list = get_primary()
        selected = [b for b in get_unconjugated() if b in available_bas and b in primary_list]
    elif quick == "Secondary":
        selected = [b for b in get_secondary() if b in available_bas]
    elif quick == "Secondary Conjugated":
        secondary_list = get_secondary()
        selected = [b for b in get_conjugated() if b in available_bas and b in secondary_list]
    elif quick == "Secondary Unconjugated":
        secondary_list = get_secondary()
        selected = [b for b in get_unconjugated() if b in available_bas and b in secondary_list]
    elif quick == "Conjugated":
        selected = [b for b in get_conjugated() if b in available_bas]
    elif quick == "Unconjugated":
        selected = [b for b in get_unconjugated() if b in available_bas]
    elif quick == "Glycine":
        selected = [b for b in get_glycine_conjugated() if b in available_bas]
    elif quick == "Taurine":
        selected = [b for b in get_taurine_conjugated() if b in available_bas]
    elif quick == "Sulfated":
        selected = [b for b in get_sulfated() if b in available_bas]
    elif quick == "Oxidized":
        selected = [b for b in get_keto_derivatives() if b in available_bas]
    elif quick == "Epimerized":
        selected = [b for b in get_iso_forms() if b in available_bas]
    elif quick == "12α-OH":
        selected = [b for b in get_12alpha_hydroxylated() if b in available_bas]
    elif quick == "Non-12α-OH":
        selected = [b for b in get_non12alpha_hydroxylated() if b in available_bas]
    elif quick == "Nor":
        selected = [b for b in get_nor_bile_acids() if b in available_bas]
    elif quick == "Show All":
        selected = available_bas
    else:
        selected = st.multiselect("Select BAs", available_bas, available_bas[:5])

    log_scale = st.checkbox("Log10 scale", key="conc_log")

    # Show LOD exclusion notice if any selected analytes were excluded
    lod_excluded = getattr(results, 'lod_excluded', {})
    if lod_excluded and selected:
        excluded_selected = {ba: pct for ba, pct in lod_excluded.items() if ba in selected}
        if excluded_selected:
            excl_list = ", ".join(f"{ba} ({pct}%)" for ba, pct in sorted(excluded_selected.items()))
            st.info(
                f"**Individual LOD exclusion** "
                f"(threshold: ≥{getattr(results, 'lod_threshold', 50)}% below LOD): "
                f"The following analytes were excluded from statistical testing: {excl_list}. "
                f"Plots are still shown but without ANOVA results."
            )

    lod_warnings = getattr(results, 'lod_warnings', {})
    if lod_warnings:
        warning_list = ", ".join(
            f"{key} ({info.get('max_group_burden_pct', 0):.1f}% max group burden)"
            for key, info in sorted(lod_warnings.items())
        )
        st.warning(
            "**Aggregate LOD warning:** total_all is always tested, but its "
            f"LOD-bound replacement burden reached the threshold for: {warning_list}."
        )

    if selected:
        if is_threeway:
            # === THREE-WAY ANOVA PLOTS ===
            fa_col, fb_col, fc_col = results.factor_a_col, results.factor_b_col, results.factor_c_col
            fa_name, fb_name, fc_name = results.factor_a_name, results.factor_b_name, results.factor_c_name

            st.markdown(f"**Three-way ANOVA**: {fa_name} x {fb_name} x {fc_name}")

            tw_stats_raw = {b: results.threeway_individual_ba.get(b) for b in selected
                           if b in results.threeway_individual_ba}
            tw_stats_log = {b: results.log_threeway_individual_ba.get(b) for b in selected
                           if b in results.log_threeway_individual_ba}
            tw_stats = tw_stats_log if log_scale else tw_stats_raw

            fig = viz.plot_threeway_multi_panel(
                data=data, value_cols=selected,
                factor_a_col=fa_col, factor_b_col=fb_col, factor_c_col=fc_col,
                threeway_results=tw_stats, ncols=6,
                factor_a_name=fa_name, factor_b_name=fb_name, factor_c_name=fc_name,
                ylabel=f'Concentration ({settings["units"]})', show_points=settings['show_points'],
                plot_type=settings['plot_type'], log_scale=log_scale
            )
            st.pyplot(fig)
            register_scale_pair('concentrations_threeway', 'threeway_multi', settings, {
                'design': 'three', 'data_source': 'sample',
                'value_cols': list(selected), 'source_cols': list(selected),
                'factor_cols': [fa_col, fb_col, fc_col],
                'factor_a_col': fa_col, 'factor_b_col': fb_col, 'factor_c_col': fc_col,
                'factor_a_name': fa_name, 'factor_b_name': fb_name, 'factor_c_name': fc_name,
                'group_col': group_col,
                'stats_source_raw': _result_source_name(results, 'individual_ba'),
                'stats_source_log': _result_source_name(results, 'individual_ba', log=True),
                'stats_columns': list(selected),
                'ylabel': f'Concentration ({settings["units"]})', 'ncols': 6,
                'log_scale': log_scale,
            })
            plt.close(fig)


            with st.expander("Interaction Plots"):
                fig_int = viz.plot_threeway_interaction_multi_panel(
                    data=data, value_cols=selected,
                    factor_a_col=fa_col, factor_b_col=fb_col, factor_c_col=fc_col,
                    threeway_results=tw_stats, ncols=6,
                    factor_a_name=fa_name, factor_b_name=fb_name, factor_c_name=fc_name,
                    ylabel=f'Concentration ({settings["units"]})'
                )
                st.pyplot(fig_int)
                register_figure_spec('concentrations_threeway_interaction', 'threeway_interaction', settings, {
                    'design': 'three', 'data_source': 'sample',
                    'value_cols': list(selected), 'source_cols': list(selected),
                    'factor_cols': [fa_col, fb_col, fc_col],
                    'factor_a_col': fa_col, 'factor_b_col': fb_col, 'factor_c_col': fc_col,
                    'factor_a_name': fa_name, 'factor_b_name': fb_name, 'factor_c_name': fc_name,
                    'group_col': group_col,
                    'stats_source_raw': _result_source_name(results, 'individual_ba'),
                    'stats_source_log': _result_source_name(results, 'individual_ba', log=True),
                    'stats_columns': list(selected),
                    'ylabel': f'Concentration ({settings["units"]})', 'ncols': 6,
                    'log_scale': log_scale,
                })
                plt.close(fig_int)

            with st.expander("Three-Way ANOVA Summary"):
                if tw_stats:
                    st.dataframe(get_threeway_differences_summary(tw_stats), hide_index=True)
        elif is_twoway:
            # === TWO-WAY ANOVA PLOTS ===
            fa_col = results.factor_a_col
            fb_col = results.factor_b_col
            fa_name = results.factor_a_name
            fb_name = results.factor_b_name

            st.markdown(f"**Two-way ANOVA**: {fa_name} x {fb_name}")

            tw_stats_raw = {b: results.twoway_individual_ba.get(b) for b in selected
                           if b in results.twoway_individual_ba}
            tw_stats_log = {b: results.log_twoway_individual_ba.get(b) for b in selected
                           if b in results.log_twoway_individual_ba}
            tw_stats = tw_stats_log if log_scale else tw_stats_raw

            fig = viz.plot_twoway_multi_panel(
                data=data, value_cols=selected,
                factor_a_col=fa_col, factor_b_col=fb_col,
                twoway_results=tw_stats, ncols=6,
                factor_a_name=fa_name, factor_b_name=fb_name,
                ylabel=f'Concentration ({settings["units"]})', show_points=settings['show_points'],
                plot_type=settings['plot_type'], log_scale=log_scale
            )
            st.pyplot(fig)
            register_scale_pair('concentrations_twoway', 'twoway_multi', settings, {
                'design': 'two', 'data_source': 'sample',
                'value_cols': list(selected), 'source_cols': list(selected),
                'factor_cols': [fa_col, fb_col],
                'factor_a_col': fa_col, 'factor_b_col': fb_col,
                'factor_a_name': fa_name, 'factor_b_name': fb_name,
                'group_col': group_col,
                'stats_source_raw': _result_source_name(results, 'individual_ba'),
                'stats_source_log': _result_source_name(results, 'individual_ba', log=True),
                'stats_columns': list(selected),
                'ylabel': f'Concentration ({settings["units"]})', 'ncols': 6,
                'log_scale': log_scale,
            })
            plt.close(fig)


            with st.expander("Interaction Plots"):
                fig_int = viz.plot_twoway_interaction_multi_panel(
                    data=data, value_cols=selected,
                    factor_a_col=fa_col, factor_b_col=fb_col,
                    twoway_results=tw_stats, ncols=6,
                    factor_a_name=fa_name, factor_b_name=fb_name,
                    ylabel=f'Concentration ({settings["units"]})'
                )
                st.pyplot(fig_int)
                register_figure_spec('concentrations_interaction', 'twoway_interaction', settings, {
                    'design': 'two', 'data_source': 'sample',
                    'value_cols': list(selected), 'source_cols': list(selected),
                    'factor_cols': [fa_col, fb_col],
                    'factor_a_col': fa_col, 'factor_b_col': fb_col,
                    'factor_a_name': fa_name, 'factor_b_name': fb_name,
                    'group_col': group_col,
                    'stats_source_raw': _result_source_name(results, 'individual_ba'),
                    'stats_source_log': _result_source_name(results, 'individual_ba', log=True),
                    'stats_columns': list(selected),
                    'ylabel': f'Concentration ({settings["units"]})', 'ncols': 6,
                    'log_scale': log_scale,
                })
                plt.close(fig_int)

            with st.expander("Two-Way ANOVA Summary"):
                if tw_stats:
                    st.dataframe(get_twoway_differences_summary(tw_stats), hide_index=True)
        else:
            # === ONE-WAY ANOVA PLOTS ===
            stats_dict_raw = {b: results.individual_ba_results.get(b) for b in selected if b in results.individual_ba_results}
            stats_dict_log = {b: results.log_individual_ba_results.get(b) for b in selected if b in results.log_individual_ba_results}
            stats_dict = stats_dict_log if log_scale else stats_dict_raw

            fig = viz.plot_multi_panel_groups_with_stats(data, selected, group_col, stats_dict,
                                                         ncols=6, plot_type=settings['plot_type'], log_scale=log_scale,
                                                         show_points=settings['show_points'], ylabel=f'Concentration ({settings["units"]})')
            st.pyplot(fig)
            register_scale_pair('concentrations', 'oneway_multi', settings, {
                'design': 'one', 'data_source': 'sample',
                'value_cols': list(selected), 'source_cols': list(selected),
                'factor_cols': [], 'group_col': group_col,
                'stats_source_raw': _result_source_name(results, 'individual_ba'),
                'stats_source_log': _result_source_name(results, 'individual_ba', log=True),
                'stats_columns': list(selected),
                'ylabel': f'Concentration ({settings["units"]})', 'ncols': 6,
                'log_scale': log_scale,
            })
            plt.close(fig)


            with st.expander("Statistical Summary"):
                st.dataframe(get_significant_differences_summary(stats_dict), hide_index=True)


def render_totals_tab(processed, settings):
    """Render totals tab."""
    st.markdown("### Aggregate Totals")

    viz = make_visualizer(settings)
    group_col = processed.structure.group_col
    results = st.session_state.analysis_results

    if not group_col:
        return

    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    data = data[data[group_col].astype(str).str.lower() != 'nan']
    is_threeway = results is not None and getattr(results, 'is_threeway', False)
    is_twoway = results is not None and results.is_twoway

    log_scale = st.checkbox("Log10 scale", key="totals_log")

    key_totals = ['total_all', 'total_primary', 'total_secondary', 'total_conjugated',
                  'total_unconjugated', 'glycine_conjugated', 'taurine_conjugated',
                  'sulfated', 'oxidized', 'epimerized', 'nor_bile_acids',
                  'total_12alpha_hydroxylated', 'total_non12alpha_hydroxylated',
                  'primary_conjugated', 'secondary_conjugated',
                  'primary_unconjugated', 'secondary_unconjugated']

    if is_threeway:
        fa_col, fb_col, fc_col = results.factor_a_col, results.factor_b_col, results.factor_c_col
        fa_name, fb_name, fc_name = results.factor_a_name, results.factor_b_name, results.factor_c_name
        combined = pd.concat([data[[fa_col, fb_col, fc_col]], processed.totals.loc[data.index]], axis=1)
        available = [t for t in key_totals if t in combined.columns]

        tw_stats_raw = {t: results.threeway_totals.get(t) for t in available if t in results.threeway_totals}
        tw_stats_log = {t: results.log_threeway_totals.get(t) for t in available if t in results.log_threeway_totals}
        tw_stats = tw_stats_log if log_scale else tw_stats_raw

        fig = viz.plot_threeway_multi_panel(
            data=combined, value_cols=available,
            factor_a_col=fa_col, factor_b_col=fb_col, factor_c_col=fc_col,
            threeway_results=tw_stats, ncols=6,
            factor_a_name=fa_name, factor_b_name=fb_name, factor_c_name=fc_name,
            ylabel=f'Concentration ({settings["units"]})', show_points=settings['show_points'],
            plot_type=settings['plot_type'], log_scale=log_scale
        )
        st.pyplot(fig)
        register_scale_pair('totals', 'threeway_multi', settings, {
            'design': 'three', 'data_source': 'totals',
            'value_cols': list(available), 'source_cols': list(available),
            'factor_cols': [fa_col, fb_col, fc_col],
            'factor_a_col': fa_col, 'factor_b_col': fb_col, 'factor_c_col': fc_col,
            'factor_a_name': fa_name, 'factor_b_name': fb_name, 'factor_c_name': fc_name,
            'group_col': group_col,
            'stats_source_raw': _result_source_name(results, 'totals'),
            'stats_source_log': _result_source_name(results, 'totals', log=True),
            'stats_columns': list(available),
            'ylabel': f'Concentration ({settings["units"]})', 'ncols': 6,
            'log_scale': log_scale,
        })
        plt.close(fig)


        with st.expander("Three-Way ANOVA Summary"):
            if tw_stats:
                st.dataframe(get_threeway_differences_summary(tw_stats), hide_index=True)
    elif is_twoway:
        fa_col = results.factor_a_col
        fb_col = results.factor_b_col
        combined = pd.concat([data[[fa_col, fb_col]], processed.totals.loc[data.index]], axis=1)
        available = [t for t in key_totals if t in combined.columns]

        tw_stats_raw = {t: results.twoway_totals.get(t) for t in available if t in results.twoway_totals}
        tw_stats_log = {t: results.log_twoway_totals.get(t) for t in available if t in results.log_twoway_totals}
        tw_stats = tw_stats_log if log_scale else tw_stats_raw

        fig = viz.plot_twoway_multi_panel(
            data=combined, value_cols=available,
            factor_a_col=fa_col, factor_b_col=fb_col,
            twoway_results=tw_stats, ncols=6,
            factor_a_name=results.factor_a_name, factor_b_name=results.factor_b_name,
            ylabel=f'Concentration ({settings["units"]})', show_points=settings['show_points'],
            plot_type=settings['plot_type'], log_scale=log_scale
        )
        st.pyplot(fig)
        register_scale_pair('totals', 'twoway_multi', settings, {
            'design': 'two', 'data_source': 'totals',
            'value_cols': list(available), 'source_cols': list(available),
            'factor_cols': [fa_col, fb_col],
            'factor_a_col': fa_col, 'factor_b_col': fb_col,
            'factor_a_name': results.factor_a_name, 'factor_b_name': results.factor_b_name,
            'group_col': group_col,
            'stats_source_raw': _result_source_name(results, 'totals'),
            'stats_source_log': _result_source_name(results, 'totals', log=True),
            'stats_columns': list(available),
            'ylabel': f'Concentration ({settings["units"]})', 'ncols': 6,
            'log_scale': log_scale,
        })
        plt.close(fig)


        with st.expander("Two-Way ANOVA Summary"):
            if tw_stats:
                st.dataframe(get_twoway_differences_summary(tw_stats), hide_index=True)
    else:
        combined = pd.concat([data[[group_col]], processed.totals.loc[data.index]], axis=1)
        available = [t for t in key_totals if t in combined.columns]

        stats_dict_raw = {t: results.totals_results.get(t) for t in available if t in results.totals_results}
        stats_dict_log = {t: results.log_totals_results.get(t) for t in available if t in results.log_totals_results}
        stats_dict = stats_dict_log if log_scale else stats_dict_raw

        fig = viz.plot_multi_panel_groups_with_stats(combined, available, group_col, stats_dict,
                                                     ncols=6, plot_type=settings['plot_type'], log_scale=log_scale,
                                                     show_points=settings['show_points'], ylabel=f'Concentration ({settings["units"]})')
        st.pyplot(fig)
        register_scale_pair('totals', 'oneway_multi', settings, {
            'design': 'one', 'data_source': 'totals',
            'value_cols': list(available), 'source_cols': list(available),
            'factor_cols': [], 'group_col': group_col,
            'stats_source_raw': _result_source_name(results, 'totals'),
            'stats_source_log': _result_source_name(results, 'totals', log=True),
            'stats_columns': list(available),
            'ylabel': f'Concentration ({settings["units"]})', 'ncols': 6,
            'log_scale': log_scale,
        })
        plt.close(fig)


        with st.expander("Statistical Summary"):
            rows = []
            for t in available:
                res = stats_dict.get(t)
                if res:
                    qvalue = res.main_test.qvalue
                    rows.append({
                        'Total': t.replace('_', ' ').title(),
                        'P-value': f"{res.main_test.pvalue:.4f}",
                        'q-value (BH)': (f"{qvalue:.4f}"
                                         if qvalue is not None and np.isfinite(qvalue) else 'N/A'),
                        'Significant': '\u2713' if res.main_test.significant else '',
                        'APA': format_apa_statistics(res)
                    })
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    # =========================================================================
    # Category Composition Pie Charts (on-demand to save memory)
    # =========================================================================
    st.markdown("---")
    st.markdown("#### Category Composition - Pie Charts")
    st.caption("Within-category breakdown: individual bile acids as percentage of each category total")

    if st.button("🥧 Generate Pie Charts", key="gen_cat_pie"):
        ba_names = processed.structure.bile_acid_cols
        if is_threeway:
            cat_pie_data = data.copy()
            cat_group_col = '_factorial_group_'
            cat_pie_data[cat_group_col] = (cat_pie_data[results.factor_a_col].astype(str) + ' - '
                                           + cat_pie_data[results.factor_b_col].astype(str) + ' - '
                                           + cat_pie_data[results.factor_c_col].astype(str))
        elif is_twoway:
            cat_pie_data = data.copy()
            cat_group_col = '_factorial_group_'
            cat_pie_data[cat_group_col] = cat_pie_data[fa_col].astype(str) + ' - ' + cat_pie_data[fb_col].astype(str)
        else:
            cat_pie_data = data
            cat_group_col = group_col

        fig_cat_pie = viz.plot_category_composition_pie_charts(
            data=cat_pie_data,
            group_col=cat_group_col,
            bile_acid_cols=ba_names,
            title='Bile Acid Category Composition by Group',
        )
        st.pyplot(fig_cat_pie)
        register_figure_spec('category_composition_pie', 'category_pie', settings, {
            'factor_cols': ([results.factor_a_col, results.factor_b_col, results.factor_c_col]
                            if is_threeway else
                            [results.factor_a_col, results.factor_b_col] if is_twoway else []),
            'group_col': group_col,
            'bile_acid_cols': list(ba_names),
            'title': 'Bile Acid Category Composition by Group',
        })
        plt.close(fig_cat_pie)


def render_percentages_tab(processed, settings):
    """Render percentages tab - comparing % composition across groups."""
    st.markdown("### Bile Acid Pool Composition (%)")
    st.caption("Compare the percentage each bile acid contributes to the total pool across groups")
    
    viz = make_visualizer(settings)
    group_col = processed.structure.group_col
    results = st.session_state.analysis_results
    
    if not group_col:
        st.warning("No group column detected.")
        return
    
    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    data = data[data[group_col].astype(str).str.lower() != 'nan']
    percentages = processed.percentages.loc[data.index].copy()
    
    # Get available percentage columns (remove _pct suffix for display)
    pct_cols = [c for c in percentages.columns if c.endswith('_pct')]
    ba_names = [c.replace('_pct', '') for c in pct_cols]
    is_threeway = results is not None and getattr(results, 'is_threeway', False)
    is_twoway = results is not None and results.is_twoway

    quick = st.pills("Quick select",
                     ["Top 10 by mean %", "Significant", "Primary", "Secondary",
                      "Conjugated", "Unconjugated", "Glycine", "Taurine",
                      "Sulfated", "Oxidized", "Epimerized",
                      "12α-OH", "Non-12α-OH", "Nor", "Show All", "Custom"],
                     default="Top 10 by mean %", key="pct_quick")

    def _pct_select(getter):
        return [f'{ba}_pct' for ba in getter() if f'{ba}_pct' in pct_cols]

    if quick == "Top 10 by mean %":
        top_pcts = percentages[pct_cols].mean().nlargest(10).index.tolist()
        selected_pct_cols = top_pcts
    elif quick == "Significant":
        if is_threeway:
            selected_pct_cols = [c for c in pct_cols if c in results.threeway_percentages
                               and _threeway_any_effect(results.threeway_percentages[c], settings['alpha'])]
        elif is_twoway:
            selected_pct_cols = [c for c in pct_cols if c in results.twoway_percentages
                               and _twoway_any_effect(results.twoway_percentages[c], settings['alpha'])]
        else:
            selected_pct_cols = [c for c in pct_cols if c in results.percentages_results
                               and results.percentages_results[c].main_test.significant]
        if not selected_pct_cols:
            st.info("No significant percentage differences found. Showing top 10 by mean %.")
            selected_pct_cols = percentages[pct_cols].mean().nlargest(10).index.tolist()
    elif quick == "Primary":
        selected_pct_cols = _pct_select(get_primary)
    elif quick == "Secondary":
        selected_pct_cols = _pct_select(get_secondary)
    elif quick == "Conjugated":
        selected_pct_cols = _pct_select(get_conjugated)
    elif quick == "Unconjugated":
        selected_pct_cols = _pct_select(get_unconjugated)
    elif quick == "Glycine":
        selected_pct_cols = _pct_select(get_glycine_conjugated)
    elif quick == "Taurine":
        selected_pct_cols = _pct_select(get_taurine_conjugated)
    elif quick == "Sulfated":
        selected_pct_cols = _pct_select(get_sulfated)
    elif quick == "Oxidized":
        selected_pct_cols = _pct_select(get_keto_derivatives)
    elif quick == "Epimerized":
        selected_pct_cols = _pct_select(get_iso_forms)
    elif quick == "12α-OH":
        selected_pct_cols = _pct_select(get_12alpha_hydroxylated)
    elif quick == "Non-12α-OH":
        selected_pct_cols = _pct_select(get_non12alpha_hydroxylated)
    elif quick == "Nor":
        selected_pct_cols = _pct_select(get_nor_bile_acids)
    elif quick == "Show All":
        selected_pct_cols = pct_cols
    else:
        selected_bas = st.multiselect("Select bile acids", ba_names, ba_names[:5], key="pct_custom")
        selected_pct_cols = [f'{ba}_pct' for ba in selected_bas]
    
    if not selected_pct_cols:
        st.warning("No bile acids selected.")
        return

    # Show significant findings summary
    if is_threeway:
        sig_pcts = [c for c in selected_pct_cols if c in results.threeway_percentages
                   and _threeway_any_effect(results.threeway_percentages[c], settings['alpha'])]
    elif is_twoway:
        sig_pcts = [c for c in selected_pct_cols if c in results.twoway_percentages
                   and _twoway_any_effect(results.twoway_percentages[c], settings['alpha'])]
    else:
        sig_pcts = [c for c in selected_pct_cols if c in results.percentages_results
                   and results.percentages_results[c].main_test.significant]
    if sig_pcts:
        sig_names = [c.replace('_pct', '') for c in sig_pcts]
        st.success(f"**Significant differences:** {', '.join(sig_names)}")

    # Rename columns for cleaner display (remove _pct suffix)
    display_cols = {col: col.replace('_pct', '') for col in selected_pct_cols}
    display_names = list(display_cols.values())

    # =========================================================================
    # SECTION 1: Statistical comparison (box/violin plots)
    # =========================================================================
    st.markdown("#### Statistical Comparison by Group")

    if is_threeway:
        fa_col, fb_col, fc_col = results.factor_a_col, results.factor_b_col, results.factor_c_col
        fa_name, fb_name, fc_name = results.factor_a_name, results.factor_b_name, results.factor_c_name

        st.markdown(f"**Three-way ANOVA**: {fa_name} x {fb_name} x {fc_name}")

        tw_stats = {c.replace('_pct', ''): results.threeway_percentages.get(c)
                   for c in selected_pct_cols if c in results.threeway_percentages}

        pct_plot_data = pd.concat([data[[fa_col, fb_col, fc_col]].reset_index(drop=True),
                                   percentages[selected_pct_cols].reset_index(drop=True)], axis=1)
        pct_plot_data = pct_plot_data.rename(columns=display_cols)

        try:
            fig = viz.plot_threeway_multi_panel(
                data=pct_plot_data, value_cols=display_names,
                factor_a_col=fa_col, factor_b_col=fb_col, factor_c_col=fc_col,
                threeway_results=tw_stats, ncols=6,
                factor_a_name=fa_name, factor_b_name=fb_name, factor_c_name=fc_name,
                ylabel='% of Total BA', show_points=settings['show_points'],
                plot_type=settings['plot_type']
            )
            st.pyplot(fig)
            register_figure_spec('percentages_threeway', 'threeway_multi', settings, {
                'design': 'three', 'data_source': 'percentages',
                'value_cols': list(display_names), 'source_cols': list(selected_pct_cols),
                'factor_cols': [fa_col, fb_col, fc_col],
                'factor_a_col': fa_col, 'factor_b_col': fb_col, 'factor_c_col': fc_col,
                'factor_a_name': fa_name, 'factor_b_name': fb_name, 'factor_c_name': fc_name,
                'group_col': group_col,
                'stats_source_raw': _result_source_name(results, 'percentages'),
                'stats_columns': list(selected_pct_cols),
                'ylabel': '% of Total BA', 'ncols': 6, 'log_scale': False,
            })
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate three-way percentage plot: {e}")

        with st.expander("Three-Way ANOVA Summary"):
            tw_stats_orig = {c: results.threeway_percentages.get(c)
                           for c in selected_pct_cols if c in results.threeway_percentages}
            if tw_stats_orig:
                st.dataframe(get_threeway_differences_summary(tw_stats_orig), hide_index=True)
    elif is_twoway:
        fa_col = results.factor_a_col
        fb_col = results.factor_b_col
        fa_name = results.factor_a_name
        fb_name = results.factor_b_name

        st.markdown(f"**Two-way ANOVA**: {fa_name} x {fb_name}")

        tw_stats = {c.replace('_pct', ''): results.twoway_percentages.get(c)
                   for c in selected_pct_cols if c in results.twoway_percentages}

        pct_plot_data = pd.concat([data[[fa_col, fb_col]].reset_index(drop=True),
                                   percentages[selected_pct_cols].reset_index(drop=True)], axis=1)
        pct_plot_data = pct_plot_data.rename(columns=display_cols)

        try:
            fig = viz.plot_twoway_multi_panel(
                data=pct_plot_data, value_cols=display_names,
                factor_a_col=fa_col, factor_b_col=fb_col,
                twoway_results=tw_stats, ncols=6,
                factor_a_name=fa_name, factor_b_name=fb_name,
                ylabel='% of Total BA', show_points=settings['show_points'],
                plot_type=settings['plot_type']
            )
            st.pyplot(fig)
            register_figure_spec('percentages_twoway', 'twoway_multi', settings, {
                'design': 'two', 'data_source': 'percentages',
                'value_cols': list(display_names), 'source_cols': list(selected_pct_cols),
                'factor_cols': [fa_col, fb_col],
                'factor_a_col': fa_col, 'factor_b_col': fb_col,
                'factor_a_name': fa_name, 'factor_b_name': fb_name,
                'group_col': group_col,
                'stats_source_raw': _result_source_name(results, 'percentages'),
                'stats_columns': list(selected_pct_cols),
                'ylabel': '% of Total BA', 'ncols': 6, 'log_scale': False,
            })
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate two-way percentage plot: {e}")

        with st.expander("Two-Way ANOVA Summary"):
            tw_stats_orig = {c: results.twoway_percentages.get(c)
                           for c in selected_pct_cols if c in results.twoway_percentages}
            if tw_stats_orig:
                st.dataframe(get_twoway_differences_summary(tw_stats_orig), hide_index=True)
    else:
        # Combine data for plotting
        plot_df = pd.concat([data[[group_col]].reset_index(drop=True),
                             percentages[selected_pct_cols].reset_index(drop=True)], axis=1)
        plot_df_display = plot_df.rename(columns=display_cols)

        # Get stats results with display names
        stats_dict = {}
        for pct_col in selected_pct_cols:
            display_name = pct_col.replace('_pct', '')
            if pct_col in results.percentages_results:
                stats_dict[display_name] = results.percentages_results[pct_col]

        fig = viz.plot_multi_panel_groups_with_stats(
            plot_df_display, display_names, group_col, stats_dict,
            ncols=6, plot_type=settings['plot_type'], log_scale=False,
            show_points=settings['show_points'], ylabel='% of Total BA'
        )

        st.pyplot(fig)
        register_figure_spec('percentages', 'oneway_multi', settings, {
            'design': 'one', 'data_source': 'percentages',
            'value_cols': list(display_names), 'source_cols': list(selected_pct_cols),
            'factor_cols': [], 'group_col': group_col,
            'stats_source_raw': _result_source_name(results, 'percentages'),
            'stats_columns': list(selected_pct_cols),
            'ylabel': '% of Total BA', 'ncols': 6, 'log_scale': False,
        })
        plt.close(fig)

    # =========================================================================
    # SECTION 2: Pie Charts - Pool Composition
    # =========================================================================
    st.markdown("---")
    st.markdown("#### Pool Composition - Pie Charts")
    st.caption("Visual breakdown of bile acid pool for each group (top contributors)")

    if is_threeway:
        # Create combined group column for composition charts
        comp_group_col = '_factorial_group_'
        comp_data = data.copy()
        comp_data[comp_group_col] = (comp_data[results.factor_a_col].astype(str) + ' - '
                                     + comp_data[results.factor_b_col].astype(str) + ' - '
                                     + comp_data[results.factor_c_col].astype(str))
        pie_df = pd.concat([comp_data[[comp_group_col]].reset_index(drop=True),
                           percentages[pct_cols].reset_index(drop=True)], axis=1)
    elif is_twoway:
        # Create combined group column for composition charts
        comp_group_col = '_factorial_group_'
        comp_data = data.copy()
        comp_data[comp_group_col] = comp_data[results.factor_a_col].astype(str) + ' - ' + comp_data[results.factor_b_col].astype(str)
        pie_df = pd.concat([comp_data[[comp_group_col]].reset_index(drop=True),
                           percentages[pct_cols].reset_index(drop=True)], axis=1)
    else:
        comp_group_col = group_col
        pie_df = pd.concat([data[[group_col]].reset_index(drop=True),
                           percentages[pct_cols].reset_index(drop=True)], axis=1)

    fig_pie = viz.plot_composition_pie_charts(
        pie_df, comp_group_col, pct_cols,
        title='Bile Acid Pool Composition by Group',
        top_n=7, other_threshold=2.0
    )
    st.pyplot(fig_pie)
    composition_factors = ([results.factor_a_col, results.factor_b_col, results.factor_c_col]
                           if is_threeway else
                           [results.factor_a_col, results.factor_b_col] if is_twoway else [])
    register_figure_spec('percentages_pie', 'composition_pie', settings, {
        'factor_cols': composition_factors, 'group_col': group_col,
        'pct_cols': list(pct_cols), 'title': 'Bile Acid Pool Composition by Group',
        'top_n': 7, 'other_threshold': 2.0,
    })
    plt.close(fig_pie)

    # =========================================================================
    # SECTION 3: Horizontal Bar Charts - Group Comparison
    # =========================================================================
    st.markdown("---")
    st.markdown("#### Pool Composition - Bar Comparison")
    st.caption("Side-by-side comparison of bile acid percentages across groups")
    st.caption("Mean displayed with SEM error bars")
    
    fig_hbar = viz.plot_composition_horizontal_bars(
        pie_df, comp_group_col, pct_cols,
        title='Bile Acid Pool Composition Comparison',
        top_n=15, show_values=True
    )
    st.pyplot(fig_hbar)
    register_figure_spec('percentages_bars', 'composition_bars', settings, {
        'factor_cols': composition_factors, 'group_col': group_col,
        'pct_cols': list(pct_cols), 'title': 'Bile Acid Pool Composition Comparison',
        'top_n': 15, 'show_values': True,
    })
    plt.close(fig_hbar)
    
    # =========================================================================
    # SECTION 4: Stacked Bar - Full Composition
    # =========================================================================
    st.markdown("---")
    st.markdown("#### Stacked Composition View")
    st.caption("Full bile acid pool breakdown for each group. Mean Percentage with +- SEM displayed.")
    
    fig_stacked = viz.plot_composition_stacked_horizontal(
        pie_df, comp_group_col, pct_cols,
        title='Complete Bile Acid Pool Composition',
        top_n=12
    )
    st.pyplot(fig_stacked)
    register_figure_spec('percentages_stacked', 'composition_stacked', settings, {
        'factor_cols': composition_factors, 'group_col': group_col,
        'pct_cols': list(pct_cols), 'title': 'Complete Bile Acid Pool Composition',
        'top_n': 12,
    })
    plt.close(fig_stacked)
    
    # =========================================================================
    # SECTION 5: Statistical Summary Table
    # =========================================================================
    with st.expander("📊 Statistical Summary"):
        if is_threeway:
            tw_stats_all = {c: results.threeway_percentages.get(c)
                           for c in selected_pct_cols if c in results.threeway_percentages}
            if tw_stats_all:
                st.dataframe(get_threeway_differences_summary(tw_stats_all), hide_index=True)
        elif is_twoway:
            tw_stats_all = {c: results.twoway_percentages.get(c)
                           for c in selected_pct_cols if c in results.twoway_percentages}
            if tw_stats_all:
                st.dataframe(get_twoway_differences_summary(tw_stats_all), hide_index=True)
        else:
            rows = []
            for pct_col in selected_pct_cols:
                ba_name = pct_col.replace('_pct', '')
                res = results.percentages_results.get(pct_col)
                if res:
                    n_sig = 0
                    if (res.main_test.significant and res.posthoc_test
                            and res.posthoc_test.pairwise_results is not None):
                        n_sig = res.posthoc_test.pairwise_results['significant'].sum()
                    qvalue = res.main_test.qvalue
                    rows.append({
                        'Bile Acid': ba_name,
                        'Test': res.main_test.test_type.value,
                        'P-value': f"{res.main_test.pvalue:.4f}",
                        'q-value (BH)': (f"{qvalue:.4f}"
                                         if qvalue is not None and np.isfinite(qvalue) else 'N/A'),
                        'Significant': '✓' if res.main_test.significant else '',
                        'Effect Size': f"{res.main_test.effect_size:.3f}" if res.main_test.effect_size else 'N/A',
                        'Sig. Pairs': n_sig,
                        'APA': format_apa_statistics(res)
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True)

    # Group mean percentages table
    with st.expander("📈 Group Mean Percentages"):
        if is_threeway:
            comp_group_col_summary = '_factorial_group_'
            summary_df_tw = pd.concat([data[[results.factor_a_col, results.factor_b_col, results.factor_c_col]].reset_index(drop=True),
                                       percentages[selected_pct_cols].reset_index(drop=True)], axis=1)
            summary_df_tw[comp_group_col_summary] = (summary_df_tw[results.factor_a_col].astype(str) + ' - '
                                                     + summary_df_tw[results.factor_b_col].astype(str) + ' - '
                                                     + summary_df_tw[results.factor_c_col].astype(str))
            summary_data = []
            for pct_col in selected_pct_cols:
                ba_name = pct_col.replace('_pct', '')
                for group in summary_df_tw[comp_group_col_summary].unique():
                    group_data = summary_df_tw[summary_df_tw[comp_group_col_summary] == group][pct_col]
                    summary_data.append({
                        'Bile Acid': ba_name,
                        'Group': group,
                        'Mean %': f"{group_data.mean():.2f}",
                        'SD': f"{group_data.std():.2f}",
                        'Median %': f"{group_data.median():.2f}"
                    })
            if summary_data:
                pivot_df = pd.DataFrame(summary_data).pivot_table(
                    index='Bile Acid', columns='Group', values='Mean %', aggfunc='first')
                pivot_df = order_pivot_columns(pivot_df, settings, comp_group_col_summary if (is_threeway or is_twoway) else group_col)
                st.dataframe(pivot_df, width="stretch")
        elif is_twoway:
            comp_group_col_summary = '_factorial_group_'
            summary_df_tw = pd.concat([data[[results.factor_a_col, results.factor_b_col]].reset_index(drop=True),
                                       percentages[selected_pct_cols].reset_index(drop=True)], axis=1)
            summary_df_tw[comp_group_col_summary] = summary_df_tw[results.factor_a_col].astype(str) + ' - ' + summary_df_tw[results.factor_b_col].astype(str)
            summary_data = []
            for pct_col in selected_pct_cols:
                ba_name = pct_col.replace('_pct', '')
                for group in summary_df_tw[comp_group_col_summary].unique():
                    group_data = summary_df_tw[summary_df_tw[comp_group_col_summary] == group][pct_col]
                    summary_data.append({
                        'Bile Acid': ba_name,
                        'Group': group,
                        'Mean %': f"{group_data.mean():.2f}",
                        'SD': f"{group_data.std():.2f}",
                        'Median %': f"{group_data.median():.2f}"
                    })
            if summary_data:
                pivot_df = pd.DataFrame(summary_data).pivot_table(
                    index='Bile Acid', columns='Group', values='Mean %', aggfunc='first')
                pivot_df = order_pivot_columns(pivot_df, settings, comp_group_col_summary if (is_threeway or is_twoway) else group_col)
                st.dataframe(pivot_df, width="stretch")
        else:
            summary_data = []
            for pct_col in selected_pct_cols:
                ba_name = pct_col.replace('_pct', '')
                for group in plot_df[group_col].unique():
                    group_data = plot_df[plot_df[group_col] == group][pct_col]
                    summary_data.append({
                        'Bile Acid': ba_name,
                        'Group': group,
                        'Mean %': f"{group_data.mean():.2f}",
                        'SD': f"{group_data.std():.2f}",
                        'Median %': f"{group_data.median():.2f}"
                    })
            if summary_data:
                pivot_df = pd.DataFrame(summary_data).pivot_table(
                    index='Bile Acid', columns='Group', values='Mean %', aggfunc='first')
                pivot_df = order_pivot_columns(pivot_df, settings, comp_group_col_summary if (is_threeway or is_twoway) else group_col)
                st.dataframe(pivot_df, width="stretch")

    # Raw data view
    with st.expander("📋 View percentage data"):
        if is_threeway:
            view_df = pd.concat([data[[results.factor_a_col, results.factor_b_col, results.factor_c_col]].reset_index(drop=True),
                                percentages[selected_pct_cols].reset_index(drop=True)], axis=1)
            st.dataframe(view_df, width="stretch")
        elif is_twoway:
            view_df = pd.concat([data[[results.factor_a_col, results.factor_b_col]].reset_index(drop=True),
                                percentages[selected_pct_cols].reset_index(drop=True)], axis=1)
            st.dataframe(view_df, width="stretch")
        else:
            st.dataframe(plot_df, width="stretch")


def render_ratios_tab(processed, settings):
    """Render ratios tab with multi-panel display."""
    st.markdown("### Ratios")

    viz = make_visualizer(settings)
    group_col = processed.structure.group_col
    results = st.session_state.analysis_results
    is_threeway = results is not None and getattr(results, 'is_threeway', False)
    is_twoway = results is not None and results.is_twoway

    if not group_col:
        st.warning("No group column detected.")
        return

    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    data = data[data[group_col].astype(str).str.lower() != 'nan']

    if is_threeway:
        fa_col, fb_col, fc_col = results.factor_a_col, results.factor_b_col, results.factor_c_col
        combined = pd.concat([data[[fa_col, fb_col, fc_col]], processed.ratios.loc[data.index]], axis=1)
    elif is_twoway:
        fa_col = results.factor_a_col
        fb_col = results.factor_b_col
        combined = pd.concat([data[[fa_col, fb_col]], processed.ratios.loc[data.index]], axis=1)
    else:
        combined = pd.concat([data[[group_col]], processed.ratios.loc[data.index]], axis=1)

    available = [c for c in processed.ratios.columns if not combined[c].isna().all()]

    if not available:
        st.warning("No ratio data available.")
        return

    # Quick select options
    quick = st.segmented_control("Quick select",
                                  ["All ratios", "Significant only", "Key ratios", "Custom"],
                                  default="All ratios", key="ratio_quick")

    # Define key ratios
    key_ratios = ['primary_to_secondary', 'glycine_to_taurine', 'conjugated_to_unconjugated',
                  '12alpha_to_non12alpha', 'primary_conjugated_to_secondary_conjugated',
                  'primary_unconjugated_to_secondary_unconjugated',
                  'CA_to_CDCA', 'TCA_to_GCA', 'GCDCA_to_TCDCA']

    if quick == "All ratios":
        selected_ratios = available
    elif quick == "Significant only":
        if is_threeway:
            selected_ratios = [r for r in available if r in results.threeway_ratios
                             and _threeway_any_effect(results.threeway_ratios[r], settings['alpha'])]
        elif is_twoway:
            selected_ratios = [r for r in available if r in results.twoway_ratios
                             and _twoway_any_effect(results.twoway_ratios[r], settings['alpha'])]
        else:
            selected_ratios = [r for r in available if r in results.ratios_results
                             and results.ratios_results[r].main_test.significant]
        if not selected_ratios:
            st.info("No significant ratio differences found. Showing all ratios.")
            selected_ratios = available
    elif quick == "Key ratios":
        selected_ratios = [r for r in key_ratios if r in available]
        if not selected_ratios:
            selected_ratios = available[:6]
    else:
        selected_ratios = st.multiselect("Select ratios", available,
                                         default=available[:6] if len(available) >= 6 else available,
                                         key="ratio_custom")

    if not selected_ratios:
        st.warning("No ratios selected.")
        return

    # Show significant findings
    if is_threeway:
        sig_ratios = [r for r in selected_ratios if r in results.threeway_ratios
                     and _threeway_any_effect(results.threeway_ratios[r], settings['alpha'])]
    elif is_twoway:
        sig_ratios = [r for r in selected_ratios if r in results.twoway_ratios
                     and _twoway_any_effect(results.twoway_ratios[r], settings['alpha'])]
    else:
        sig_ratios = [r for r in selected_ratios if r in results.ratios_results
                     and results.ratios_results[r].main_test.significant]
    if sig_ratios:
        st.success(f"**Significant differences:** {', '.join(sig_ratios)}")

    log_scale = st.checkbox("Log₁₀ scale", key="ratio_log")

    if is_threeway:
        fa_name, fb_name, fc_name = results.factor_a_name, results.factor_b_name, results.factor_c_name
        st.markdown(f"**Three-way ANOVA**: {fa_name} x {fb_name} x {fc_name}")

        tw_stats_raw = {r: results.threeway_ratios.get(r) for r in selected_ratios
                       if r in results.threeway_ratios}
        tw_stats_log = {r: results.log_threeway_ratios.get(r) for r in selected_ratios
                       if r in results.log_threeway_ratios}
        tw_stats = tw_stats_log if log_scale else tw_stats_raw

        fig = viz.plot_threeway_multi_panel(
            data=combined, value_cols=selected_ratios,
            factor_a_col=fa_col, factor_b_col=fb_col, factor_c_col=fc_col,
            threeway_results=tw_stats, ncols=6,
            factor_a_name=fa_name, factor_b_name=fb_name, factor_c_name=fc_name,
            ylabel='Ratio', show_points=settings['show_points'],
            plot_type=settings['plot_type'], log_scale=log_scale
        )
        st.pyplot(fig)
        register_scale_pair('ratios_threeway', 'threeway_multi', settings, {
            'design': 'three', 'data_source': 'ratios',
            'value_cols': list(selected_ratios), 'source_cols': list(selected_ratios),
            'factor_cols': [fa_col, fb_col, fc_col],
            'factor_a_col': fa_col, 'factor_b_col': fb_col, 'factor_c_col': fc_col,
            'factor_a_name': fa_name, 'factor_b_name': fb_name, 'factor_c_name': fc_name,
            'group_col': group_col,
            'stats_source_raw': _result_source_name(results, 'ratios'),
            'stats_source_log': _result_source_name(results, 'ratios', log=True),
            'stats_columns': list(selected_ratios),
            'ylabel': 'Ratio', 'ncols': 6, 'log_scale': log_scale,
        })
        plt.close(fig)


        with st.expander("Interaction Plots"):
            fig_int = viz.plot_threeway_interaction_multi_panel(
                data=combined, value_cols=selected_ratios,
                factor_a_col=fa_col, factor_b_col=fb_col, factor_c_col=fc_col,
                threeway_results=tw_stats, ncols=6,
                factor_a_name=fa_name, factor_b_name=fb_name, factor_c_name=fc_name,
                ylabel='Ratio'
            )
            st.pyplot(fig_int)
            register_figure_spec('ratios_threeway_interaction', 'threeway_interaction', settings, {
                'design': 'three', 'data_source': 'ratios',
                'value_cols': list(selected_ratios), 'source_cols': list(selected_ratios),
                'factor_cols': [fa_col, fb_col, fc_col],
                'factor_a_col': fa_col, 'factor_b_col': fb_col, 'factor_c_col': fc_col,
                'factor_a_name': fa_name, 'factor_b_name': fb_name, 'factor_c_name': fc_name,
                'group_col': group_col,
                'stats_source_raw': _result_source_name(results, 'ratios'),
                'stats_source_log': _result_source_name(results, 'ratios', log=True),
                'stats_columns': list(selected_ratios),
                'ylabel': 'Ratio', 'ncols': 6, 'log_scale': log_scale,
            })
            plt.close(fig_int)

        with st.expander("Three-Way ANOVA Summary"):
            if tw_stats:
                st.dataframe(get_threeway_differences_summary(tw_stats), hide_index=True)
    elif is_twoway:
        fa_name = results.factor_a_name
        fb_name = results.factor_b_name
        st.markdown(f"**Two-way ANOVA**: {fa_name} x {fb_name}")

        tw_stats_raw = {r: results.twoway_ratios.get(r) for r in selected_ratios
                       if r in results.twoway_ratios}
        tw_stats_log = {r: results.log_twoway_ratios.get(r) for r in selected_ratios
                       if r in results.log_twoway_ratios}
        tw_stats = tw_stats_log if log_scale else tw_stats_raw

        fig = viz.plot_twoway_multi_panel(
            data=combined, value_cols=selected_ratios,
            factor_a_col=fa_col, factor_b_col=fb_col,
            twoway_results=tw_stats, ncols=6,
            factor_a_name=fa_name, factor_b_name=fb_name,
            ylabel='Ratio', show_points=settings['show_points'],
            plot_type=settings['plot_type'], log_scale=log_scale
        )
        st.pyplot(fig)
        register_scale_pair('ratios_twoway', 'twoway_multi', settings, {
            'design': 'two', 'data_source': 'ratios',
            'value_cols': list(selected_ratios), 'source_cols': list(selected_ratios),
            'factor_cols': [fa_col, fb_col],
            'factor_a_col': fa_col, 'factor_b_col': fb_col,
            'factor_a_name': fa_name, 'factor_b_name': fb_name,
            'group_col': group_col,
            'stats_source_raw': _result_source_name(results, 'ratios'),
            'stats_source_log': _result_source_name(results, 'ratios', log=True),
            'stats_columns': list(selected_ratios),
            'ylabel': 'Ratio', 'ncols': 6, 'log_scale': log_scale,
        })
        plt.close(fig)


        with st.expander("Interaction Plots"):
            fig_int = viz.plot_twoway_interaction_multi_panel(
                data=combined, value_cols=selected_ratios,
                factor_a_col=fa_col, factor_b_col=fb_col,
                twoway_results=tw_stats, ncols=6,
                factor_a_name=fa_name, factor_b_name=fb_name,
                ylabel='Ratio'
            )
            st.pyplot(fig_int)
            register_figure_spec('ratios_interaction', 'twoway_interaction', settings, {
                'design': 'two', 'data_source': 'ratios',
                'value_cols': list(selected_ratios), 'source_cols': list(selected_ratios),
                'factor_cols': [fa_col, fb_col],
                'factor_a_col': fa_col, 'factor_b_col': fb_col,
                'factor_a_name': fa_name, 'factor_b_name': fb_name,
                'group_col': group_col,
                'stats_source_raw': _result_source_name(results, 'ratios'),
                'stats_source_log': _result_source_name(results, 'ratios', log=True),
                'stats_columns': list(selected_ratios),
                'ylabel': 'Ratio', 'ncols': 6, 'log_scale': log_scale,
            })
            plt.close(fig_int)

        with st.expander("Two-Way ANOVA Summary"):
            if tw_stats:
                st.dataframe(get_twoway_differences_summary(tw_stats), hide_index=True)
    else:
        # Get stats for selected ratios
        stats_dict_raw = {r: results.ratios_results.get(r) for r in selected_ratios if r in results.ratios_results}
        stats_dict_log = {r: results.log_ratios_results.get(r) for r in selected_ratios if r in results.log_ratios_results}
        stats_dict = stats_dict_log if log_scale else stats_dict_raw

        # Multi-panel figure
        fig = viz.plot_multi_panel_groups_with_stats(
            combined, selected_ratios, group_col, stats_dict,
            ncols=6, plot_type=settings['plot_type'], log_scale=log_scale,
            show_points=settings['show_points'], ylabel='Ratio'
        )
        st.pyplot(fig)
        register_scale_pair('ratios', 'oneway_multi', settings, {
            'design': 'one', 'data_source': 'ratios',
            'value_cols': list(selected_ratios), 'source_cols': list(selected_ratios),
            'factor_cols': [], 'group_col': group_col,
            'stats_source_raw': _result_source_name(results, 'ratios'),
            'stats_source_log': _result_source_name(results, 'ratios', log=True),
            'stats_columns': list(selected_ratios),
            'ylabel': 'Ratio', 'ncols': 6, 'log_scale': log_scale,
        })
        plt.close(fig)

        # Statistical summary
        with st.expander("📊 Statistical Summary"):
            rows = []
            for r in selected_ratios:
                res = stats_dict.get(r)
                if res:
                    n_sig = 0
                    if (res.main_test.significant and res.posthoc_test
                            and res.posthoc_test.pairwise_results is not None):
                        n_sig = res.posthoc_test.pairwise_results['significant'].sum()
                    qvalue = res.main_test.qvalue
                    rows.append({
                        'Ratio': r,
                        'Test': res.main_test.test_type.value,
                        'P-value': f"{res.main_test.pvalue:.4f}",
                        'q-value (BH)': (f"{qvalue:.4f}"
                                         if qvalue is not None and np.isfinite(qvalue) else 'N/A'),
                        'Significant': '✓' if res.main_test.significant else '',
                        'Effect Size': f"{res.main_test.effect_size:.3f}" if res.main_test.effect_size else 'N/A',
                        'Sig. Pairs': n_sig,
                        'APA': format_apa_statistics(res)
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True)

        # Group means table
        with st.expander("📈 Group Mean Ratios"):
            summary_data = []
            for ratio in selected_ratios:
                for group in combined[group_col].unique():
                    group_data = combined[combined[group_col] == group][ratio]
                    summary_data.append({
                        'Ratio': ratio,
                        'Group': group,
                        'Mean': f"{group_data.mean():.3f}",
                        'SD': f"{group_data.std():.3f}",
                        'Median': f"{group_data.median():.3f}"
                    })
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                pivot_df = summary_df.pivot_table(index='Ratio', columns='Group', values='Mean', aggfunc='first')
                pivot_df = order_pivot_columns(pivot_df, settings, group_col)
                st.dataframe(pivot_df, width="stretch")

    # Raw data
    with st.expander("📋 View ratio data"):
        if is_threeway:
            display_cols = [fa_col, fb_col, fc_col] + selected_ratios
        elif is_twoway:
            display_cols = [fa_col, fb_col] + selected_ratios
        else:
            display_cols = [group_col] + selected_ratios
        st.dataframe(combined[display_cols], width="stretch")


def render_hydrophobicity_tab(processed, settings):
    """Render hydrophobicity index tab with group comparisons."""
    st.markdown("### Hydrophobicity Index")

    viz = make_visualizer(settings)
    group_col = processed.structure.group_col
    results = st.session_state.analysis_results
    hydro_df = st.session_state.hydrophobicity
    is_threeway = results is not None and getattr(results, 'is_threeway', False)
    is_twoway = results is not None and results.is_twoway

    if not group_col:
        st.warning("No group column detected.")
        return

    if hydro_df is None or hydro_df.empty:
        st.warning("Hydrophobicity data not available.")
        return

    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    data = data[data[group_col].astype(str).str.lower() != 'nan']

    if is_threeway:
        fa_col, fb_col, fc_col = results.factor_a_col, results.factor_b_col, results.factor_c_col
        combined = pd.concat([data[[fa_col, fb_col, fc_col]], hydro_df.loc[data.index]], axis=1)
    elif is_twoway:
        fa_col = results.factor_a_col
        fb_col = results.factor_b_col
        combined = pd.concat([data[[fa_col, fb_col]], hydro_df.loc[data.index]], axis=1)
    else:
        combined = pd.concat([data[[group_col]], hydro_df.loc[data.index]], axis=1)

    hydro_cols = [c for c in hydro_df.columns if not combined[c].isna().all()]

    if not hydro_cols:
        st.warning("No hydrophobicity data available for selected samples.")
        return

    tissue = settings.get('tissue', 'blood')
    tissue_ph = HydrophobicityCalculator.PH_ENVIRONMENTS[tissue]
    tissue_label = tissue.capitalize()
    st.info(f"**HI_pool**: concentration-weighted hydrophobicity index (Heuman scale) at **pH {tissue_ph}** ({tissue_label}). "
            "**Hydrophobic Load**: total concentration x HI_pool (captures both composition and burden).")

    if is_threeway:
        fa_name, fb_name, fc_name = results.factor_a_name, results.factor_b_name, results.factor_c_name
        st.markdown(f"**Three-way ANOVA**: {fa_name} x {fb_name} x {fc_name}")

        tw_stats = {c: results.threeway_hydrophobicity.get(c) for c in hydro_cols
                   if c in results.threeway_hydrophobicity}

        fig = viz.plot_threeway_multi_panel(
            data=combined, value_cols=hydro_cols,
            factor_a_col=fa_col, factor_b_col=fb_col, factor_c_col=fc_col,
            threeway_results=tw_stats, ncols=2,
            factor_a_name=fa_name, factor_b_name=fb_name, factor_c_name=fc_name,
            ylabel='HI Score', show_points=settings['show_points'],
            plot_type=settings['plot_type'], log_scale=False
        )
        st.pyplot(fig)
        register_figure_spec('hydrophobicity_threeway', 'threeway_multi', settings, {
            'design': 'three', 'data_source': 'hydro',
            'value_cols': list(hydro_cols), 'source_cols': list(hydro_cols),
            'factor_cols': [fa_col, fb_col, fc_col],
            'factor_a_col': fa_col, 'factor_b_col': fb_col, 'factor_c_col': fc_col,
            'factor_a_name': fa_name, 'factor_b_name': fb_name, 'factor_c_name': fc_name,
            'group_col': group_col,
            'stats_source_raw': _result_source_name(results, 'hydrophobicity'),
            'stats_columns': list(hydro_cols),
            'ylabel': 'HI Score', 'ncols': 2, 'log_scale': False,
        })
        plt.close(fig)

        with st.expander("Three-Way ANOVA Summary"):
            if tw_stats:
                st.dataframe(get_threeway_differences_summary(tw_stats), hide_index=True)

    elif is_twoway:
        fa_name = results.factor_a_name
        fb_name = results.factor_b_name
        st.markdown(f"**Two-way ANOVA**: {fa_name} x {fb_name}")

        tw_stats = {c: results.twoway_hydrophobicity.get(c) for c in hydro_cols
                   if c in results.twoway_hydrophobicity}

        fig = viz.plot_twoway_multi_panel(
            data=combined, value_cols=hydro_cols,
            factor_a_col=fa_col, factor_b_col=fb_col,
            twoway_results=tw_stats, ncols=2,
            factor_a_name=fa_name, factor_b_name=fb_name,
            ylabel='HI Score', show_points=settings['show_points'],
            plot_type=settings['plot_type'], log_scale=False
        )
        st.pyplot(fig)
        register_figure_spec('hydrophobicity_twoway', 'twoway_multi', settings, {
            'design': 'two', 'data_source': 'hydro',
            'value_cols': list(hydro_cols), 'source_cols': list(hydro_cols),
            'factor_cols': [fa_col, fb_col],
            'factor_a_col': fa_col, 'factor_b_col': fb_col,
            'factor_a_name': fa_name, 'factor_b_name': fb_name,
            'group_col': group_col,
            'stats_source_raw': _result_source_name(results, 'hydrophobicity'),
            'stats_columns': list(hydro_cols),
            'ylabel': 'HI Score', 'ncols': 2, 'log_scale': False,
        })
        plt.close(fig)

        with st.expander("Two-Way ANOVA Summary"):
            if tw_stats:
                st.dataframe(get_twoway_differences_summary(tw_stats), hide_index=True)

    else:
        stats_dict = {c: results.hydrophobicity_results.get(c) for c in hydro_cols
                     if c in results.hydrophobicity_results}

        fig = viz.plot_multi_panel_groups_with_stats(
            combined, hydro_cols, group_col, stats_dict,
            ncols=2, plot_type=settings['plot_type'], log_scale=False,
            show_points=settings['show_points'], ylabel='HI Score'
        )
        st.pyplot(fig)
        register_figure_spec('hydrophobicity', 'oneway_multi', settings, {
            'design': 'one', 'data_source': 'hydro',
            'value_cols': list(hydro_cols), 'source_cols': list(hydro_cols),
            'factor_cols': [], 'group_col': group_col,
            'stats_source_raw': _result_source_name(results, 'hydrophobicity'),
            'stats_columns': list(hydro_cols),
            'ylabel': 'HI Score', 'ncols': 2, 'log_scale': False,
        })
        plt.close(fig)

        with st.expander("Statistical Summary"):
            rows = []
            for c in hydro_cols:
                res = stats_dict.get(c)
                if res:
                    qvalue = res.main_test.qvalue
                    rows.append({
                        'Metric': c,
                        'Test': res.main_test.test_type.value,
                        'P-value': f"{res.main_test.pvalue:.4f}",
                        'q-value (BH)': (f"{qvalue:.4f}"
                                         if qvalue is not None and np.isfinite(qvalue) else 'N/A'),
                        'Significant': 'Yes' if res.main_test.significant else '',
                        'Effect Size': f"{res.main_test.effect_size:.3f}" if res.main_test.effect_size else 'N/A',
                        'APA': format_apa_statistics(res)
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True)

    with st.expander("Group Means"):
        summary_data = []
        for metric in hydro_cols:
            gcol = group_col
            if is_threeway:
                gcol = fa_col
            elif is_twoway:
                gcol = fa_col
            for group in combined[gcol].unique():
                group_data = combined[combined[gcol] == group][metric]
                summary_data.append({
                    'Metric': metric,
                    'Group': group,
                    'Mean': f"{group_data.mean():.4f}",
                    'SD': f"{group_data.std():.4f}",
                    'Median': f"{group_data.median():.4f}"
                })
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            pivot_df = summary_df.pivot_table(index='Metric', columns='Group', values='Mean', aggfunc='first')
            pivot_df = order_pivot_columns(pivot_df, settings, gcol)
            st.dataframe(pivot_df, width="stretch")

    with st.expander(f"Effective HI Reference Table (pH {tissue_ph} — {tissue_label})"):
        hi_calc = HydrophobicityCalculator(pH=tissue_ph)
        ref = hi_calc.calculate_protonation_table()
        available_bas = [c for c in processed.structure.bile_acid_cols if c in ref]
        ref_rows = []
        for ba in sorted(available_bas, key=lambda x: ref[x]['hi_effective'], reverse=True):
            info = ref[ba]
            ref_rows.append({
                'Species': ba,
                'HI (ionized)': f"{info['hi_ionized']:+.2f}",
                'HI (protonated)': f"{info['hi_protonated']:+.2f}",
                'HI (effective)': f"{info['hi_effective']:+.4f}",
                'pKa': f"{info['pKa']:.2f}",
                '% Protonated': f"{info['f_protonated']*100:.4f}%",
            })
        if ref_rows:
            st.dataframe(pd.DataFrame(ref_rows), hide_index=True, height=400)

    with st.expander("View hydrophobicity data"):
        if is_threeway:
            display_cols = [fa_col, fb_col, fc_col] + hydro_cols
        elif is_twoway:
            display_cols = [fa_col, fb_col] + hydro_cols
        else:
            display_cols = [group_col] + hydro_cols
        st.dataframe(combined[display_cols], width="stretch")


def _count_sig_twoway(twoway_dict, alpha=0.05):
    """Count significant results in a two-way ANOVA results dictionary."""
    count = 0
    for res in twoway_dict.values():
        tw = res.twoway_result
        if any(effect_pvalue(tw, p_attr, q_attr) < alpha for p_attr, q_attr in [
            ('factor_a_pvalue', 'factor_a_qvalue'),
            ('factor_b_pvalue', 'factor_b_qvalue'),
            ('interaction_pvalue', 'interaction_qvalue'),
        ]):
            count += 1
    return count


def _count_sig_threeway(threeway_dict, alpha=0.05):
    """Count significant results in a three-way ANOVA results dictionary."""
    effect_attrs = [
        ('factor_a_pvalue', 'factor_a_qvalue'),
        ('factor_b_pvalue', 'factor_b_qvalue'),
        ('factor_c_pvalue', 'factor_c_qvalue'),
        ('interaction_ab_pvalue', 'interaction_ab_qvalue'),
        ('interaction_ac_pvalue', 'interaction_ac_qvalue'),
        ('interaction_bc_pvalue', 'interaction_bc_qvalue'),
        ('interaction_abc_pvalue', 'interaction_abc_qvalue'),
    ]
    count = 0
    for res in threeway_dict.values():
        tw = res.threeway_result
        if any(effect_pvalue(tw, p_attr, q_attr) < alpha
               for p_attr, q_attr in effect_attrs):
            count += 1
    return count


def _twoway_any_effect(res, alpha):
    tw = res.twoway_result
    return any(effect_pvalue(tw, p_attr, q_attr) < alpha for p_attr, q_attr in [
        ('factor_a_pvalue', 'factor_a_qvalue'),
        ('factor_b_pvalue', 'factor_b_qvalue'),
        ('interaction_pvalue', 'interaction_qvalue'),
    ])


def _threeway_any_effect(res, alpha):
    tw = res.threeway_result
    return any(effect_pvalue(tw, p_attr, q_attr) < alpha for p_attr, q_attr in [
        ('factor_a_pvalue', 'factor_a_qvalue'),
        ('factor_b_pvalue', 'factor_b_qvalue'),
        ('factor_c_pvalue', 'factor_c_qvalue'),
        ('interaction_ab_pvalue', 'interaction_ab_qvalue'),
        ('interaction_ac_pvalue', 'interaction_ac_qvalue'),
        ('interaction_bc_pvalue', 'interaction_bc_qvalue'),
        ('interaction_abc_pvalue', 'interaction_abc_qvalue'),
    ])


def render_statistics_tab(processed, settings):
    """Statistics Summary tab."""
    st.markdown("### Statistics Summary")
    results = st.session_state.analysis_results
    is_threeway = results is not None and getattr(results, 'is_threeway', False)
    is_twoway = results is not None and results.is_twoway
    alpha = settings['alpha']

    if results is not None:
        ba_cols = set(getattr(processed.structure, 'bile_acid_cols', []))
        aggregate_excluded = {
            key: pct for key, pct in getattr(results, 'lod_excluded', {}).items()
            if key not in ba_cols
        }
        if aggregate_excluded:
            detail = ", ".join(f"{key} ({pct}% max group burden)"
                               for key, pct in sorted(aggregate_excluded.items()))
            st.info(
                f"**Aggregate LOD exclusion** (threshold: "
                f"≥{getattr(results, 'aggregate_lod_threshold', 5):g}% LOD-bound burden): "
                f"{detail}. total_all is always retained with a warning when applicable."
            )

    if is_threeway:
        sig_t = _count_sig_threeway(results.threeway_totals, alpha)
        sig_b = _count_sig_threeway(results.threeway_individual_ba, alpha)
        sig_p = _count_sig_threeway(results.threeway_percentages, alpha)
        sig_r = _count_sig_threeway(results.threeway_ratios, alpha)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sig. Totals", f"{sig_t}/{len(results.threeway_totals)}")
        c2.metric("Sig. BAs", f"{sig_b}/{len(results.threeway_individual_ba)}")
        c3.metric("Sig. Percentages", f"{sig_p}/{len(results.threeway_percentages)}")
        c4.metric("Sig. Ratios", f"{sig_r}/{len(results.threeway_ratios)}")

        st.markdown(f"**Design**: {results.factor_a_name} x {results.factor_b_name} x {results.factor_c_name} (Three-Way ANOVA)")

        st.markdown("#### Totals")
        if results.threeway_totals:
            st.dataframe(get_threeway_differences_summary(results.threeway_totals), hide_index=True)

        st.markdown("#### Individual BAs")
        if results.threeway_individual_ba:
            sig_ba_tw = {k: v for k, v in results.threeway_individual_ba.items()
                        if _threeway_any_effect(v, alpha)}
            if sig_ba_tw:
                st.dataframe(get_threeway_differences_summary(sig_ba_tw), hide_index=True)
            else:
                st.info("No significant individual BA differences.")

        st.markdown("#### Percentages")
        if results.threeway_percentages:
            sig_pct_tw = {k.replace('_pct', ''): v for k, v in results.threeway_percentages.items()
                         if _threeway_any_effect(v, alpha)}
            if sig_pct_tw:
                st.dataframe(get_threeway_differences_summary(sig_pct_tw), hide_index=True)
            else:
                st.info("No significant percentage differences.")

        st.markdown("#### Ratios")
        if results.threeway_ratios:
            sig_rat_tw = {k: v for k, v in results.threeway_ratios.items()
                         if _threeway_any_effect(v, alpha)}
            if sig_rat_tw:
                st.dataframe(get_threeway_differences_summary(sig_rat_tw), hide_index=True)
            else:
                st.info("No significant ratio differences.")

        # APA report
        with st.expander("📝 APA-Formatted Report"):
            for category, tw_dict, label in [
                ("Totals", results.threeway_totals, "totals"),
                ("Individual BAs", results.threeway_individual_ba, "individual"),
                ("Ratios", results.threeway_ratios, "ratios"),
            ]:
                if tw_dict:
                    st.markdown(f"**{category}:**")
                    for name, res in tw_dict.items():
                        apa = format_threeway_apa(res)
                        if apa:
                            st.markdown(f"*{name}*: {apa}")
    elif is_twoway:
        sig_t = _count_sig_twoway(results.twoway_totals, alpha)
        sig_b = _count_sig_twoway(results.twoway_individual_ba, alpha)
        sig_p = _count_sig_twoway(results.twoway_percentages, alpha)
        sig_r = _count_sig_twoway(results.twoway_ratios, alpha)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sig. Totals", f"{sig_t}/{len(results.twoway_totals)}")
        c2.metric("Sig. BAs", f"{sig_b}/{len(results.twoway_individual_ba)}")
        c3.metric("Sig. Percentages", f"{sig_p}/{len(results.twoway_percentages)}")
        c4.metric("Sig. Ratios", f"{sig_r}/{len(results.twoway_ratios)}")

        st.markdown(f"**Design**: {results.factor_a_name} x {results.factor_b_name} (Two-Way ANOVA)")

        st.markdown("#### Totals")
        if results.twoway_totals:
            st.dataframe(get_twoway_differences_summary(results.twoway_totals), hide_index=True)

        st.markdown("#### Individual BAs")
        if results.twoway_individual_ba:
            sig_ba_tw = {k: v for k, v in results.twoway_individual_ba.items()
                        if _twoway_any_effect(v, alpha)}
            if sig_ba_tw:
                st.dataframe(get_twoway_differences_summary(sig_ba_tw), hide_index=True)
            else:
                st.info("No significant individual BA differences.")

        st.markdown("#### Percentages")
        if results.twoway_percentages:
            sig_pct_tw = {k.replace('_pct', ''): v for k, v in results.twoway_percentages.items()
                         if _twoway_any_effect(v, alpha)}
            if sig_pct_tw:
                st.dataframe(get_twoway_differences_summary(sig_pct_tw), hide_index=True)
            else:
                st.info("No significant percentage differences.")

        st.markdown("#### Ratios")
        if results.twoway_ratios:
            sig_rat_tw = {k: v for k, v in results.twoway_ratios.items()
                         if _twoway_any_effect(v, alpha)}
            if sig_rat_tw:
                st.dataframe(get_twoway_differences_summary(sig_rat_tw), hide_index=True)
            else:
                st.info("No significant ratio differences.")

        # APA report
        with st.expander("📝 APA-Formatted Report"):
            for category, tw_dict, label in [
                ("Totals", results.twoway_totals, "totals"),
                ("Individual BAs", results.twoway_individual_ba, "individual"),
                ("Ratios", results.twoway_ratios, "ratios"),
            ]:
                if tw_dict:
                    st.markdown(f"**{category}:**")
                    for name, res in tw_dict.items():
                        apa = format_twoway_apa(res)
                        if apa:
                            st.markdown(f"*{name}*: {apa}")
    else:
        sig_t = sum(1 for r in results.totals_results.values() if r.main_test.significant)
        sig_b = sum(1 for r in results.individual_ba_results.values() if r.main_test.significant)
        sig_p = sum(1 for r in results.percentages_results.values() if r.main_test.significant)
        sig_r = sum(1 for r in results.ratios_results.values() if r.main_test.significant)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sig. Totals", f"{sig_t}/{len(results.totals_results)}")
        c2.metric("Sig. BAs", f"{sig_b}/{len(results.individual_ba_results)}")
        c3.metric("Sig. Percentages", f"{sig_p}/{len(results.percentages_results)}")
        c4.metric("Sig. Ratios", f"{sig_r}/{len(results.ratios_results)}")

        st.markdown("#### Totals")
        st.dataframe(get_significant_differences_summary(results.totals_results), hide_index=True)

        st.markdown("#### Individual BAs (Significant)")
        sig_ba = {k: v for k, v in results.individual_ba_results.items() if v.main_test.significant}
        if sig_ba:
            st.dataframe(get_significant_differences_summary(sig_ba), hide_index=True)
        else:
            st.info("No significant individual BA differences.")

        st.markdown("#### Percentages (Significant)")
        sig_pct = {k: v for k, v in results.percentages_results.items() if v.main_test.significant}
        if sig_pct:
            sig_pct_display = {k.replace('_pct', ''): v for k, v in sig_pct.items()}
            st.dataframe(get_significant_differences_summary(sig_pct_display), hide_index=True)
        else:
            st.info("No significant percentage differences.")


@st.fragment
def render_export_tab(processed, settings):
    """Export tab.

    Workbooks and reports are generated by the download button's deferred
    callable, so rendering the tab no longer builds every export on each rerun.
    """
    st.markdown("### Export Results")
    st.caption("Files are generated only when you click a download button.")
    results = st.session_state.analysis_results
    report_gen = st.session_state.report_generator

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Individual Downloads")

        # Get metadata columns (Sample_ID and Group)
        id_cols = []
        if processed.structure.sample_id_col and processed.structure.sample_id_col in processed.sample_data.columns:
            id_cols.append(processed.structure.sample_id_col)
        if processed.structure.group_col and processed.structure.group_col in processed.sample_data.columns:
            id_cols.append(processed.structure.group_col)

        metadata = processed.sample_data[id_cols] if id_cols else pd.DataFrame(index=processed.sample_data.index)

        # Export each dataset with metadata. The closure builds one workbook only
        # when its button is clicked, so no workbook bytes are retained here.
        for name, df in [("Concentrations", processed.concentrations),
                         ("Totals", processed.totals),
                         ("Ratios", processed.ratios),
                         ("Percentages", processed.percentages)]:
            def _build_workbook(df=df, metadata=metadata):
                export_df = pd.concat([metadata, df], axis=1)
                return df_to_excel_bytes(export_df)

            st.download_button(
                f"📥 {name} (Excel)",
                _build_workbook,
                f"bile_acid_{name.lower()}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{name.lower()}"
            )

        # LOD-highlighted Excel export
        st.markdown("---")
        lod_label = {"half_lod": "LOD/2", "lod": "LOD value", "zero": "Zero", "drop": "NaN"}
        st.download_button(
            "📥 All Data (Excel) - LOD highlighted",
            lambda: create_full_data_excel_with_highlighting(processed, settings['lod_handling']),
            f"bile_acid_data_lod_highlighted_{datetime.now():%Y%m%d}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_lod_excel"
        )
        st.caption(f"Yellow cells = values replaced (below LOD → {lod_label[settings['lod_handling']]})")

    with col2:
        st.markdown("#### Complete Reports")
        if report_gen:
            def _build_report():
                buf = BytesIO()
                try:
                    report_gen.display_orders = report_display_orders(settings)
                    report_gen.save_excel_report(buf)
                    return buf.getvalue()
                finally:
                    report_gen.analysis_sheets.clear()

            st.download_button("📥 Statistical Report (Excel)", _build_report,
                               f"statistical_report_{datetime.now():%Y%m%d}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="download_excel_report")

        # Generate the ZIP on click and write it to disk; figures are consumed
        # one at a time by iter_all_export_figures().
        if st.button("📦 Generate Complete Package (ZIP)", key="generate_zip"):
            clear_zip_package()
            try:
                with st.status("Generating export package...", expanded=True) as status:
                    st.write("Rendering figures one at a time...")
                    figure_errors = []
                    zip_path, figure_count, errors = create_results_zip(
                        processed,
                        results,
                        iter_export_figures(processed, results, settings, errors=figure_errors),
                        report_gen,
                        settings,
                    )
                    errors.extend(figure_errors)
                    st.session_state.zip_path = zip_path
                    st.session_state.zip_figure_count = figure_count
                    if errors:
                        status.update(label="Package ready with warnings", state="complete", expanded=True)
                        for err in errors:
                            st.warning(err)
                    else:
                        status.update(label="Package ready!", state="complete", expanded=False)
            except Exception as exc:
                st.error(f"Could not generate the export package: {exc}")

        zip_path = st.session_state.get('zip_path')
        if zip_path and Path(zip_path).exists():
            def _read_zip():
                try:
                    return Path(zip_path).read_bytes()
                except OSError:
                    return b''

            st.download_button("📥 Download Complete Package (ZIP)", _read_zip,
                               f"bile_acid_analysis_{datetime.now():%Y%m%d_%H%M}.zip",
                               "application/zip",
                               key="download_zip")
            st.caption(f"📊 Package includes {st.session_state.zip_figure_count} figures, "
                       "including any figures viewed in the tabs with their saved style/settings.")
            if st.button("Clear generated package", key="clear_zip"):
                clear_zip_package()
                st.rerun()

    st.markdown("---")



def render_heatmap_tab(processed, settings):
    """Render heatmap tab - faceted heatmap of bile acid expression by category."""
    st.markdown("### Faceted Heatmap")
    st.caption("Visualize bile acid expression patterns across samples, grouped by category")

    viz = make_visualizer(settings)
    group_col = processed.structure.group_col

    if not group_col:
        st.warning("No group column detected.")
        return

    available_bas = set(processed.structure.bile_acid_cols)

    # Build available categories (only those with detected BAs and a totals column)
    available_categories = {}
    for cat_key, ba_list in ANALYSIS_GROUPS.items():
        present_bas = [ba for ba in ba_list if ba in available_bas]
        if present_bas and cat_key in processed.totals.columns:
            display = ANALYSIS_GROUP_DISPLAY_NAMES.get(cat_key, cat_key)
            available_categories[display] = cat_key

    if not available_categories:
        st.warning("No categories available.")
        return

    # Category selection
    sorted_display_names = sorted(available_categories.keys())
    default_selection = [n for n in ["Primary", "Secondary", "Oxidized (Keto)", "Epimerized (Iso)"]
                        if n in sorted_display_names]
    if not default_selection:
        default_selection = sorted_display_names[:3]

    selected_display = st.multiselect(
        "Select categories",
        sorted_display_names,
        default=default_selection,
        key="heatmap_categories"
    )

    if not selected_display:
        st.info("Select one or more categories to generate a heatmap.")
        return

    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        norm_mode = st.selectbox("Normalization", ["Z-score", "Row %", "None"],
                                 index=0, key="heatmap_norm")
        norm_map = {"Z-score": "zscore", "Row %": "row", "None": "none"}
        normalization = norm_map[norm_mode]
    with col2:
        cmap = st.selectbox("Color map", ["RdBu_r", "coolwarm", "viridis", "YlOrRd", "PiYG"],
                            index=0, key="heatmap_cmap")
    with col3:
        st.write("")  # spacer

    # Build selected_categories dict: {cat_key: [BA1, BA2, ...]}
    selected_cats = {}
    for display in selected_display:
        cat_key = available_categories[display]
        present_bas = [ba for ba in ANALYSIS_GROUPS[cat_key] if ba in available_bas]
        selected_cats[cat_key] = present_bas

    # Generate button
    if st.button("Generate Heatmap", type="primary", key="heatmap_generate"):
        with st.spinner("Generating heatmap..."):
            data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
            data = data[data[group_col].astype(str).str.lower() != 'nan']
            if len(data) > 150:
                st.caption(f"Large cohort ({len(data)} samples): the heatmap is capped at "
                           "24 × 24 in to keep memory bounded.")

            # Build factor_cols: use detected factors, fallback to group_col
            if processed.structure.factors:
                fc = processed.structure.factors
            else:
                fc = {group_col: group_col}

            fig = viz.plot_faceted_heatmap(
                sample_data=data,
                totals=processed.totals.loc[data.index],
                group_col=group_col,
                selected_categories=selected_cats,
                category_display_names=ANALYSIS_GROUP_DISPLAY_NAMES,
                normalization=normalization,
                cmap=cmap,
                sample_id_col=processed.structure.sample_id_col,
                factor_cols=fc,
            )
            st.session_state['heatmap_fig'] = fig_to_bytes(fig, dpi=100)
            register_figure_spec('faceted_heatmap', 'heatmap', settings, {
                'group_col': group_col,
                'selected_categories': selected_cats,
                'normalization': normalization,
                'cmap': cmap,
                'sample_id_col': processed.structure.sample_id_col,
                'factor_cols': dict(fc),
            })
            plt.close(fig)

    # Display cached heatmap
    if 'heatmap_fig' in st.session_state and st.session_state['heatmap_fig'] is not None:
        st.image(st.session_state['heatmap_fig'], use_container_width=True)


def main():
    """Main entry point."""
    init_session_state()
    settings = render_sidebar()
    
    st.markdown("# 🧬 Bile Acid Analysis Pipeline")
    uploaded = st.file_uploader("Upload Excel/ODS file", ['xlsx', 'xls', 'ods'])
    
    if uploaded:
        # Detect replacements even when the new file has the same name.
        uploaded_id = getattr(uploaded, 'file_id', None) or f"{uploaded.name}:{uploaded.size}"
        file_changed = st.session_state.get('uploaded_file_id') != uploaded_id
        if file_changed:
            reset_analysis_state(clear_upload=True)
            st.session_state.uploaded_file_id = uploaded_id
            st.session_state.last_settings = get_data_affecting_settings(settings)

        # Keep one stable temp file per upload instead of rewriting it every rerun.
        tmp_path = st.session_state.get('uploaded_tmp_path')
        if not tmp_path or not Path(tmp_path).exists():
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            st.session_state.uploaded_tmp_path = tmp_path

        # Sheet selection dropdown
        processor = BileAcidDataProcessor(
            lod_handling=settings['lod_handling'],
            dilution_factor=settings['dilution_factor'],
        )
        try:
            selectable_sheets, auto_detected_sheet = processor.get_selectable_sheets(tmp_path)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        if len(selectable_sheets) > 1:
            # Determine default: use previous selection if still valid, otherwise auto-detected
            if st.session_state.selected_sheet in selectable_sheets:
                default_idx = selectable_sheets.index(st.session_state.selected_sheet)
            else:
                default_idx = selectable_sheets.index(auto_detected_sheet)

            selected_sheet = st.selectbox(
                "Select data sheet",
                selectable_sheets,
                index=default_idx,
                help="Choose which sheet contains your sample data. The LC-MS data sheet (standard curves) is excluded."
            )
        else:
            selected_sheet = selectable_sheets[0] if selectable_sheets else None

        # Detect sheet change
        sheet_changed = (st.session_state.selected_sheet != selected_sheet)
        if sheet_changed:
            reset_analysis_state(clear_upload=False)
            st.session_state.selected_sheet = selected_sheet
            st.session_state.last_settings = get_data_affecting_settings(settings)

        # Check if settings changed (LOD handling, alpha, etc.)
        settings_changed = check_settings_changed(settings)
        if settings_changed:
            st.toast("Settings changed — reprocessing data...")

        # Reprocess data if file, sheet, or settings changed
        if file_changed or sheet_changed or settings_changed or st.session_state.processed_data is None:
            with st.status("Processing data...", expanded=True) as status:
                st.write(f"Reading sheet: **{selected_sheet}**...")

                try:
                    st.write("Detecting structure & applying LOD handling...")
                    processed = processor.load_and_process(tmp_path, sheet_name=selected_sheet)
                    st.session_state.processed_data = processed

                    tissue_ph = HydrophobicityCalculator.PH_ENVIRONMENTS[settings['tissue']]
                    hi_calc = HydrophobicityCalculator(pH=tissue_ph)
                    hi_pool = hi_calc.calculate_pool_hi(processed.concentrations, processed.structure.bile_acid_cols)
                    hi_load = hi_calc.calculate_hydrophobic_load(processed.concentrations, processed.structure.bile_acid_cols)
                    st.session_state.hydrophobicity = pd.DataFrame({"HI_pool": hi_pool, "hydrophobic_load": hi_load})
                    st.session_state.hydro_tissue = settings['tissue']

                    status.update(label="Data processed!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Processing failed", state="error")
                    st.error(f"Error: {e}")
                    return
    
    if st.session_state.processed_data:
        processed = st.session_state.processed_data

        if st.session_state.hydro_tissue != settings['tissue'] and st.session_state.hydrophobicity is not None:
            tissue_ph = HydrophobicityCalculator.PH_ENVIRONMENTS[settings['tissue']]
            hi_calc = HydrophobicityCalculator(pH=tissue_ph)
            hi_pool = hi_calc.calculate_pool_hi(processed.concentrations, processed.structure.bile_acid_cols)
            hi_load = hi_calc.calculate_hydrophobic_load(processed.concentrations, processed.structure.bile_acid_cols)
            hydro_df = pd.DataFrame({"HI_pool": hi_pool, "hydrophobic_load": hi_load})
            st.session_state.hydrophobicity = hydro_df
            st.session_state.hydro_tissue = settings['tissue']
            st.session_state.stats_sections_done.discard('hydrophobicity')
            st.session_state.stats_computed = False
            # Keep the cached generator and every other section's results.  Only
            # the hydrophobicity inputs/results changed, so the next Hydro tab
            # visit reruns that section in place instead of replacing the whole
            # result object (which would silently drop the other sections).
            report_gen = st.session_state.report_generator
            if report_gen is not None:
                report_gen.hydrophobicity = hydro_df
                report_gen.hydro_ph = tissue_ph
                report_gen.hydro_tissue = settings['tissue']
                report_gen.results.hydrophobicity_results.clear()
                report_gen.results.twoway_hydrophobicity.clear()
                report_gen.results.threeway_hydrophobicity.clear()
                report_gen.analysis_sheets.clear()
                st.session_state.analysis_results = report_gen.results
            # Only hydrophobicity recipes depend on the tissue pH.
            clear_figure_specs(lambda name: name.startswith('hydrophobicity'))

        quality = validate_data_quality(processed)
        c1, c2, c3 = st.columns(3)
        c1.metric("Samples", quality['n_samples'])
        c2.metric("Bile Acids", quality['n_bile_acids_detected'])
        c3.metric("Groups", quality.get('n_groups', 'N/A'))
        
        # Show LOD info
        lod_source_label = "Auto-detected from standards" if quality.get('lod_source') == 'standards' else "Default (no standards found)"
        lod_display = {"half_lod": "LOD/2", "lod": "LOD value", "zero": "Zero", "drop": "NaN"}
        st.caption(f"📊 LOD: **{lod_source_label}** (below-LOD → {lod_display[settings['lod_handling']]}) | α = **{settings['alpha']}**")

        # Show factor detection info
        if processed.structure.n_factors >= 3:
            factor_names = list(processed.structure.factors.keys())
            source_label = {"metadata_sheet": "metadata sheet", "prefix_columns": "Factor_ columns"}.get(
                processed.structure.factor_source, "unknown")
            st.info(f"🔬 **Three-way design detected** ({source_label}): **{factor_names[0]}** × **{factor_names[1]}** × **{factor_names[2]}**")
        elif processed.structure.n_factors >= 2:
            factor_names = list(processed.structure.factors.keys())
            source_label = {"metadata_sheet": "metadata sheet", "prefix_columns": "Factor_ columns"}.get(
                processed.structure.factor_source, "unknown")
            st.info(f"🔬 **Two-way design detected** ({source_label}): **{factor_names[0]}** × **{factor_names[1]}**")
        
        # User-defined group order for figures (display only)
        render_group_order_controls(processed)
        settings['level_orders'] = build_level_orders(processed, st.session_state.level_orders)

        # Show per-analyte LOD details
        if quality.get('analyte_lods'):
            with st.expander("🔬 Per-Analyte LOD Details"):
                lod_rows = []
                for ba in processed.structure.bile_acid_cols:
                    lod_val = quality['analyte_lods'].get(ba, 'N/A')
                    count = quality.get('analyte_lod_counts', {}).get(ba, 0)
                    n_samples = quality['n_samples']
                    pct = f"{count/n_samples*100:.0f}%" if n_samples > 0 else "0%"
                    lod_rows.append({
                        'Analyte': ba,
                        'LOD (nM)': lod_val if isinstance(lod_val, str) else f"{lod_val:.1f}",
                        'Below LOD': count,
                        '% Below': pct
                    })
                if lod_rows:
                    lod_df = pd.DataFrame(lod_rows)
                    # Highlight high replacement rates
                    st.dataframe(lod_df, width="stretch", hide_index=True)
        
        # One unified selector for one-, two-, and three-way designs. Only the
        # active section is rendered/computed; st.tabs rendered every section on
        # every rerun and was the main cause of Cloud OOM failures.
        tab_options = ["📈 Concentrations", "📊 Totals", "📉 Percentages", "🔢 Ratios",
                       "🧪 Hydrophobicity", "🗺️ Heatmap", "📊 Statistics", "💾 Export"]
        active_tab = st.segmented_control("Section", tab_options, default="📈 Concentrations")
        if not active_tab:
            active_tab = tab_options[0]

        # Map each view to the statistics sections it needs.
        tab_sections = {
            "📈 Concentrations": ['individual_ba'],
            "📊 Totals": ['totals'],
            "📉 Percentages": ['percentages'],
            "🔢 Ratios": ['ratios'],
            "🧪 Hydrophobicity": ['hydrophobicity'],
            "🗺️ Heatmap": [],
            "📊 Statistics": ['individual_ba', 'totals', 'ratios', 'percentages', 'categories', 'hydrophobicity'],
            "💾 Export": ['individual_ba', 'totals', 'ratios', 'percentages', 'categories', 'hydrophobicity'],
        }
        needed = tab_sections.get(active_tab, [])
        missing = [s for s in needed if s not in st.session_state.stats_sections_done]
        if missing:
            # Drop any stale figures before computing the next section.
            plt.close('all')
            gc.collect()
            label = f"Computing {', '.join(missing)} statistics..."
            with st.status(label, expanded=False) as status:
                for sec in missing:
                    ensure_section_computed(processed, settings, sec)
                status.update(label="Statistics ready", state="complete")

        if active_tab == "📈 Concentrations":
            render_concentrations_tab(processed, settings)
        elif active_tab == "📊 Totals":
            render_totals_tab(processed, settings)
        elif active_tab == "📉 Percentages":
            render_percentages_tab(processed, settings)
        elif active_tab == "🔢 Ratios":
            render_ratios_tab(processed, settings)
        elif active_tab == "🧪 Hydrophobicity":
            render_hydrophobicity_tab(processed, settings)
        elif active_tab == "🗺️ Heatmap":
            render_heatmap_tab(processed, settings)
        elif active_tab == "📊 Statistics":
            render_statistics_tab(processed, settings)
        elif active_tab == "💾 Export":
            render_export_tab(processed, settings)
    
    else:
        # Removing the uploader releases the analysis and its temp file.  This
        # keeps abandoned sessions from retaining processed data indefinitely.
        if st.session_state.get('processed_data') is not None or st.session_state.get('uploaded_tmp_path'):
            reset_analysis_state(clear_upload=True)

        # Show expected input format when no data is loaded
        st.markdown("---")
        st.markdown("## 📋 Expected Input Format")
        
        st.markdown("### LC-MS sheet")
        st.markdown("""
        - **Rows:** First few are Std curves, followed by samples
        - **Columns:** "Data Filename" (Std curve and sample names), followed by bile acid species
        """)


        st.markdown("**Example:**")
        
        # Create example dataframe
        example_lcms_df = pd.DataFrame({
            'Data Filename': ['Std 1 ngmL', 'Std 3  ngmL', 'Std 10 ngmL'],
            'CA': [1.3, 3.0, 9.9],
            'TCA': [1.1, 2.9, 10.1],
            'GCA': ['----', 3.1, 10.0],
            'TCDCA': ['----', '----', 10.2],
            '...': ['...', '...', '...']
        })
        st.dataframe(example_lcms_df.astype(str), hide_index=True, width="content")

        st.markdown("### Sample sheet")
        st.markdown("""
        - **Rows:** Samples
        - **Columns:** Bile acid species (matching panel names)
        - **First column(s):** Sample ID, Group/Type
        - **Values:** Concentrations (typically ng/mL)
        - **Below LOD:** Can be "-----", "LOD", "BLQ", "ND", etc.
        """)
        
        st.markdown("**Example:**")
        
        # Create example dataframe
        example_df = pd.DataFrame({
            'Type': ['Aged-1', 'Aged-2', 'Young-1'],
            'Sample_ID': ['S001', 'S002', 'S003'],
            'CA': [125.4, 142.8, 98.6],
            'TCA': [312.5, 287.9, '-----'],
            'GCA': [1520.3, 1380.7, 1245.2],
            'TCDCA': [15.2, 18.4, 12.8],
            '...': ['...', '...', '...']
        })
        st.dataframe(example_df.astype(str), hide_index=True, width="content")
        
        st.markdown("""**For Multiple Independent Variables need to add "Factor_" in front of it""")
        st.markdown("**Example:**")
        # Create example dataframe
        example_df = pd.DataFrame({
            'Factor_Age': ['Aged', 'Aged', 'Young'],
            'Factor_Sex': ['Male', 'Female', 'Female'],
            'Sample_ID': ['S001', 'S002', 'S003'],
            'CA': [125.4, 142.8, 98.6],
            'TCA': [312.5, 287.9, '-----'],
            'GCA': [1520.3, 1380.7, 1245.2],
            'TCDCA': [15.2, 18.4, 12.8],
            '...': ['...', '...', '...']
        })
        st.dataframe(example_df.astype(str), hide_index=True, width='content')

        st.info("💡 The pipeline auto-detects bile acid columns and group assignments from your data.")


if __name__ == "__main__":
    main()
