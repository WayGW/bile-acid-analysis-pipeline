"""
Bile Acid Analysis Pipeline - Streamlit Application
====================================================

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pathlib import Path
import sys
import tempfile
import zipfile
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config.bile_acid_species import (
    BILE_ACID_PANEL, get_glycine_conjugated, get_taurine_conjugated,
    get_primary, get_secondary, get_conjugated, get_unconjugated
)
from modules.data_processing import BileAcidDataProcessor, ProcessedData, validate_data_quality
from modules.statistical_tests import StatisticalAnalyzer, format_analysis_report
from modules.visualization import BileAcidVisualizer, create_summary_figure
from modules.report_generation import (
    ExcelReportGenerator, SignificancePlotter, 
    ComprehensiveAnalysisResults, format_apa_statistics,
    get_significant_differences_summary
)

st.set_page_config(page_title="Bile Acid Analysis Pipeline", page_icon="🧬", layout="wide")


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'processed_data': None,
        'analysis_results': ComprehensiveAnalysisResults(),
        'figures': {},
        'report_generator': None,
        'stats_computed': False,
        'last_file': None,
        'last_settings': None  # Track settings that affect data/stats
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_data_affecting_settings(settings):
    """Extract settings that affect data processing and statistics."""
    return {
        'lod_handling': settings['lod_handling'],
        'alpha': settings['alpha']
    }


def check_settings_changed(settings):
    """Check if data-affecting settings have changed, reset caches if so."""
    current = get_data_affecting_settings(settings)
    
    if st.session_state.last_settings != current:
        # Settings changed - need to reprocess data and recompute stats
        st.session_state.stats_computed = False
        st.session_state.figures = {}
        st.session_state.last_settings = current
        return True
    return False


def fig_to_bytes(fig, format='png', dpi=300):
    """Convert matplotlib figure to bytes."""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def store_figure(fig, name):
    """Store a figure in session state."""
    st.session_state.figures[name] = fig


def compute_all_statistics(processed, settings):
    """Compute all statistics once and cache."""
    if st.session_state.stats_computed:
        return st.session_state.analysis_results
    
    group_col = processed.structure.group_col
    if not group_col:
        return None
    
    report_gen = ExcelReportGenerator(
        data=processed.sample_data,
        group_col=group_col,
        bile_acid_cols=processed.structure.bile_acid_cols,
        totals=processed.totals,
        ratios=processed.ratios,
        percentages=processed.percentages,
        alpha=settings['alpha']
    )
    
    results = report_gen.run_all_statistics()
    st.session_state.analysis_results = results
    st.session_state.report_generator = report_gen
    st.session_state.stats_computed = True
    return results


def get_sig_pairs(result, max_pairs=5):
    """Extract significant pairs from result."""
    if not result or not result.posthoc_test or result.posthoc_test.pairwise_results is None:
        return []
    pairs = []
    df = result.posthoc_test.pairwise_results
    sig = df[df['significant']].sort_values('pvalue_adj' if 'pvalue_adj' in df else 'pvalue')
    for _, row in sig.head(max_pairs).iterrows():
        p = row.get('pvalue_adj', row.get('pvalue', 1.0))
        annot = '***' if p < 0.001 else ('**' if p < 0.01 else '*' if p < 0.05 else '')
        if annot:
            pairs.append((row['group1'], row['group2'], annot))
    return pairs


def generate_all_export_figures(processed, results, settings):
    """Generate all figure variations for export package."""
    from config.bile_acid_species import get_primary, get_secondary, get_glycine_conjugated, get_taurine_conjugated
    
    figures = {}
    viz = BileAcidVisualizer(color_palette=settings['color_palette'], style=settings['plot_style'])
    group_col = processed.structure.group_col
    
    if not group_col or not results:
        return figures
    
    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    data = data[data[group_col].astype(str).str.lower() != 'nan']
    available_bas = processed.structure.bile_acid_cols
    
    # === CONCENTRATIONS TAB FIGURES ===
    conc_selections = {
        'top10': processed.concentrations.mean().nlargest(10).index.tolist(),
        'significant': [b for b in available_bas if b in results.individual_ba_results 
                       and results.individual_ba_results[b].main_test.significant][:10],
        'primary': [b for b in get_primary() if b in available_bas][:10],
        'secondary': [b for b in get_secondary() if b in available_bas][:10],
    }
    
    for sel_name, selected in conc_selections.items():
        if not selected:
            continue
        stats_dict = {b: results.individual_ba_results.get(b) for b in selected if b in results.individual_ba_results}
        
        for log_scale in [False, True]:
            suffix = f"_{sel_name}{'_log' if log_scale else ''}"
            try:
                fig = viz.plot_multi_panel_groups_with_stats(
                    data, selected, group_col, stats_dict, ncols=3,
                    plot_type=settings['plot_type'], log_scale=log_scale,
                    show_points=settings['show_points'])
                figures[f'concentrations{suffix}'] = fig
            except Exception:
                pass
    
    # === TOTALS TAB FIGURES ===
    totals_combined = pd.concat([data[[group_col]], processed.totals.loc[data.index]], axis=1)
    
    # Key totals
    key_totals = ['total_all', 'total_primary', 'total_secondary', 'total_conjugated', 
                  'total_unconjugated', 'glycine_conjugated', 'taurine_conjugated']
    available_totals = [t for t in key_totals if t in totals_combined.columns]
    
    if available_totals:
        totals_stats = {t: results.totals_results.get(t) for t in available_totals if t in results.totals_results}
        for log_scale in [False, True]:
            suffix = '_log' if log_scale else ''
            try:
                fig = viz.plot_multi_panel_groups_with_stats(
                    totals_combined, available_totals, group_col, totals_stats, ncols=3,
                    plot_type=settings['plot_type'], log_scale=log_scale,
                    show_points=settings['show_points'])
                figures[f'totals{suffix}'] = fig
            except Exception:
                pass
    
    # === PERCENTAGES TAB FIGURES ===
    pct_cols = [col for col in processed.percentages.columns if col.endswith('_pct')]
    if pct_cols:
        # Top 10 by mean percentage
        top_pct = processed.percentages[pct_cols].mean().nlargest(10).index.tolist()
        pct_combined = pd.concat([data[[group_col]], processed.percentages.loc[data.index, top_pct]], axis=1)
        
        # Rename columns for display (remove _pct suffix)
        display_cols = [col.replace('_pct', '') for col in top_pct]
        pct_display = pct_combined.rename(columns={old: new for old, new in zip(top_pct, display_cols)})
        
        pct_stats = {}
        for pct_col, disp_col in zip(top_pct, display_cols):
            if pct_col in results.percentages_results:
                pct_stats[disp_col] = results.percentages_results[pct_col]
        
        try:
            fig = viz.plot_multi_panel_groups_with_stats(
                pct_display, display_cols, group_col, pct_stats, ncols=3,
                plot_type=settings['plot_type'], log_scale=False,
                show_points=settings['show_points'])
            figures['percentages_top10'] = fig
        except Exception:
            pass
    
    # === RATIOS TAB FIGURES ===
    ratio_cols = [col for col in processed.ratios.columns if not processed.ratios[col].isna().all()]
    if ratio_cols:
        ratios_combined = pd.concat([data[[group_col]], processed.ratios.loc[data.index, ratio_cols]], axis=1)
        ratios_stats = {r: results.ratios_results.get(r) for r in ratio_cols if r in results.ratios_results}
        
        for log_scale in [False, True]:
            suffix = '_log' if log_scale else ''
            try:
                fig = viz.plot_multi_panel_groups_with_stats(
                    ratios_combined, ratio_cols[:9], group_col, ratios_stats, ncols=3,
                    plot_type=settings['plot_type'], log_scale=log_scale,
                    show_points=settings['show_points'])
                figures[f'ratios{suffix}'] = fig
            except Exception:
                pass
    
    return figures


def create_results_zip(processed, results, figures, report_gen, settings=None):
    """Create ZIP with all results."""
    # Get metadata columns
    id_cols = []
    if processed.structure.sample_id_col and processed.structure.sample_id_col in processed.sample_data.columns:
        id_cols.append(processed.structure.sample_id_col)
    if processed.structure.group_col and processed.structure.group_col in processed.sample_data.columns:
        id_cols.append(processed.structure.group_col)
    
    metadata = processed.sample_data[id_cols] if id_cols else pd.DataFrame(index=processed.sample_data.index)
    
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Data files with metadata
        for name, df in [('concentrations', processed.concentrations), 
                         ('percentages', processed.percentages),
                         ('totals', processed.totals), 
                         ('ratios', processed.ratios)]:
            csv_buf = BytesIO()
            export_df = pd.concat([metadata, df], axis=1)
            export_df.to_csv(csv_buf, index=False)
            zf.writestr(f'data/{name}.csv', csv_buf.getvalue())
        
        # Full data
        full = pd.concat([processed.sample_data, processed.totals, processed.ratios, processed.percentages], axis=1)
        csv_buf = BytesIO()
        full.to_csv(csv_buf, index=False)
        zf.writestr('data/full_analysis.csv', csv_buf.getvalue())
        
        # LOD-highlighted Excel
        lod_handling = settings.get('lod_handling', 'half_lod') if settings else 'half_lod'
        lod_excel_bytes = create_full_data_excel_with_highlighting(processed, lod_handling)
        zf.writestr('data/full_data_lod_highlighted.xlsx', lod_excel_bytes)
        
        # Excel report
        if report_gen:
            excel_buf = BytesIO()
            report_gen.save_excel_report(excel_buf)
            excel_buf.seek(0)
            zf.writestr('reports/statistical_report.xlsx', excel_buf.getvalue())
        
        # Figures
        for name, fig in figures.items():
            if fig:
                zf.writestr(f'figures/{name}.png', fig_to_bytes(fig, 'png'))
                zf.writestr(f'figures/{name}.pdf', fig_to_bytes(fig, 'pdf'))
    
    buf.seek(0)
    return buf.getvalue()


def create_full_data_excel_with_highlighting(processed, lod_handling='half_lod') -> bytes:
    """Create Excel workbook with yellow highlighting on cells where LOD values were replaced."""
    from openpyxl.styles import PatternFill
    
    lod_display = {"half_lod": "LOD/2", "lod": "LOD value", "zero": "Zero", "drop": "NaN (excluded)"}
    
    buf = BytesIO()
    
    # Get metadata columns
    id_cols = []
    if processed.structure.sample_id_col and processed.structure.sample_id_col in processed.sample_data.columns:
        id_cols.append(processed.structure.sample_id_col)
    if processed.structure.group_col and processed.structure.group_col in processed.sample_data.columns:
        id_cols.append(processed.structure.group_col)
    
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
    
    st.sidebar.markdown("### 📊 Detection Limits")
    lod_handling = st.sidebar.selectbox(
        "Below-LOD handling",
        ["half_lod", "lod", "zero", "drop"],
        format_func=lambda x: {"half_lod": "LOD/2 (recommended)", "lod": "LOD value", 
                               "zero": "Zero", "drop": "NaN (exclude)"}[x],
        help="LOD values are auto-detected per analyte from the LC-MS standards sheet"
    )
    
    st.sidebar.markdown("### 📈 Analysis")
    alpha = st.sidebar.slider("Significance (α)", 0.01, 0.10, 0.05, 0.01)
    plot_type = st.sidebar.selectbox("Plot type", ["box", "violin", "bar"])
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
    
    return {'lod_handling': lod_handling, 'alpha': alpha, 'plot_type': plot_type,
            'show_points': show_points,
            'color_palette': color_palette, 'plot_style': plot_style}


def render_concentrations_tab(processed, settings):
    """Render concentrations tab."""
    st.markdown("### Individual Bile Acid Concentrations")
    
    viz = BileAcidVisualizer(color_palette=settings['color_palette'], style=settings['plot_style'])
    group_col = processed.structure.group_col
    results = st.session_state.analysis_results
    
    if not group_col:
        st.warning("No group column detected.")
        return
    
    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    data = data[data[group_col].astype(str).str.lower() != 'nan']
    available_bas = processed.structure.bile_acid_cols
    
    col1, col2 = st.columns([1, 2])
    with col1:
        quick = st.selectbox("Quick select", ["Top 10", "Significant", "Primary", "Secondary", "Custom"])
    
    if quick == "Top 10":
        selected = processed.concentrations.mean().nlargest(10).index.tolist()
    elif quick == "Significant":
        selected = [b for b in available_bas if b in results.individual_ba_results 
                   and results.individual_ba_results[b].main_test.significant][:10]
        if not selected:
            st.info("No significant individual BAs found.")
            selected = processed.concentrations.mean().nlargest(5).index.tolist()
    elif quick == "Primary":
        selected = [b for b in get_primary() if b in available_bas][:10]
    elif quick == "Secondary":
        selected = [b for b in get_secondary() if b in available_bas][:10]
    else:
        with col2:
            selected = st.multiselect("Select BAs", available_bas, available_bas[:5])
    
    log_scale = st.checkbox("Log₁₀ scale", key="conc_log")
    
    if selected:
        stats_dict = {b: results.individual_ba_results.get(b) for b in selected if b in results.individual_ba_results}
        
        # Display version
        fig = viz.plot_multi_panel_groups_with_stats(data, selected, group_col, stats_dict, 
                                                     ncols=3, plot_type=settings['plot_type'], log_scale=log_scale,
                                                     show_points=settings['show_points'])
        st.pyplot(fig)
        store_figure(fig, f'concentrations{"_log" if log_scale else ""}')
        plt.close(fig)
        
        # Generate other version for export
        fig_other = viz.plot_multi_panel_groups_with_stats(data, selected, group_col, stats_dict,
                                                           ncols=3, plot_type=settings['plot_type'], log_scale=not log_scale,
                                                           show_points=settings['show_points'])
        store_figure(fig_other, f'concentrations{"_log" if not log_scale else ""}')
        plt.close(fig_other)
        
        with st.expander("📊 Statistical Summary"):
            st.dataframe(get_significant_differences_summary(stats_dict), hide_index=True)


def render_totals_tab(processed, settings):
    """Render totals tab."""
    st.markdown("### Aggregate Totals")
    
    viz = BileAcidVisualizer(color_palette=settings['color_palette'], style=settings['plot_style'])
    group_col = processed.structure.group_col
    results = st.session_state.analysis_results
    
    if not group_col:
        return
    
    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    combined = pd.concat([data[[group_col]], processed.totals.loc[data.index]], axis=1)
    
    log_scale = st.checkbox("Log₁₀ scale", key="totals_log")
    
    # Key totals to show
    key_totals = ['total_all', 'total_primary', 'total_secondary', 'total_conjugated', 
                  'total_unconjugated', 'glycine_conjugated', 'taurine_conjugated']
    available = [t for t in key_totals if t in combined.columns]
    
    stats_dict = {t: results.totals_results.get(t) for t in available if t in results.totals_results}
    
    # Display version
    fig = viz.plot_multi_panel_groups_with_stats(combined, available, group_col, stats_dict,
                                                 ncols=3, plot_type=settings['plot_type'], log_scale=log_scale,
                                                 show_points=settings['show_points'])
    st.pyplot(fig)
    store_figure(fig, f'totals{"_log" if log_scale else ""}')
    plt.close(fig)
    
    # Generate other version for export
    fig_other = viz.plot_multi_panel_groups_with_stats(combined, available, group_col, stats_dict,
                                                       ncols=3, plot_type=settings['plot_type'], log_scale=not log_scale,
                                                       show_points=settings['show_points'])
    store_figure(fig_other, f'totals{"_log" if not log_scale else ""}')
    plt.close(fig_other)
    
    with st.expander("📊 Statistical Summary"):
        rows = []
        for t in available:
            res = results.totals_results.get(t)
            if res:
                rows.append({
                    'Total': t.replace('_', ' ').title(),
                    'P-value': f"{res.main_test.pvalue:.4f}",
                    'Significant': '✓' if res.main_test.significant else '',
                    'APA': format_apa_statistics(res)
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def render_percentages_tab(processed, settings):
    """Render percentages tab - comparing % composition across groups."""
    st.markdown("### Bile Acid Pool Composition (%)")
    st.caption("Compare the percentage each bile acid contributes to the total pool across groups")
    
    viz = BileAcidVisualizer(color_palette=settings['color_palette'], style=settings['plot_style'])
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
    
    col1, col2 = st.columns([1, 2])
    with col1:
        quick = st.selectbox("Quick select", 
                            ["Top 10 by mean %", "Significant", "Primary", "Secondary", 
                             "Glycine conjugated", "Taurine conjugated", "Custom"],
                            key="pct_quick")
    
    if quick == "Top 10 by mean %":
        top_pcts = percentages[pct_cols].mean().nlargest(10).index.tolist()
        selected_pct_cols = top_pcts
    elif quick == "Significant":
        selected_pct_cols = [c for c in pct_cols if c in results.percentages_results 
                           and results.percentages_results[c].main_test.significant][:10]
        if not selected_pct_cols:
            st.info("No significant percentage differences found. Showing top 10 by mean %.")
            selected_pct_cols = percentages[pct_cols].mean().nlargest(10).index.tolist()
    elif quick == "Primary":
        primary_bas = get_primary()
        selected_pct_cols = [f'{ba}_pct' for ba in primary_bas if f'{ba}_pct' in pct_cols][:10]
    elif quick == "Secondary":
        secondary_bas = get_secondary()
        selected_pct_cols = [f'{ba}_pct' for ba in secondary_bas if f'{ba}_pct' in pct_cols][:10]
    elif quick == "Glycine conjugated":
        glycine_bas = get_glycine_conjugated()
        selected_pct_cols = [f'{ba}_pct' for ba in glycine_bas if f'{ba}_pct' in pct_cols][:10]
    elif quick == "Taurine conjugated":
        taurine_bas = get_taurine_conjugated()
        selected_pct_cols = [f'{ba}_pct' for ba in taurine_bas if f'{ba}_pct' in pct_cols][:10]
    else:
        with col2:
            selected_bas = st.multiselect("Select bile acids", ba_names, ba_names[:5], key="pct_custom")
            selected_pct_cols = [f'{ba}_pct' for ba in selected_bas]
    
    if not selected_pct_cols:
        st.warning("No bile acids selected.")
        return
    
    # Show significant findings summary
    sig_pcts = [c for c in selected_pct_cols if c in results.percentages_results 
               and results.percentages_results[c].main_test.significant]
    if sig_pcts:
        sig_names = [c.replace('_pct', '') for c in sig_pcts]
        st.success(f"**Significant differences:** {', '.join(sig_names)}")
    
    # Combine data for plotting
    plot_df = pd.concat([data[[group_col]].reset_index(drop=True), 
                         percentages[selected_pct_cols].reset_index(drop=True)], axis=1)
    
    # Rename columns for cleaner display (remove _pct suffix)
    display_cols = {col: col.replace('_pct', '') for col in selected_pct_cols}
    plot_df_display = plot_df.rename(columns=display_cols)
    display_names = list(display_cols.values())
    
    # Get stats results with display names
    stats_dict = {}
    for pct_col in selected_pct_cols:
        display_name = pct_col.replace('_pct', '')
        if pct_col in results.percentages_results:
            stats_dict[display_name] = results.percentages_results[pct_col]
    
    # =========================================================================
    # SECTION 1: Statistical comparison (box/violin plots)
    # =========================================================================
    st.markdown("#### Statistical Comparison by Group")
    
    fig = viz.plot_multi_panel_groups_with_stats(
        plot_df_display, display_names, group_col, stats_dict,
        ncols=3, plot_type=settings['plot_type'], log_scale=False,
        show_points=settings['show_points']
    )
    
    # Update y-axis labels to show percentage
    for ax in fig.get_axes():
        if ax.get_visible():
            ax.set_ylabel('% of Total BA')
    
    st.pyplot(fig)
    store_figure(fig, 'percentages')
    plt.close(fig)
    
    # =========================================================================
    # SECTION 2: Pie Charts - Pool Composition
    # =========================================================================
    st.markdown("---")
    st.markdown("#### Pool Composition - Pie Charts")
    st.caption("Visual breakdown of bile acid pool for each group (top contributors)")
    
    # Use top 15 percentage columns for pie charts (show full composition)
    all_pct_cols = percentages[pct_cols].mean().nlargest(15).index.tolist()
    pie_df = pd.concat([data[[group_col]].reset_index(drop=True), 
                        percentages[all_pct_cols].reset_index(drop=True)], axis=1)
    
    fig_pie = viz.plot_composition_pie_charts(
        pie_df, group_col, all_pct_cols,
        title='Bile Acid Pool Composition by Group',
        top_n=10, other_threshold=2.0
    )
    st.pyplot(fig_pie)
    store_figure(fig_pie, 'percentages_pie')
    plt.close(fig_pie)
    
    # =========================================================================
    # SECTION 3: Horizontal Bar Charts - Group Comparison
    # =========================================================================
    st.markdown("---")
    st.markdown("#### Pool Composition - Bar Comparison")
    st.caption("Side-by-side comparison of bile acid percentages across groups")
    
    fig_hbar = viz.plot_composition_horizontal_bars(
        pie_df, group_col, all_pct_cols,
        title='Bile Acid Pool Composition Comparison',
        top_n=15, show_values=True
    )
    st.pyplot(fig_hbar)
    store_figure(fig_hbar, 'percentages_bars')
    plt.close(fig_hbar)
    
    # =========================================================================
    # SECTION 4: Stacked Bar - Full Composition
    # =========================================================================
    st.markdown("---")
    st.markdown("#### Stacked Composition View")
    st.caption("Full bile acid pool breakdown for each group")
    
    fig_stacked = viz.plot_composition_stacked_horizontal(
        pie_df, group_col, all_pct_cols,
        title='Complete Bile Acid Pool Composition',
        top_n=12
    )
    st.pyplot(fig_stacked)
    store_figure(fig_stacked, 'percentages_stacked')
    plt.close(fig_stacked)
    
    # =========================================================================
    # SECTION 5: Statistical Summary Table
    # =========================================================================
    with st.expander("📊 Statistical Summary"):
        rows = []
        for pct_col in selected_pct_cols:
            ba_name = pct_col.replace('_pct', '')
            res = results.percentages_results.get(pct_col)
            if res:
                n_sig = 0
                if res.posthoc_test and res.posthoc_test.pairwise_results is not None:
                    n_sig = res.posthoc_test.pairwise_results['significant'].sum()
                rows.append({
                    'Bile Acid': ba_name,
                    'Test': res.main_test.test_type.value,
                    'P-value': f"{res.main_test.pvalue:.4f}",
                    'Significant': '✓' if res.main_test.significant else '',
                    'Effect Size': f"{res.main_test.effect_size:.3f}" if res.main_test.effect_size else 'N/A',
                    'Sig. Pairs': n_sig,
                    'APA': format_apa_statistics(res)
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
    
    # Group mean percentages table
    with st.expander("📈 Group Mean Percentages"):
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
            summary_df = pd.DataFrame(summary_data)
            # Pivot for easier reading
            pivot_df = summary_df.pivot_table(
                index='Bile Acid', 
                columns='Group', 
                values='Mean %',
                aggfunc='first'
            )
            st.dataframe(pivot_df, use_container_width=True)
    
    # Raw data view
    with st.expander("📋 View percentage data"):
        st.dataframe(plot_df, use_container_width=True)


def render_ratios_tab(processed, settings):
    """Render ratios tab with multi-panel display."""
    st.markdown("### Clinical Ratios")
    
    viz = BileAcidVisualizer(color_palette=settings['color_palette'], style=settings['plot_style'])
    group_col = processed.structure.group_col
    results = st.session_state.analysis_results
    
    if not group_col:
        st.warning("No group column detected.")
        return
    
    data = processed.sample_data[processed.sample_data[group_col].notna()].copy()
    data = data[data[group_col].astype(str).str.lower() != 'nan']
    combined = pd.concat([data[[group_col]], processed.ratios.loc[data.index]], axis=1)
    available = [c for c in processed.ratios.columns if not combined[c].isna().all()]
    
    if not available:
        st.warning("No ratio data available.")
        return
    
    # Quick select options
    col1, col2 = st.columns([1, 2])
    with col1:
        quick = st.selectbox("Quick select", 
                            ["All ratios", "Significant only", "Key ratios", "Custom"],
                            key="ratio_quick")
    
    # Define key clinical ratios
    key_ratios = ['primary_to_secondary', 'glycine_to_taurine', 'conjugated_to_unconjugated',
                  'CA_to_CDCA', 'TCA_to_GCA', 'GCDCA_to_TCDCA']
    
    if quick == "All ratios":
        selected_ratios = available
    elif quick == "Significant only":
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
        with col2:
            selected_ratios = st.multiselect("Select ratios", available, 
                                             default=available[:6] if len(available) >= 6 else available,
                                             key="ratio_custom")
    
    if not selected_ratios:
        st.warning("No ratios selected.")
        return
    
    # Show significant findings
    sig_ratios = [r for r in selected_ratios if r in results.ratios_results 
                 and results.ratios_results[r].main_test.significant]
    if sig_ratios:
        st.success(f"**Significant differences:** {', '.join(sig_ratios)}")
    
    log_scale = st.checkbox("Log₁₀ scale", key="ratio_log")
    
    # Get stats for selected ratios
    stats_dict = {r: results.ratios_results.get(r) for r in selected_ratios if r in results.ratios_results}
    
    # Multi-panel figure
    fig = viz.plot_multi_panel_groups_with_stats(
        combined, selected_ratios, group_col, stats_dict,
        ncols=3, plot_type=settings['plot_type'], log_scale=log_scale,
        show_points=settings['show_points']
    )
    st.pyplot(fig)
    store_figure(fig, f'ratios{"_log" if log_scale else ""}')
    plt.close(fig)
    
    # Generate other version for export
    fig_other = viz.plot_multi_panel_groups_with_stats(
        combined, selected_ratios, group_col, stats_dict,
        ncols=3, plot_type=settings['plot_type'], log_scale=not log_scale,
        show_points=settings['show_points']
    )
    store_figure(fig_other, f'ratios{"_log" if not log_scale else ""}')
    plt.close(fig_other)
    
    # Statistical summary
    with st.expander("📊 Statistical Summary"):
        rows = []
        for r in selected_ratios:
            res = results.ratios_results.get(r)
            if res:
                n_sig = 0
                if res.posthoc_test and res.posthoc_test.pairwise_results is not None:
                    n_sig = res.posthoc_test.pairwise_results['significant'].sum()
                rows.append({
                    'Ratio': r,
                    'Test': res.main_test.test_type.value,
                    'P-value': f"{res.main_test.pvalue:.4f}",
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
            st.dataframe(pivot_df, use_container_width=True)
    
    # Raw data
    with st.expander("📋 View ratio data"):
        display_cols = [group_col] + selected_ratios
        st.dataframe(combined[display_cols], use_container_width=True)


def render_statistics_tab(processed, settings):
    """Statistics summary tab."""
    st.markdown("### Statistics Summary")
    results = st.session_state.analysis_results
    
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
        # Clean up names for display
        sig_pct_display = {k.replace('_pct', ''): v for k, v in sig_pct.items()}
        st.dataframe(get_significant_differences_summary(sig_pct_display), hide_index=True)
    else:
        st.info("No significant percentage differences.")


@st.fragment
def render_export_tab(processed, settings):
    """Export tab."""
    st.markdown("### Export Results")
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
    
        # Export each dataset with metadata
        for name, df in [("Concentrations", processed.concentrations), 
                     ("Totals", processed.totals),
                     ("Ratios", processed.ratios), 
                     ("Percentages", processed.percentages)]:
            export_df = pd.concat([metadata, df], axis=1)
            st.download_button(
                f"📥 {name} (CSV)", 
                export_df.to_csv(index=False), 
                f"bile_acid_{name.lower()}.csv", 
                "text/csv",
                key=f"download_{name.lower()}"
            )
        
        # LOD-highlighted Excel export
        st.markdown("---")
        lod_excel = create_full_data_excel_with_highlighting(processed, settings['lod_handling'])
        lod_label = {"half_lod": "LOD/2", "lod": "LOD value", "zero": "Zero", "drop": "NaN"}
        st.download_button(
            "📥 All Data (Excel) - LOD highlighted",
            lod_excel,
            f"bile_acid_data_lod_highlighted_{datetime.now():%Y%m%d}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_lod_excel"
        )
        st.caption(f"Yellow cells = values replaced (below LOD → {lod_label[settings['lod_handling']]})")
    
    with col2:
        st.markdown("#### Complete Reports")
        if report_gen:
            buf = BytesIO()
            report_gen.save_excel_report(buf)
            buf.seek(0)
            st.download_button("📥 Statistical Report (Excel)", buf.getvalue(),
                              f"statistical_report_{datetime.now():%Y%m%d}.xlsx",
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              key="download_excel_report")
        
        # Generate all figures and ZIP
        with st.spinner("Generating all figures for export..."):
            # Generate comprehensive figure set for all dropdown options
            all_figures = generate_all_export_figures(processed, results, settings)
            # Merge with any existing figures (e.g., from viewing tabs)
            combined_figures = {**st.session_state.figures, **all_figures}
            zip_bytes = create_results_zip(processed, results, combined_figures, report_gen, settings)
        
        st.download_button("📥 Complete Package (ZIP)", zip_bytes, 
                          f"bile_acid_analysis_{datetime.now():%Y%m%d_%H%M}.zip", 
                          "application/zip",
                          key="download_zip")
        
        st.caption(f"📊 Package includes {len(combined_figures)} figures covering all analysis options")
    
    st.markdown("---")
    st.markdown("#### Summary Figure")
    if report_gen and st.button("🎨 Generate Summary Figure", key="gen_summary_fig"):
        plotter = SignificancePlotter()
        fig = plotter.plot_multi_panel_with_significance(
            processed.sample_data, processed.structure.group_col, report_gen,
            ['Total_All_BAs', 'Total_Primary', 'Total_Secondary', 'Total_Conjugated'], 2, settings['plot_type'])
        st.pyplot(fig)
        store_figure(fig, 'summary')
        c1, c2 = st.columns(2)
        c1.download_button("📥 PNG", fig_to_bytes(fig), "summary.png", "image/png", key="download_summary_png")
        c2.download_button("📥 PDF", fig_to_bytes(fig, 'pdf'), "summary.pdf", "application/pdf", key="download_summary_pdf")
        plt.close(fig)


def main():
    """Main entry point."""
    init_session_state()
    settings = render_sidebar()
    
    st.markdown("# 🧬 Bile Acid Analysis Pipeline")
    uploaded = st.file_uploader("Upload Excel/ODS file", ['xlsx', 'xls', 'ods'])
    
    if uploaded:
        # Check if file changed
        file_changed = st.session_state.last_file != uploaded.name
        if file_changed:
            st.session_state.stats_computed = False
            st.session_state.figures = {}
            st.session_state.last_file = uploaded.name
            st.session_state.last_settings = None  # Reset settings tracking for new file
        
        # Check if settings changed (LOD handling, alpha, etc.)
        settings_changed = check_settings_changed(settings)
        
        # Reprocess data if file or settings changed
        if file_changed or settings_changed or st.session_state.processed_data is None:
            with st.spinner("Processing data..." + (" (settings changed)" if settings_changed else "")):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                
                try:
                    processor = BileAcidDataProcessor(lod_handling=settings['lod_handling'])
                    processed = processor.load_and_process(tmp_path)
                    st.session_state.processed_data = processed
                    if settings_changed:
                        st.success("✅ Data reprocessed with new settings!")
                    else:
                        st.success("✅ Data loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
    
    if st.session_state.processed_data:
        processed = st.session_state.processed_data
        
        quality = validate_data_quality(processed)
        c1, c2, c3 = st.columns(3)
        c1.metric("Samples", quality['n_samples'])
        c2.metric("Bile Acids", quality['n_bile_acids_detected'])
        c3.metric("Groups", quality.get('n_groups', 'N/A'))
        
        # Show LOD info
        lod_source_label = "Auto-detected from standards" if quality.get('lod_source') == 'standards' else "Default (no standards found)"
        lod_display = {"half_lod": "LOD/2", "lod": "LOD value", "zero": "Zero", "drop": "NaN"}
        st.caption(f"📊 LOD: **{lod_source_label}** (below-LOD → {lod_display[settings['lod_handling']]}) | α = **{settings['alpha']}**")
        
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
                    st.dataframe(lod_df, use_container_width=True, hide_index=True)
        
        with st.spinner("Computing statistics..."):
            compute_all_statistics(processed, settings)
        
        tabs = st.tabs(["📈 Concentrations", "📊 Totals", "📉 Percentages", "🔢 Ratios", "📊 Statistics", "💾 Export"])
        with tabs[0]: render_concentrations_tab(processed, settings)
        with tabs[1]: render_totals_tab(processed, settings)
        with tabs[2]: render_percentages_tab(processed, settings)
        with tabs[3]: render_ratios_tab(processed, settings)
        with tabs[4]: render_statistics_tab(processed, settings)
        with tabs[5]: render_export_tab(processed, settings)
    
    else:
        # Show expected input format when no data is loaded
        st.markdown("---")
        st.markdown("### 📋 Expected Input Format")
        
        st.markdown("""
        - **Rows:** Samples
        - **Columns:** Bile acid species (matching panel names)
        - **First column(s):** Sample ID, Group/Type
        - **Values:** Concentrations (typically nmol/L)
        - **Below LOD:** Can be "-----", "LOD", "BLQ", "ND", etc.
        """)
        
        st.markdown("**Example:**")
        
        # Create example dataframe
        example_df = pd.DataFrame({
            'Type': ['HD-1', 'HD-2', 'AC-1'],
            'Sample_ID': ['81-0210', '81-0211', '60-677'],
            'TCA': [37.62, 66.66, 45.2],
            'GCA': [194.1, 287.58, '-----'],
            'TCDCA': [231.9, 158.46, 189.3],
            'GCDCA': [3220.44, 1534.02, 2890.1],
            '...': ['...', '...', '...']
        })
        st.dataframe(example_df, hide_index=True, use_container_width=False)
        
        st.info("💡 The pipeline auto-detects bile acid columns and group assignments from your data.")


if __name__ == "__main__":
    main()
