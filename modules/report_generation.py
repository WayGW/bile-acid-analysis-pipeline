"""
Report Generation Module for Bile Acid Analysis
=================================================

Generates organized Excel workbooks with:
- Separate sheets for each analysis type
- Raw data → calculations → statistical results on each sheet
- Transparency in data flow

Also handles figure generation with significance annotations.
"""

import gc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import from other modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.bile_acid_species import (
    BILE_ACID_PANEL, ANALYSIS_GROUPS, CLINICAL_RATIOS,
    get_glycine_conjugated, get_taurine_conjugated,
    get_primary, get_secondary, get_conjugated, get_unconjugated,
    get_sulfated, get_keto_derivatives, get_iso_forms, get_nor_bile_acids,
    get_12alpha_hydroxylated, get_non12alpha_hydroxylated
)
from modules.statistical_tests import (
    StatisticalAnalyzer, FullAnalysisResult,
    TwoWayResult, FullTwoWayAnalysisResult, format_twoway_apa,
    ThreeWayResult, FullThreeWayAnalysisResult, format_threeway_apa
)


@dataclass
class ComprehensiveAnalysisResults:
    """Container for all statistical results."""
    individual_ba_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)
    totals_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)
    ratios_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)
    percentages_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)
    category_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)

    # Two-way ANOVA results (populated when n_factors == 2)
    is_twoway: bool = False
    twoway_individual_ba: Dict[str, FullTwoWayAnalysisResult] = field(default_factory=dict)
    twoway_totals: Dict[str, FullTwoWayAnalysisResult] = field(default_factory=dict)
    twoway_ratios: Dict[str, FullTwoWayAnalysisResult] = field(default_factory=dict)
    twoway_percentages: Dict[str, FullTwoWayAnalysisResult] = field(default_factory=dict)
    factor_a_name: str = ""
    factor_b_name: str = ""
    factor_a_col: str = ""
    factor_b_col: str = ""

    # Three-way ANOVA results (populated when n_factors == 3)
    is_threeway: bool = False
    threeway_individual_ba: Dict[str, FullThreeWayAnalysisResult] = field(default_factory=dict)
    threeway_totals: Dict[str, FullThreeWayAnalysisResult] = field(default_factory=dict)
    threeway_ratios: Dict[str, FullThreeWayAnalysisResult] = field(default_factory=dict)
    threeway_percentages: Dict[str, FullThreeWayAnalysisResult] = field(default_factory=dict)
    factor_c_name: str = ""
    factor_c_col: str = ""

    # Log₁₀-transformed results (parallel to raw results above — excludes percentages)
    log_individual_ba_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)
    log_totals_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)
    log_ratios_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)
    log_twoway_individual_ba: Dict[str, FullTwoWayAnalysisResult] = field(default_factory=dict)
    log_twoway_totals: Dict[str, FullTwoWayAnalysisResult] = field(default_factory=dict)
    log_twoway_ratios: Dict[str, FullTwoWayAnalysisResult] = field(default_factory=dict)
    log_threeway_individual_ba: Dict[str, FullThreeWayAnalysisResult] = field(default_factory=dict)
    log_threeway_totals: Dict[str, FullThreeWayAnalysisResult] = field(default_factory=dict)
    log_threeway_ratios: Dict[str, FullThreeWayAnalysisResult] = field(default_factory=dict)

    # LOD exclusion tracking
    lod_excluded: Dict[str, float] = field(default_factory=dict)  # {analyte: lod_pct}
    lod_threshold: int = 50


@dataclass
class AnalysisSheet:
    """Data structure for a single analysis sheet."""
    name: str
    description: str
    bile_acid_columns: List[str]
    raw_data: pd.DataFrame
    group_totals: pd.DataFrame
    group_percentages: pd.DataFrame
    statistical_result: Optional[FullAnalysisResult]


def format_apa_statistics(result: FullAnalysisResult) -> str:
    """Format statistical result in APA style."""
    if not result:
        return ""
    
    test = result.main_test
    test_type = test.test_type.value
    
    # Format based on test type
    if 'kruskal' in test_type.lower():
        # Kruskal-Wallis: H(df) = X.XX, p = .XXX, η² = X.XX
        df = len(result.assumptions.group_sizes) - 1
        stat_str = f"H({df}) = {test.statistic:.2f}"
    elif 'anova' in test_type.lower() or 'f_oneway' in test_type.lower():
        # ANOVA: F(df1, df2) = X.XX, p = .XXX, η² = X.XX
        df1 = len(result.assumptions.group_sizes) - 1
        df2 = result.assumptions.total_n - len(result.assumptions.group_sizes)
        stat_str = f"F({df1}, {df2}) = {test.statistic:.2f}"
    elif 'mann_whitney' in test_type.lower():
        stat_str = f"U = {test.statistic:.2f}"
    elif 't_test' in test_type.lower() or 'ttest' in test_type.lower():
        df = test.df if test.df else result.assumptions.total_n - 2
        stat_str = f"t({df:.0f}) = {test.statistic:.2f}"
    else:
        stat_str = f"stat = {test.statistic:.2f}"
    
    # Format p-value
    if test.pvalue < 0.001:
        p_str = "p < .001"
    else:
        p_str = f"p = {test.pvalue:.3f}"
    
    # Effect size
    if test.effect_size is not None:
        if test.effect_size_type == 'eta_squared':
            effect_str = f"η² = {test.effect_size:.2f}"
        elif test.effect_size_type == 'cohens_d':
            effect_str = f"d = {test.effect_size:.2f}"
        else:
            effect_str = f"{test.effect_size_type} = {test.effect_size:.2f}"
        return f"{stat_str}, {p_str}, {effect_str}"
    
    return f"{stat_str}, {p_str}"


def get_significant_differences_summary(results: Dict[str, FullAnalysisResult]) -> pd.DataFrame:
    """Create summary table of statistical results."""
    rows = []
    for name, result in results.items():
        if result is None:
            continue
        
        n_sig_pairs = 0
        sig_pairs_str = ""
        if result.posthoc_test and result.posthoc_test.pairwise_results is not None:
            sig_df = result.posthoc_test.pairwise_results[result.posthoc_test.pairwise_results['significant']]
            n_sig_pairs = len(sig_df)
            if n_sig_pairs > 0:
                pairs = [f"{r['group1']} vs {r['group2']}" for _, r in sig_df.head(3).iterrows()]
                sig_pairs_str = "; ".join(pairs)
                if n_sig_pairs > 3:
                    sig_pairs_str += f" (+{n_sig_pairs - 3} more)"
        
        rows.append({
            'Variable': name,
            'Test': result.main_test.test_type.value,
            'Statistic': f"{result.main_test.statistic:.3f}",
            'P-value': f"{result.main_test.pvalue:.4f}",
            'Significant': '✓' if result.main_test.significant else '',
            'Effect Size': f"{result.main_test.effect_size:.3f}" if result.main_test.effect_size else 'N/A',
            'Interpretation': result.main_test.effect_size_interpretation or '',
            'Sig. Pairs': n_sig_pairs,
            'Pairwise': sig_pairs_str
        })
    
    return pd.DataFrame(rows)


def get_twoway_differences_summary(results: Dict[str, FullTwoWayAnalysisResult]) -> pd.DataFrame:
    """Create summary table of two-way ANOVA results across all analytes."""
    rows = []
    for name, result in results.items():
        if result is None:
            continue

        tw = result.twoway_result

        def _stars(p):
            if np.isnan(p): return ''
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return ''

        n_sig_posthoc = 0
        if tw.posthoc_results is not None:
            n_sig_posthoc = tw.posthoc_results['significant'].sum()

        rows.append({
            'Variable': name,
            'Test': tw.test_type.value,
            f'{tw.factor_a_name} F': f"{tw.factor_a_stat:.2f}" if not np.isnan(tw.factor_a_stat) else 'N/A',
            f'{tw.factor_a_name} p': f"{tw.factor_a_pvalue:.4f}" if not np.isnan(tw.factor_a_pvalue) else 'N/A',
            f'{tw.factor_a_name} sig': _stars(tw.factor_a_pvalue),
            f'{tw.factor_b_name} F': f"{tw.factor_b_stat:.2f}" if not np.isnan(tw.factor_b_stat) else 'N/A',
            f'{tw.factor_b_name} p': f"{tw.factor_b_pvalue:.4f}" if not np.isnan(tw.factor_b_pvalue) else 'N/A',
            f'{tw.factor_b_name} sig': _stars(tw.factor_b_pvalue),
            'Interaction F': f"{tw.interaction_stat:.2f}" if not np.isnan(tw.interaction_stat) else 'N/A',
            'Interaction p': f"{tw.interaction_pvalue:.4f}" if not np.isnan(tw.interaction_pvalue) else 'N/A',
            'Interaction sig': _stars(tw.interaction_pvalue),
            'Post-hoc type': tw.posthoc_type,
            'Sig. post-hoc pairs': n_sig_posthoc,
        })

    return pd.DataFrame(rows)


def get_threeway_differences_summary(results: Dict[str, FullThreeWayAnalysisResult]) -> pd.DataFrame:
    """Create summary table of three-way ANOVA results across all analytes."""
    rows = []
    for name, result in results.items():
        if result is None:
            continue

        tw = result.threeway_result

        def _stars(p):
            if np.isnan(p): return ''
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return ''

        def _fmt(val):
            return f"{val:.2f}" if not np.isnan(val) else 'N/A'

        def _fmt_p(val):
            return f"{val:.4f}" if not np.isnan(val) else 'N/A'

        n_sig_posthoc = 0
        if tw.posthoc_results is not None:
            n_sig_posthoc = tw.posthoc_results['significant'].sum()

        rows.append({
            'Variable': name,
            'Test': tw.test_type.value,
            f'{tw.factor_a_name} F': _fmt(tw.factor_a_stat),
            f'{tw.factor_a_name} p': _fmt_p(tw.factor_a_pvalue),
            f'{tw.factor_a_name} sig': _stars(tw.factor_a_pvalue),
            f'{tw.factor_b_name} F': _fmt(tw.factor_b_stat),
            f'{tw.factor_b_name} p': _fmt_p(tw.factor_b_pvalue),
            f'{tw.factor_b_name} sig': _stars(tw.factor_b_pvalue),
            f'{tw.factor_c_name} F': _fmt(tw.factor_c_stat),
            f'{tw.factor_c_name} p': _fmt_p(tw.factor_c_pvalue),
            f'{tw.factor_c_name} sig': _stars(tw.factor_c_pvalue),
            f'{tw.factor_a_name}\u00d7{tw.factor_b_name} p': _fmt_p(tw.interaction_ab_pvalue),
            f'{tw.factor_a_name}\u00d7{tw.factor_c_name} p': _fmt_p(tw.interaction_ac_pvalue),
            f'{tw.factor_b_name}\u00d7{tw.factor_c_name} p': _fmt_p(tw.interaction_bc_pvalue),
            f'{tw.factor_a_name}\u00d7{tw.factor_b_name}\u00d7{tw.factor_c_name} p': _fmt_p(tw.interaction_abc_pvalue),
            'Post-hoc type': tw.posthoc_type,
            'Sig. post-hoc pairs': n_sig_posthoc,
        })

    return pd.DataFrame(rows)


class ExcelReportGenerator:
    """
    Generate organized Excel reports with multiple sheets.
    
    Each sheet contains:
    1. Raw concentration data for relevant bile acids
    2. Calculated totals per sample
    3. Group summary statistics
    4. Statistical test results
    """
    
    # Define analysis categories and their bile acid members
    ANALYSIS_CATEGORIES = {
        'Total_All_BAs': {
            'description': 'All bile acids in panel',
            'get_columns': lambda available: available,
        },
        'Total_Conjugated': {
            'description': 'All conjugated bile acids (Glycine + Taurine + Sulfated)',
            'get_columns': lambda available: [c for c in get_conjugated() if c in available],
        },
        'Total_Unconjugated': {
            'description': 'All unconjugated (free) bile acids',
            'get_columns': lambda available: [c for c in get_unconjugated() if c in available],
        },
        'Glycine_Conjugated': {
            'description': 'Glycine-conjugated bile acids',
            'get_columns': lambda available: [c for c in get_glycine_conjugated() if c in available],
        },
        'Taurine_Conjugated': {
            'description': 'Taurine-conjugated bile acids',
            'get_columns': lambda available: [c for c in get_taurine_conjugated() if c in available],
        },
        'Sulfated': {
            'description': 'Sulfated bile acids',
            'get_columns': lambda available: [c for c in get_sulfated() if c in available],
        },
        'Total_Primary': {
            'description': 'Primary bile acids (synthesized in liver)',
            'get_columns': lambda available: [c for c in get_primary() if c in available],
        },
        'Total_Secondary': {
            'description': 'Secondary bile acids (bacterial modification)',
            'get_columns': lambda available: [c for c in get_secondary() if c in available],
        },
        'Oxidized_Keto': {
            'description': 'Oxidized (keto) bile acids',
            'get_columns': lambda available: [c for c in get_keto_derivatives() if c in available],
        },
        'Epimerized_Iso': {
            'description': 'Epimerized (iso/allo) bile acids',
            'get_columns': lambda available: [c for c in get_iso_forms() if c in available],
        },
        'Nor_Bile_Acids': {
            'description': 'Nor bile acids (side-chain shortened)',
            'get_columns': lambda available: [c for c in get_nor_bile_acids() if c in available],
        },
        '12alpha_Hydroxylated': {
            'description': '12-alpha hydroxylated bile acids (CYP8B1-dependent)',
            'get_columns': lambda available: [c for c in get_12alpha_hydroxylated() if c in available],
        },
        'Non_12alpha_Hydroxylated': {
            'description': 'Non-12-alpha hydroxylated bile acids',
            'get_columns': lambda available: [c for c in get_non12alpha_hydroxylated() if c in available],
        },
        'Primary_Conjugated': {
            'description': 'Primary conjugated bile acids',
            'get_columns': lambda available: [c for c in get_primary() if c in available and c in get_conjugated()],
        },
        'Primary_Unconjugated': {
            'description': 'Primary unconjugated bile acids',
            'get_columns': lambda available: [c for c in get_primary() if c in available and c in get_unconjugated()],
        },
        'Secondary_Conjugated': {
            'description': 'Secondary conjugated bile acids',
            'get_columns': lambda available: [c for c in get_secondary() if c in available and c in get_conjugated()],
        },
        'Secondary_Unconjugated': {
            'description': 'Secondary unconjugated bile acids',
            'get_columns': lambda available: [c for c in get_secondary() if c in available and c in get_unconjugated()],
        },
    }
    
    def __init__(
        self,
        data: pd.DataFrame,
        group_col: str,
        sample_id_col: Optional[str] = None,
        bile_acid_cols: Optional[List[str]] = None,
        totals: Optional[pd.DataFrame] = None,
        ratios: Optional[pd.DataFrame] = None,
        percentages: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
        # Two-way ANOVA factor info
        factors: Optional[Dict[str, str]] = None,  # {display_name: column_name}
        n_factors: int = 0,
        # LOD exclusion settings
        analyte_lod_counts: Optional[Dict[str, int]] = None,
        analyte_lod_rows: Optional[Dict[str, List[int]]] = None,
        n_samples: int = 0,
        lod_threshold: int = 50,
        units: str = "nmol/L",
    ):
        """
        Initialize report generator.

        Args:
            data: DataFrame with sample data
            group_col: Column name for group labels
            sample_id_col: Column name for sample IDs
            bile_acid_cols: List of bile acid column names
            totals: Pre-computed totals DataFrame
            ratios: Pre-computed ratios DataFrame
            percentages: Pre-computed percentages DataFrame
            alpha: Significance level for statistical tests
            factors: Dict of {factor_display_name: factor_column_name} for two-way ANOVA
            n_factors: Number of factors (0=auto, 1=one-way, 2=two-way)
            analyte_lod_counts: Dict of {analyte_name: count_of_lod_replaced_values}
            analyte_lod_rows: Dict of {analyte_name: [row_indices_with_lod_replacement]}
            n_samples: Total number of samples (for computing LOD percentages)
            lod_threshold: Exclude analytes with >= this % LOD-replaced from stats (0=disable)
        """
        self.data = data  # Reference only — avoid copy to save memory on Cloud
        self.group_col = group_col
        self.sample_id_col = sample_id_col
        self.alpha = alpha
        self.totals = totals
        self.ratios = ratios
        self.percentages = percentages

        # Two-way factor info
        self.factors = factors or {}
        self.n_factors = n_factors

        # LOD exclusion settings
        self.analyte_lod_counts = analyte_lod_counts or {}
        self.analyte_lod_rows = analyte_lod_rows or {}
        self.n_samples = n_samples if n_samples > 0 else len(data)
        self.lod_threshold = lod_threshold
        self.lod_excluded: Dict[str, float] = {}  # {analyte: lod_pct} for excluded analytes
        self.units = units

        # Identify bile acid columns
        if bile_acid_cols:
            self.ba_cols = [c for c in bile_acid_cols if c in data.columns]
        else:
            # Auto-detect from panel
            self.ba_cols = [c for c in BILE_ACID_PANEL.keys() if c in data.columns]

        self.analyzer = StatisticalAnalyzer(alpha=alpha)
        self.analysis_sheets: Dict[str, AnalysisSheet] = {}
        self.results = ComprehensiveAnalysisResults()

    def _should_exclude_for_lod(self, col: str, data: pd.DataFrame = None) -> bool:
        """
        Check if an analyte should be excluded from statistical testing
        due to high LOD replacement rate.

        Uses the modified rule: an analyte is KEPT if any single group
        has >= (100 - threshold)% detected (non-LOD) values. This preserves
        analytes where detection rate varies meaningfully across groups
        (e.g., detected in treatment but not control).

        Returns True if the analyte should be EXCLUDED.
        """
        if self.lod_threshold <= 0:
            return False  # Threshold disabled

        lod_count = self.analyte_lod_counts.get(col, 0)
        if lod_count == 0:
            return False  # No LOD replacements

        # Global LOD percentage
        lod_pct = (lod_count / self.n_samples * 100) if self.n_samples > 0 else 0

        if lod_pct < self.lod_threshold:
            return False  # Below threshold globally

        # Modified rule: check per-group using LOD row indices
        lod_row_set = set(self.analyte_lod_rows.get(col, []))
        if lod_row_set and data is not None and self.group_col in data.columns:
            min_detected_pct = 100 - self.lod_threshold
            for group_name, group_df in data.groupby(self.group_col):
                n_group = len(group_df)
                if n_group == 0:
                    continue
                # Count how many of this group's rows are LOD-replaced
                n_lod_in_group = len(lod_row_set.intersection(group_df.index))
                group_detected_pct = ((n_group - n_lod_in_group) / n_group * 100)
                # If this group has enough detected values, keep the analyte
                if group_detected_pct >= min_detected_pct:
                    return False

        # Exceeded threshold in all groups — exclude
        self.lod_excluded[col] = round(lod_pct, 1)
        return True
    
    def _should_exclude_category_for_lod(self, category_col: str, data: pd.DataFrame = None) -> bool:
        """
        Check if a totals category should be excluded from statistical testing
        based on the LOD status of its constituent bile acids.

        A category is excluded if the proportion of its constituent BAs that were
        LOD-excluded (or would be) meets or exceeds the LOD threshold. This prevents
        categories like nor_bile_acids from showing significance when nearly all
        constituent BAs are LOD-replaced.
        """
        if self.lod_threshold <= 0:
            return False

        # Look up constituent BAs for this category
        if category_col == 'total_all':
            constituent_bas = [c for c in self.ba_cols if c in (self.data.columns if data is None else data.columns)]
        elif category_col in ANALYSIS_GROUPS:
            constituent_bas = [c for c in ANALYSIS_GROUPS[category_col] if c in self.ba_cols]
        else:
            return False  # Unknown category, don't exclude

        if not constituent_bas:
            return False

        # Count how many constituents are LOD-excluded or would be
        n_excluded = 0
        for ba in constituent_bas:
            if ba in self.lod_excluded:
                n_excluded += 1
            elif self._should_exclude_for_lod(ba, data):
                # This BA wasn't checked yet but would be excluded
                n_excluded += 1

        excluded_pct = (n_excluded / len(constituent_bas)) * 100
        if excluded_pct >= self.lod_threshold:
            self.lod_excluded[category_col] = round(excluded_pct, 1)
            return True

        return False

    def _finalize_results(self) -> ComprehensiveAnalysisResults:
        """Attach LOD exclusion info to results before returning."""
        self.results.lod_excluded = self.lod_excluded.copy()
        self.results.lod_threshold = self.lod_threshold
        return self.results

    def _get_valid_data(self) -> pd.DataFrame:
        """Filter valid groups from data (cached)."""
        if not hasattr(self, '_valid_data_cache'):
            vd = self.data[self.data[self.group_col].notna()].copy()
            self._valid_data_cache = vd[vd[self.group_col].astype(str).str.lower() != 'nan']
        return self._valid_data_cache

    @staticmethod
    def _log(msg: str):
        """Print a timestamped log message (visible in Streamlit Cloud logs)."""
        from datetime import datetime
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    # ------------------------------------------------------------------
    # Section runners — called individually per tab
    # ------------------------------------------------------------------

    def run_section(self, section: str) -> ComprehensiveAnalysisResults:
        """
        Run statistics for a single section and return updated results.

        section: 'individual_ba' | 'totals' | 'ratios' | 'percentages' | 'categories'
        """
        valid_data = self._get_valid_data()

        # Set up factor metadata once (idempotent)
        self._init_factor_metadata()

        if self.n_factors == 3 and len(self.factors) >= 3:
            self._run_threeway_section(valid_data, section)
        elif self.n_factors == 2 and len(self.factors) >= 2:
            self._run_twoway_section(valid_data, section)
        else:
            self._run_oneway_section(valid_data, section)

        return self._finalize_results()

    def _init_factor_metadata(self):
        """Set factor names/cols on results (idempotent)."""
        if self.n_factors >= 2 and len(self.factors) >= 2:
            factor_items = list(self.factors.items())
            fa_name, fa_col = factor_items[0]
            fb_name, fb_col = factor_items[1]
            if self.n_factors >= 3 and len(self.factors) >= 3:
                fc_name, fc_col = factor_items[2]
                self.results.is_threeway = True
                self.results.factor_c_name = fc_name
                self.results.factor_c_col = fc_col
            else:
                self.results.is_twoway = True
            self.results.factor_a_name = fa_name
            self.results.factor_b_name = fb_name
            self.results.factor_a_col = fa_col
            self.results.factor_b_col = fb_col

    # --- Log₁₀ transform helper ---

    @staticmethod
    def _log_transform_column(data: pd.DataFrame, col: str) -> pd.Series:
        """Log₁₀-transform a column, handling zeros/negatives with a floor value."""
        vals = pd.to_numeric(data[col], errors='coerce')
        min_positive = vals[vals > 0].min()
        floor_val = min_positive / 10 if pd.notna(min_positive) else 0.001
        return np.log10(vals.clip(lower=floor_val))

    def _prepare_log_data(self, data: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, str]:
        """Create a copy of data with a log₁₀-transformed column. Returns (data_copy, log_col_name)."""
        log_col = f'{col}_log10'
        data_copy = data.copy()
        data_copy[log_col] = self._log_transform_column(data_copy, col)
        return data_copy, log_col

    # --- ONE-WAY sections ---

    def _run_oneway_section(self, valid_data, section):
        if section == 'individual_ba':
            self._log(f"Running one-way ANOVA for {len(self.ba_cols)} individual bile acids...")
            for ba in self.ba_cols:
                if ba in valid_data.columns:
                    if self._should_exclude_for_lod(ba, valid_data):
                        continue
                    try:
                        result = self.analyzer.analyze(valid_data, ba, self.group_col)
                        self.results.individual_ba_results[ba] = result
                    except Exception as e:
                        print(f"Could not analyze {ba}: {e}")
                    try:
                        log_data, log_col = self._prepare_log_data(valid_data, ba)
                        log_result = self.analyzer.analyze(log_data, log_col, self.group_col)
                        log_result.variable_name = ba
                        self.results.log_individual_ba_results[ba] = log_result
                    except Exception as e:
                        print(f"Could not analyze {ba} (log₁₀): {e}")
            self._log(f"  Done: {len(self.results.individual_ba_results)} individual BAs analyzed")

        elif section == 'totals':
            if self.totals is not None:
                self._log(f"Running one-way ANOVA for {len(self.totals.columns)} totals...")
                combined = pd.concat([valid_data[[self.group_col]], self.totals.loc[valid_data.index]], axis=1)
                for col in self.totals.columns:
                    if not combined[col].isna().all():
                        if self._should_exclude_category_for_lod(col, valid_data):
                            continue
                        try:
                            result = self.analyzer.analyze(combined, col, self.group_col)
                            self.results.totals_results[col] = result
                        except Exception as e:
                            print(f"Could not analyze {col}: {e}")
                        try:
                            log_data, log_col = self._prepare_log_data(combined, col)
                            log_result = self.analyzer.analyze(log_data, log_col, self.group_col)
                            log_result.variable_name = col
                            self.results.log_totals_results[col] = log_result
                        except Exception as e:
                            print(f"Could not analyze {col} (log₁₀): {e}")
                self._log(f"  Done: {len(self.results.totals_results)} totals analyzed")

        elif section == 'ratios':
            if self.ratios is not None:
                self._log(f"Running one-way ANOVA for {len(self.ratios.columns)} ratios...")
                combined = pd.concat([valid_data[[self.group_col]], self.ratios.loc[valid_data.index]], axis=1)
                for col in self.ratios.columns:
                    if not combined[col].isna().all():
                        try:
                            result = self.analyzer.analyze(combined, col, self.group_col)
                            self.results.ratios_results[col] = result
                        except Exception as e:
                            print(f"Could not analyze {col}: {e}")
                        try:
                            log_data, log_col = self._prepare_log_data(combined, col)
                            log_result = self.analyzer.analyze(log_data, log_col, self.group_col)
                            log_result.variable_name = col
                            self.results.log_ratios_results[col] = log_result
                        except Exception as e:
                            print(f"Could not analyze {col} (log₁₀): {e}")
                self._log(f"  Done: {len(self.results.ratios_results)} ratios analyzed")

        elif section == 'percentages':
            if self.percentages is not None:
                self._log(f"Running one-way ANOVA for {len(self.percentages.columns)} percentages...")
                combined = pd.concat([valid_data[[self.group_col]], self.percentages.loc[valid_data.index]], axis=1)
                for col in self.percentages.columns:
                    if not combined[col].isna().all():
                        base_ba = col.replace('_pct', '') if col.endswith('_pct') else col
                        if base_ba in self.lod_excluded or self._should_exclude_for_lod(base_ba, valid_data):
                            continue
                        try:
                            result = self.analyzer.analyze(combined, col, self.group_col)
                            self.results.percentages_results[col] = result
                        except Exception as e:
                            print(f"Could not analyze {col}: {e}")
                self._log(f"  Done: {len(self.results.percentages_results)} percentages analyzed")

        elif section == 'categories':
            self._log("Running one-way ANOVA for category sheets...")
            self.generate_all_sheets()
            for cat_name, sheet in self.analysis_sheets.items():
                if sheet.statistical_result:
                    self.results.category_results[cat_name] = sheet.statistical_result
            self._log(f"  Done: {len(self.results.category_results)} categories analyzed")

    # --- TWO-WAY sections ---

    def _run_twoway_section(self, valid_data, section):
        factor_items = list(self.factors.items())
        fa_name, fa_col = factor_items[0]
        fb_name, fb_col = factor_items[1]

        if fa_col not in valid_data.columns or fb_col not in valid_data.columns:
            print(f"Factor columns not found in data: {fa_col}, {fb_col}")
            return

        if section == 'individual_ba':
            self._log(f"Running two-way ANOVA ({fa_name}×{fb_name}) for {len(self.ba_cols)} individual bile acids...")
            for i, ba in enumerate(self.ba_cols):
                if ba in valid_data.columns:
                    if self._should_exclude_for_lod(ba, valid_data):
                        continue
                    try:
                        result = self.analyzer.analyze_twoway(
                            valid_data, ba, fa_col, fb_col, fa_name, fb_name
                        )
                        self.results.twoway_individual_ba[ba] = result
                    except Exception as e:
                        print(f"Could not analyze {ba} (two-way): {e}")
                    try:
                        log_data, log_col = self._prepare_log_data(valid_data, ba)
                        log_result = self.analyzer.analyze_twoway(
                            log_data, log_col, fa_col, fb_col, fa_name, fb_name
                        )
                        log_result.variable_name = ba
                        self.results.log_twoway_individual_ba[ba] = log_result
                    except Exception as e:
                        print(f"Could not analyze {ba} (two-way log₁₀): {e}")
                    if (i + 1) % 20 == 0:
                        gc.collect()
            gc.collect()
            self._log(f"  Done: {len(self.results.twoway_individual_ba)} individual BAs analyzed")

        elif section == 'totals':
            if self.totals is not None:
                self._log(f"Running two-way ANOVA for {len(self.totals.columns)} totals...")
                combined = pd.concat([valid_data[[fa_col, fb_col]], self.totals.loc[valid_data.index]], axis=1)
                for col in self.totals.columns:
                    if not combined[col].isna().all():
                        if self._should_exclude_category_for_lod(col, valid_data):
                            continue
                        try:
                            result = self.analyzer.analyze_twoway(
                                combined, col, fa_col, fb_col, fa_name, fb_name
                            )
                            self.results.twoway_totals[col] = result
                        except Exception as e:
                            print(f"Could not analyze {col} (two-way): {e}")
                        try:
                            log_data, log_col = self._prepare_log_data(combined, col)
                            log_result = self.analyzer.analyze_twoway(
                                log_data, log_col, fa_col, fb_col, fa_name, fb_name
                            )
                            log_result.variable_name = col
                            self.results.log_twoway_totals[col] = log_result
                        except Exception as e:
                            print(f"Could not analyze {col} (two-way log₁₀): {e}")
                del combined
                gc.collect()
                self._log(f"  Done: {len(self.results.twoway_totals)} totals analyzed")

        elif section == 'ratios':
            if self.ratios is not None:
                self._log(f"Running two-way ANOVA for {len(self.ratios.columns)} ratios...")
                combined = pd.concat([valid_data[[fa_col, fb_col]], self.ratios.loc[valid_data.index]], axis=1)
                for col in self.ratios.columns:
                    if not combined[col].isna().all():
                        try:
                            result = self.analyzer.analyze_twoway(
                                combined, col, fa_col, fb_col, fa_name, fb_name
                            )
                            self.results.twoway_ratios[col] = result
                        except Exception as e:
                            print(f"Could not analyze {col} (two-way): {e}")
                        try:
                            log_data, log_col = self._prepare_log_data(combined, col)
                            log_result = self.analyzer.analyze_twoway(
                                log_data, log_col, fa_col, fb_col, fa_name, fb_name
                            )
                            log_result.variable_name = col
                            self.results.log_twoway_ratios[col] = log_result
                        except Exception as e:
                            print(f"Could not analyze {col} (two-way log₁₀): {e}")
                del combined
                gc.collect()
                self._log(f"  Done: {len(self.results.twoway_ratios)} ratios analyzed")

        elif section == 'percentages':
            if self.percentages is not None:
                self._log(f"Running two-way ANOVA for {len(self.percentages.columns)} percentages...")
                combined = pd.concat([valid_data[[fa_col, fb_col]], self.percentages.loc[valid_data.index]], axis=1)
                for col in self.percentages.columns:
                    if not combined[col].isna().all():
                        base_ba = col.replace('_pct', '') if col.endswith('_pct') else col
                        if base_ba in self.lod_excluded or self._should_exclude_for_lod(base_ba, valid_data):
                            continue
                        try:
                            result = self.analyzer.analyze_twoway(
                                combined, col, fa_col, fb_col, fa_name, fb_name
                            )
                            self.results.twoway_percentages[col] = result
                        except Exception as e:
                            print(f"Could not analyze {col} (two-way): {e}")
                del combined
                gc.collect()
                self._log(f"  Done: {len(self.results.twoway_percentages)} percentages analyzed")

        elif section == 'categories':
            self._log("Running two-way ANOVA for category sheets...")
            self.generate_all_sheets()
            for cat_name, sheet in self.analysis_sheets.items():
                if sheet.statistical_result:
                    self.results.category_results[cat_name] = sheet.statistical_result
            self._log(f"  Done: {len(self.results.category_results)} categories analyzed")

    # --- THREE-WAY sections ---

    def _run_threeway_section(self, valid_data, section):
        factor_items = list(self.factors.items())
        fa_name, fa_col = factor_items[0]
        fb_name, fb_col = factor_items[1]
        fc_name, fc_col = factor_items[2]

        if fa_col not in valid_data.columns or fb_col not in valid_data.columns or fc_col not in valid_data.columns:
            print(f"Factor columns not found in data: {fa_col}, {fb_col}, {fc_col}")
            return

        if section == 'individual_ba':
            self._log(f"Running three-way ANOVA ({fa_name}×{fb_name}×{fc_name}) for {len(self.ba_cols)} individual bile acids...")
            for i, ba in enumerate(self.ba_cols):
                if ba in valid_data.columns:
                    if self._should_exclude_for_lod(ba, valid_data):
                        continue
                    try:
                        result = self.analyzer.analyze_threeway(
                            valid_data, ba, fa_col, fb_col, fc_col, fa_name, fb_name, fc_name
                        )
                        self.results.threeway_individual_ba[ba] = result
                    except Exception as e:
                        print(f"Could not analyze {ba} (three-way): {e}")
                    try:
                        log_data, log_col = self._prepare_log_data(valid_data, ba)
                        log_result = self.analyzer.analyze_threeway(
                            log_data, log_col, fa_col, fb_col, fc_col, fa_name, fb_name, fc_name
                        )
                        log_result.variable_name = ba
                        self.results.log_threeway_individual_ba[ba] = log_result
                    except Exception as e:
                        print(f"Could not analyze {ba} (three-way log₁₀): {e}")
                    if (i + 1) % 20 == 0:
                        self._log(f"  Progress: {i + 1}/{len(self.ba_cols)} BAs...")
                        gc.collect()
            gc.collect()
            self._log(f"  Done: {len(self.results.threeway_individual_ba)} individual BAs analyzed")

        elif section == 'totals':
            if self.totals is not None:
                self._log(f"Running three-way ANOVA for {len(self.totals.columns)} totals...")
                combined = pd.concat([valid_data[[fa_col, fb_col, fc_col]], self.totals.loc[valid_data.index]], axis=1)
                for col in self.totals.columns:
                    if not combined[col].isna().all():
                        if self._should_exclude_category_for_lod(col, valid_data):
                            continue
                        try:
                            result = self.analyzer.analyze_threeway(
                                combined, col, fa_col, fb_col, fc_col, fa_name, fb_name, fc_name
                            )
                            self.results.threeway_totals[col] = result
                        except Exception as e:
                            print(f"Could not analyze {col} (three-way): {e}")
                        try:
                            log_data, log_col = self._prepare_log_data(combined, col)
                            log_result = self.analyzer.analyze_threeway(
                                log_data, log_col, fa_col, fb_col, fc_col, fa_name, fb_name, fc_name
                            )
                            log_result.variable_name = col
                            self.results.log_threeway_totals[col] = log_result
                        except Exception as e:
                            print(f"Could not analyze {col} (three-way log₁₀): {e}")
                del combined
                gc.collect()
                self._log(f"  Done: {len(self.results.threeway_totals)} totals analyzed")

        elif section == 'ratios':
            if self.ratios is not None:
                self._log(f"Running three-way ANOVA for {len(self.ratios.columns)} ratios...")
                combined = pd.concat([valid_data[[fa_col, fb_col, fc_col]], self.ratios.loc[valid_data.index]], axis=1)
                for col in self.ratios.columns:
                    if not combined[col].isna().all():
                        try:
                            result = self.analyzer.analyze_threeway(
                                combined, col, fa_col, fb_col, fc_col, fa_name, fb_name, fc_name
                            )
                            self.results.threeway_ratios[col] = result
                        except Exception as e:
                            print(f"Could not analyze {col} (three-way): {e}")
                        try:
                            log_data, log_col = self._prepare_log_data(combined, col)
                            log_result = self.analyzer.analyze_threeway(
                                log_data, log_col, fa_col, fb_col, fc_col, fa_name, fb_name, fc_name
                            )
                            log_result.variable_name = col
                            self.results.log_threeway_ratios[col] = log_result
                        except Exception as e:
                            print(f"Could not analyze {col} (three-way log₁₀): {e}")
                del combined
                gc.collect()
                self._log(f"  Done: {len(self.results.threeway_ratios)} ratios analyzed")

        elif section == 'percentages':
            if self.percentages is not None:
                self._log(f"Running three-way ANOVA for {len(self.percentages.columns)} percentages...")
                combined = pd.concat([valid_data[[fa_col, fb_col, fc_col]], self.percentages.loc[valid_data.index]], axis=1)
                for col in self.percentages.columns:
                    if not combined[col].isna().all():
                        base_ba = col.replace('_pct', '') if col.endswith('_pct') else col
                        if base_ba in self.lod_excluded or self._should_exclude_for_lod(base_ba, valid_data):
                            continue
                        try:
                            result = self.analyzer.analyze_threeway(
                                combined, col, fa_col, fb_col, fc_col, fa_name, fb_name, fc_name
                            )
                            self.results.threeway_percentages[col] = result
                        except Exception as e:
                            print(f"Could not analyze {col} (three-way): {e}")
                del combined
                gc.collect()
                self._log(f"  Done: {len(self.results.threeway_percentages)} percentages analyzed")

        elif section == 'categories':
            self._log("Running three-way ANOVA for category sheets...")
            self.generate_all_sheets()
            for cat_name, sheet in self.analysis_sheets.items():
                if sheet.statistical_result:
                    self.results.category_results[cat_name] = sheet.statistical_result
            self._log(f"  Done: {len(self.results.category_results)} categories analyzed")

    def run_all_statistics(self) -> ComprehensiveAnalysisResults:
        """Run all statistical analyses at once (legacy entry point)."""
        for section in ['individual_ba', 'totals', 'ratios', 'percentages', 'categories']:
            self.run_section(section)
        return self._finalize_results()

    # Legacy methods removed — replaced by run_section() and section runners above.

    def _calculate_sheet_data(
        self,
        category_name: str,
        ba_columns: List[str],
        description: str
    ) -> AnalysisSheet:
        """Calculate all data for a single analysis sheet."""

        # Get raw data for these bile acids
        id_cols = [self.group_col]
        if self.sample_id_col and self.sample_id_col in self.data.columns:
            id_cols = [self.sample_id_col] + id_cols

        # For two-way designs, include all factor columns
        if self.n_factors >= 2:
            for factor_name, factor_col in self.factors.items():
                if factor_col in self.data.columns and factor_col not in id_cols:
                    id_cols.append(factor_col)

        available_ba_cols = [c for c in ba_columns if c in self.data.columns]
        raw_data = self.data[id_cols + available_ba_cols].copy()

        # Calculate total for each sample
        raw_data['Total'] = raw_data[available_ba_cols].sum(axis=1)

        # Calculate total of ALL bile acids for percentage calculation
        all_ba_total = self.data[self.ba_cols].sum(axis=1)
        raw_data['Pct_of_Total_BA'] = (raw_data['Total'] / all_ba_total * 100).round(2)

        # Group summary statistics
        # In two-way mode, create factorial group for summary
        if self.n_factors >= 2 and len(self.factors) >= 2:
            factor_cols = [fc for fc in self.factors.values() if fc in raw_data.columns]
            factor_names = list(self.factors.keys())
            if len(factor_cols) >= 3:
                raw_data['_factorial_group_'] = (raw_data[factor_cols[0]].astype(str) + ' / ' +
                                                 raw_data[factor_cols[1]].astype(str) + ' / ' +
                                                 raw_data[factor_cols[2]].astype(str))
                summary_group_col = '_factorial_group_'
                summary_group_label = f'{factor_names[0]} / {factor_names[1]} / {factor_names[2]}'
            elif len(factor_cols) >= 2:
                raw_data['_factorial_group_'] = raw_data[factor_cols[0]].astype(str) + ' / ' + raw_data[factor_cols[1]].astype(str)
                summary_group_col = '_factorial_group_'
                summary_group_label = f'{factor_names[0]} / {factor_names[1]}'
            else:
                summary_group_col = self.group_col
                summary_group_label = self.group_col
        else:
            summary_group_col = self.group_col
            summary_group_label = self.group_col

        group_totals = raw_data.groupby(summary_group_col).agg({
            'Total': ['count', 'mean', 'std', 'sem', 'median', 'min', 'max'],
            'Pct_of_Total_BA': ['mean', 'std', 'sem']
        }).round(4)

        # Rename the index to use readable label
        group_totals.index.name = summary_group_label

        # Flatten column names
        group_totals.columns = ['_'.join(col).strip() for col in group_totals.columns]
        group_totals = group_totals.rename(columns={'Total_count': 'n'})

        # Calculate 95% CI
        group_totals['Total_CI95_lower'] = group_totals['Total_mean'] - 1.96 * group_totals['Total_sem']
        group_totals['Total_CI95_upper'] = group_totals['Total_mean'] + 1.96 * group_totals['Total_sem']

        # Individual BA percentages within this category
        group_percentages = pd.DataFrame()
        if len(available_ba_cols) > 1:
            for ba in available_ba_cols:
                raw_data[f'{ba}_pct'] = (raw_data[ba] / raw_data['Total'] * 100).round(2)

            pct_cols = [f'{ba}_pct' for ba in available_ba_cols]
            group_percentages = raw_data.groupby(summary_group_col)[pct_cols].mean().round(2)
            group_percentages.index.name = summary_group_label

        # Statistical analysis
        stat_result = None
        if self.n_factors >= 3 and len(self.factors) >= 3:
            # Three-way mode: run three-way ANOVA on the category total
            factor_cols_list = list(self.factors.values())
            factor_names_list = list(self.factors.keys())
            fa_col, fb_col, fc_col = factor_cols_list[0], factor_cols_list[1], factor_cols_list[2]
            fa_name, fb_name, fc_name = factor_names_list[0], factor_names_list[1], factor_names_list[2]
            if fa_col in raw_data.columns and fb_col in raw_data.columns and fc_col in raw_data.columns:
                try:
                    stat_result = self.analyzer.analyze_threeway(
                        raw_data, 'Total', fa_col, fb_col, fc_col, fa_name, fb_name, fc_name
                    )
                except Exception as e:
                    print(f"Could not run three-way ANOVA for {category_name}: {e}")
        elif self.n_factors >= 2 and len(self.factors) >= 2:
            # Two-way mode: run two-way ANOVA on the category total
            factor_cols_list = list(self.factors.values())
            fa_col = factor_cols_list[0]
            fb_col = factor_cols_list[1]
            fa_name = list(self.factors.keys())[0]
            fb_name = list(self.factors.keys())[1]
            if fa_col in raw_data.columns and fb_col in raw_data.columns:
                try:
                    stat_result = self.analyzer.analyze_twoway(
                        raw_data, 'Total', fa_col, fb_col, fa_name, fb_name
                    )
                except Exception as e:
                    print(f"Could not run two-way ANOVA for {category_name}: {e}")
        elif len(self.data[self.group_col].unique()) >= 2:
            try:
                stat_result = self.analyzer.analyze(raw_data, 'Total', self.group_col)
            except Exception as e:
                print(f"Could not analyze {category_name}: {e}")

        # Drop internal columns before storing
        export_raw_data = raw_data.drop(columns=['_factorial_group_'], errors='ignore')

        return AnalysisSheet(
            name=category_name,
            description=description,
            bile_acid_columns=available_ba_cols,
            raw_data=export_raw_data,
            group_totals=group_totals,
            group_percentages=group_percentages,
            statistical_result=stat_result
        )
    
    def generate_all_sheets(self) -> Dict[str, AnalysisSheet]:
        """Generate analysis data for all categories."""
        
        for category_name, category_info in self.ANALYSIS_CATEGORIES.items():
            ba_cols = category_info['get_columns'](self.ba_cols)
            
            if ba_cols:  # Only create sheet if we have data
                sheet = self._calculate_sheet_data(
                    category_name,
                    ba_cols,
                    category_info['description']
                )
                self.analysis_sheets[category_name] = sheet
        
        return self.analysis_sheets
    
    def _write_sheet(
        self,
        writer: pd.ExcelWriter,
        sheet: AnalysisSheet
    ):
        """Write a single analysis sheet to Excel."""
        
        sheet_name = sheet.name[:31]  # Excel sheet name limit
        
        # Track current row for writing
        current_row = 0
        
        # 1. Header and description
        header_df = pd.DataFrame({
            'Analysis': [sheet.name],
            'Description': [sheet.description],
            'Bile Acids Included': [', '.join(sheet.bile_acid_columns)],
            'N Bile Acids': [len(sheet.bile_acid_columns)]
        })
        header_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
        current_row += 3
        
        # 2. Raw data section
        section_header = pd.DataFrame({'': ['RAW DATA (Concentrations)']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row, 
                               index=False, header=False)
        current_row += 1
        
        sheet.raw_data.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
        current_row += len(sheet.raw_data) + 3
        
        # 3. Group summary statistics
        section_header = pd.DataFrame({'': ['GROUP SUMMARY STATISTICS']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 1
        
        sheet.group_totals.to_excel(writer, sheet_name=sheet_name, startrow=current_row)
        current_row += len(sheet.group_totals) + 3
        
        # 4. Percentage breakdown (if applicable)
        if not sheet.group_percentages.empty:
            section_header = pd.DataFrame({'': ['PERCENTAGE BREAKDOWN (% of category total)']})
            section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                   index=False, header=False)
            current_row += 1
            
            sheet.group_percentages.to_excel(writer, sheet_name=sheet_name, startrow=current_row)
            current_row += len(sheet.group_percentages) + 3
        
        # 5. Statistical results
        if sheet.statistical_result:
            result = sheet.statistical_result

            section_header = pd.DataFrame({'': ['STATISTICAL ANALYSIS']})
            section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                   index=False, header=False)
            current_row += 1

            # Check if this is a three-way, two-way, or one-way result
            if isinstance(result, FullThreeWayAnalysisResult):
                # ============================================================
                # THREE-WAY ANOVA RESULTS
                # ============================================================
                tw = result.threeway_result
                fa_name = self.results.factor_a_name or 'Factor A'
                fb_name = self.results.factor_b_name or 'Factor B'
                fc_name = self.results.factor_c_name or 'Factor C'

                test_label = pd.DataFrame({'': [f'Test: {tw.test_type.value} ({fa_name} \u00d7 {fb_name} \u00d7 {fc_name})']})
                test_label.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                   index=False, header=False)
                current_row += 1

                apa_text = format_threeway_apa(result)
                apa_df = pd.DataFrame({'APA Format': [apa_text]})
                apa_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                current_row += 2

                if tw.anova_table is not None and not tw.anova_table.empty:
                    anova_header = pd.DataFrame({'': ['ANOVA Table']})
                    anova_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                         index=False, header=False)
                    current_row += 1
                    tw.anova_table.to_excel(writer, sheet_name=sheet_name,
                                            startrow=current_row, index=False)
                    current_row += len(tw.anova_table) + 1

                if result.descriptive_stats is not None and not result.descriptive_stats.empty:
                    desc = result.descriptive_stats.copy()
                    desc = desc[(desc['factor_a'] != '__MARGINAL__') &
                                (desc['factor_b'] != '__MARGINAL__') &
                                (desc['factor_c'] != '__MARGINAL__')]
                    if not desc.empty:
                        desc_header = pd.DataFrame({'': ['Cell Descriptive Statistics']})
                        desc_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                            index=False, header=False)
                        current_row += 1
                        desc.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                        current_row += len(desc) + 1

                if tw.posthoc_results is not None and not tw.posthoc_results.empty:
                    ph_header = pd.DataFrame({'': [f'Post-hoc: {tw.posthoc_type}']})
                    ph_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                      index=False, header=False)
                    current_row += 1
                    tw.posthoc_results.to_excel(writer, sheet_name=sheet_name,
                                                startrow=current_row, index=False)

            elif isinstance(result, FullTwoWayAnalysisResult):
                # ============================================================
                # TWO-WAY ANOVA RESULTS
                # ============================================================
                tw = result.twoway_result
                fa_name = self.results.factor_a_name if self.results.factor_a_name else 'Factor A'
                fb_name = self.results.factor_b_name if self.results.factor_b_name else 'Factor B'

                # Test type
                test_label = pd.DataFrame({'': [f'Test: {tw.test_type.value} ({fa_name} × {fb_name})']})
                test_label.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                   index=False, header=False)
                current_row += 1

                # APA formatted result
                apa_text = format_twoway_apa(result)
                apa_df = pd.DataFrame({'APA Format': [apa_text]})
                apa_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                current_row += 2

                # ANOVA table
                if tw.anova_table is not None and not tw.anova_table.empty:
                    anova_header = pd.DataFrame({'': ['ANOVA Table']})
                    anova_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                         index=False, header=False)
                    current_row += 1
                    tw.anova_table.to_excel(writer, sheet_name=sheet_name,
                                            startrow=current_row, index=False)
                    current_row += len(tw.anova_table) + 1
                else:
                    # Build manual ANOVA summary
                    anova_rows = []
                    for source, stat, pval, es_key in [
                        (fa_name, tw.factor_a_stat, tw.factor_a_pvalue, fa_name),
                        (fb_name, tw.factor_b_stat, tw.factor_b_pvalue, fb_name),
                        (f'{fa_name}×{fb_name}', tw.interaction_stat, tw.interaction_pvalue,
                         f'{fa_name}×{fb_name}'),
                    ]:
                        sig = 'Yes' if (not np.isnan(pval) and pval < tw.alpha) else ('No' if not np.isnan(pval) else 'N/A')
                        anova_rows.append({
                            'Source': source,
                            'F': f'{stat:.4f}' if not np.isnan(stat) else 'N/A',
                            'p': f'{pval:.6f}' if not np.isnan(pval) else 'N/A',
                            'partial_η²': f'{tw.effect_sizes.get(es_key, np.nan):.4f}'
                                if not np.isnan(tw.effect_sizes.get(es_key, np.nan)) else 'N/A',
                            'Significant': sig
                        })
                    anova_df = pd.DataFrame(anova_rows)
                    anova_df.to_excel(writer, sheet_name=sheet_name,
                                      startrow=current_row, index=False)
                    current_row += len(anova_df) + 1

                # Cell descriptive stats
                if result.descriptive_stats is not None and not result.descriptive_stats.empty:
                    desc = result.descriptive_stats.copy()
                    desc = desc[(desc['factor_a'] != '__MARGINAL__') & (desc['factor_b'] != '__MARGINAL__')]
                    if not desc.empty:
                        desc_header = pd.DataFrame({'': ['Cell Descriptive Statistics']})
                        desc_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                            index=False, header=False)
                        current_row += 1
                        desc.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                        current_row += len(desc) + 1

                # Post-hoc results
                if tw.posthoc_results is not None and not tw.posthoc_results.empty:
                    ph_header = pd.DataFrame({'': [f'Post-hoc: {tw.posthoc_type}']})
                    ph_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                      index=False, header=False)
                    current_row += 1
                    tw.posthoc_results.to_excel(writer, sheet_name=sheet_name,
                                                startrow=current_row, index=False)
            else:
                # ============================================================
                # ONE-WAY ANALYSIS RESULTS (unchanged)
                # ============================================================

                # APA formatted result
                apa_df = pd.DataFrame({'APA Format': [format_apa_statistics(result)]})
                apa_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                current_row += 2

                # Assumption tests
                assumptions_data = {
                    'Test': ['Normality Test', 'Homoscedasticity Test'],
                    'Result': [
                        'Passed' if result.assumptions.overall_normality else 'Failed',
                        'Passed' if result.assumptions.homoscedasticity_passed else 'Failed'
                    ],
                    'P-value': [
                        ', '.join([f"{g}: {p:.4f}" for g, p in result.assumptions.normality_pvalues.items()]),
                        f"{result.assumptions.homoscedasticity_pvalue:.4f}"
                    ]
                }
                assumptions_df = pd.DataFrame(assumptions_data)
                assumptions_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                current_row += 4

                # Main test result
                main_test_data = {
                    'Test Used': [result.main_test.test_type.value],
                    'Reason': [result.assumptions.recommendation_reason],
                    'Statistic': [f"{result.main_test.statistic:.4f}"],
                    'P-value': [f"{result.main_test.pvalue:.6f}"],
                    'Significant': ['Yes' if result.main_test.significant else 'No'],
                    'Effect Size': [f"{result.main_test.effect_size:.4f}" if result.main_test.effect_size else 'N/A'],
                    'Effect Type': [result.main_test.effect_size_type or 'N/A'],
                    'Interpretation': [result.main_test.effect_size_interpretation or 'N/A']
                }
                main_test_df = pd.DataFrame(main_test_data)
                main_test_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                current_row += 4

                # Post-hoc results
                if result.posthoc_test and result.posthoc_test.pairwise_results is not None:
                    section_header = pd.DataFrame({'': [f'POST-HOC COMPARISONS ({result.posthoc_test.test_type.value})']})
                    section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                           index=False, header=False)
                    current_row += 1

                    posthoc_df = result.posthoc_test.pairwise_results.copy()
                    posthoc_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
    
    def save_excel_report(self, filepath) -> Path:
        """Save complete Excel report with all analysis sheets."""

        if not self.analysis_sheets:
            self.generate_all_sheets()

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            if self.n_factors >= 3 and self.results.is_threeway:
                # ============================================================
                # THREE-WAY ANOVA REPORT
                # ============================================================
                self._write_threeway_overview_sheet(writer)

                if self.results.threeway_individual_ba:
                    self._write_threeway_results_sheet(
                        writer, 'Individual_BA',
                        'Individual Bile Acid Three-Way ANOVA',
                        self.results.threeway_individual_ba
                    )
                if self.results.log_threeway_individual_ba:
                    self._write_threeway_results_sheet(
                        writer, 'Individual_BA_log10',
                        'Individual Bile Acid Three-Way ANOVA (log10)',
                        self.results.log_threeway_individual_ba
                    )
                if self.results.threeway_totals:
                    self._write_threeway_results_sheet(
                        writer, 'Totals',
                        'Total Bile Acid Categories Three-Way ANOVA',
                        self.results.threeway_totals
                    )
                if self.results.log_threeway_totals:
                    self._write_threeway_results_sheet(
                        writer, 'Totals_log10',
                        'Total Bile Acid Categories Three-Way ANOVA (log10)',
                        self.results.log_threeway_totals
                    )
                if self.results.threeway_ratios:
                    self._write_threeway_results_sheet(
                        writer, 'Ratios',
                        'Ratios Three-Way ANOVA',
                        self.results.threeway_ratios
                    )
                if self.results.log_threeway_ratios:
                    self._write_threeway_results_sheet(
                        writer, 'Ratios_log10',
                        'Ratios Three-Way ANOVA (log10)',
                        self.results.log_threeway_ratios
                    )
                if self.results.threeway_percentages:
                    self._write_threeway_results_sheet(
                        writer, 'Percentages',
                        'Bile Acid Percentages Three-Way ANOVA',
                        self.results.threeway_percentages
                    )

                for sheet_name, sheet in self.analysis_sheets.items():
                    self._write_sheet(writer, sheet)

            elif self.n_factors >= 2 and self.results.is_twoway:
                # ============================================================
                # TWO-WAY ANOVA REPORT
                # ============================================================
                self._write_twoway_overview_sheet(writer)

                # Individual bile acids
                if self.results.twoway_individual_ba:
                    self._write_twoway_results_sheet(
                        writer, 'Individual_BA',
                        'Individual Bile Acid Two-Way ANOVA',
                        self.results.twoway_individual_ba
                    )
                if self.results.log_twoway_individual_ba:
                    self._write_twoway_results_sheet(
                        writer, 'Individual_BA_log10',
                        'Individual Bile Acid Two-Way ANOVA (log10)',
                        self.results.log_twoway_individual_ba
                    )

                # Totals
                if self.results.twoway_totals:
                    self._write_twoway_results_sheet(
                        writer, 'Totals',
                        'Total Bile Acid Categories Two-Way ANOVA',
                        self.results.twoway_totals
                    )
                if self.results.log_twoway_totals:
                    self._write_twoway_results_sheet(
                        writer, 'Totals_log10',
                        'Total Bile Acid Categories Two-Way ANOVA (log10)',
                        self.results.log_twoway_totals
                    )

                # Ratios
                if self.results.twoway_ratios:
                    self._write_twoway_results_sheet(
                        writer, 'Ratios',
                        'Ratios Two-Way ANOVA',
                        self.results.twoway_ratios
                    )
                if self.results.log_twoway_ratios:
                    self._write_twoway_results_sheet(
                        writer, 'Ratios_log10',
                        'Ratios Two-Way ANOVA (log10)',
                        self.results.log_twoway_ratios
                    )

                # Percentages
                if self.results.twoway_percentages:
                    self._write_twoway_results_sheet(
                        writer, 'Percentages',
                        'Bile Acid Percentages Two-Way ANOVA',
                        self.results.twoway_percentages
                    )

                # Also include category sheets for reference
                for sheet_name, sheet in self.analysis_sheets.items():
                    self._write_sheet(writer, sheet)
            else:
                # ============================================================
                # ONE-WAY REPORT
                # ============================================================
                self._write_overview_sheet(writer)
                for sheet_name, sheet in self.analysis_sheets.items():
                    self._write_sheet(writer, sheet)
                # Individual bile acid concentrations tab
                if self.results.individual_ba_results:
                    self._write_concentrations_sheet(writer)
                if self.results.log_individual_ba_results:
                    self._write_concentrations_sheet(
                        writer, 'Concentrations_log10',
                        'INDIVIDUAL BILE ACID CONCENTRATION COMPARISONS (log₁₀)',
                        self.results.log_individual_ba_results
                    )
                if self.results.log_totals_results:
                    self._write_concentrations_sheet(
                        writer, 'Totals_log10',
                        'BILE ACID TOTALS COMPARISONS (log₁₀)',
                        self.results.log_totals_results
                    )
                if self.results.log_ratios_results:
                    self._write_concentrations_sheet(
                        writer, 'Ratios_log10',
                        'BILE ACID RATIO COMPARISONS (log₁₀)',
                        self.results.log_ratios_results
                    )

        return filepath
    
    def _write_overview_sheet(self, writer: pd.ExcelWriter):
        """Write overview/summary sheet."""
        
        current_row = 0
        
        # Title
        title_df = pd.DataFrame({'': ['BILE ACID ANALYSIS REPORT']})
        title_df.to_excel(writer, sheet_name='Overview', startrow=current_row, 
                         index=False, header=False)
        current_row += 2
        
        # Data summary
        _excl_ba = {k: v for k, v in self.lod_excluded.items() if k in self.ba_cols}
        _excl_cat = {k: v for k, v in self.lod_excluded.items() if k not in self.ba_cols}
        lod_params = ['LOD Exclusion Threshold', 'Analytes Excluded (LOD)', 'Categories Excluded (LOD)'] if self.lod_threshold > 0 else []
        lod_values = [
            f'{self.lod_threshold}% replacement',
            f'{len(_excl_ba)} of {len(self.ba_cols)} ({len(_excl_ba)/len(self.ba_cols)*100:.1f}%)' if self.ba_cols else '0',
            f'{len(_excl_cat)}' if _excl_cat else 'None'
        ] if self.lod_threshold > 0 else []
        summary_data = {
            'Parameter': ['Total Samples', 'Groups', 'Bile Acids Measured', 'Significance Level',
                          'Data Transformation'] + lod_params,
            'Value': [
                len(self.data),
                ', '.join(self.data[self.group_col].unique().astype(str)),
                len(self.ba_cols),
                f"α = {self.alpha}",
                'Raw + log₁₀ (concentrations, totals, ratios); raw only (percentages)'
            ] + lod_values
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Overview', startrow=current_row, index=False)
        current_row += len(summary_df) + 2

        # LOD excluded analytes detail table
        if self.lod_excluded:
            lod_header = pd.DataFrame({'': ['LOD-EXCLUDED ANALYTES']})
            lod_header.to_excel(writer, sheet_name='Overview', startrow=current_row,
                               index=False, header=False)
            current_row += 1
            lod_detail_rows = [{'Analyte': col, 'LOD %': f'{pct}%', 'Type': 'Category' if col in ANALYSIS_GROUPS or col == 'total_all' else 'Individual'}
                               for col, pct in sorted(self.lod_excluded.items())]
            lod_detail_df = pd.DataFrame(lod_detail_rows)
            lod_detail_df.to_excel(writer, sheet_name='Overview', startrow=current_row, index=False)
            current_row += len(lod_detail_df) + 2

        # Results summary table
        section_header = pd.DataFrame({'': ['RESULTS SUMMARY']})
        section_header.to_excel(writer, sheet_name='Overview', startrow=current_row,
                               index=False, header=False)
        current_row += 1
        
        results_rows = []
        for sheet_name, sheet in self.analysis_sheets.items():
            if sheet.statistical_result:
                result = sheet.statistical_result
                
                # Get significant comparisons
                sig_comparisons = []
                if result.posthoc_test and result.posthoc_test.pairwise_results is not None:
                    sig_pairs = result.posthoc_test.pairwise_results[
                        result.posthoc_test.pairwise_results['significant'] == True
                    ]
                    for _, row in sig_pairs.iterrows():
                        sig_comparisons.append(f"{row['group1']} vs {row['group2']}")
                
                results_rows.append({
                    'Analysis': sheet_name,
                    'Test': result.main_test.test_type.value,
                    'P-value': f"{result.main_test.pvalue:.6f}",
                    'Significant': 'Yes' if result.main_test.significant else 'No',
                    'Effect Size': f"{result.main_test.effect_size:.3f}" if result.main_test.effect_size else 'N/A',
                    'Significant Comparisons': '; '.join(sig_comparisons) if sig_comparisons else 'None'
                })
        
        if results_rows:
            results_df = pd.DataFrame(results_rows)
            results_df.to_excel(writer, sheet_name='Overview', startrow=current_row, index=False)

    def _write_twoway_overview_sheet(self, writer: pd.ExcelWriter):
        """Write overview sheet for two-way ANOVA report."""
        current_row = 0
        sheet_name = 'Overview'

        # Title
        title_df = pd.DataFrame({'': ['BILE ACID ANALYSIS REPORT — TWO-WAY ANOVA']})
        title_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                         index=False, header=False)
        current_row += 2

        # Design info
        fa_name = self.results.factor_a_name
        fb_name = self.results.factor_b_name
        fa_col = self.results.factor_a_col
        fb_col = self.results.factor_b_col

        valid_data = self.data[self.data[self.group_col].notna()].copy()
        valid_data = valid_data[valid_data[self.group_col].astype(str).str.lower() != 'nan']

        fa_levels = sorted(valid_data[fa_col].unique().astype(str))
        fb_levels = sorted(valid_data[fb_col].unique().astype(str))

        # Cell sizes
        cell_sizes = []
        for a in fa_levels:
            for b in fb_levels:
                n = len(valid_data[(valid_data[fa_col].astype(str) == a) & (valid_data[fb_col].astype(str) == b)])
                cell_sizes.append(f"{a}-{b}: n={n}")

        _excl_ba_tw = {k: v for k, v in self.lod_excluded.items() if k in self.ba_cols}
        _excl_cat_tw = {k: v for k, v in self.lod_excluded.items() if k not in self.ba_cols}
        lod_params_tw = ['LOD Exclusion Threshold', 'Analytes Excluded (LOD)', 'Categories Excluded (LOD)'] if self.lod_threshold > 0 else []
        lod_values_tw = [
            f'{self.lod_threshold}% replacement',
            f'{len(_excl_ba_tw)} of {len(self.ba_cols)} ({len(_excl_ba_tw)/len(self.ba_cols)*100:.1f}%)' if self.ba_cols else '0',
            f'{len(_excl_cat_tw)}' if _excl_cat_tw else 'None'
        ] if self.lod_threshold > 0 else []
        summary_data = {
            'Parameter': [
                'Total Samples', 'Experimental Design',
                f'Factor A: {fa_name}', f'Factor B: {fb_name}',
                'Cell Sizes', 'Bile Acids Measured',
                'Significance Level', 'Non-parametric Method',
                'Data Transformation'
            ] + lod_params_tw,
            'Value': [
                len(valid_data), f'{fa_name} x {fb_name} factorial',
                ', '.join(fa_levels), ', '.join(fb_levels),
                '; '.join(cell_sizes), len(self.ba_cols),
                f'α = {self.alpha}', 'ART ANOVA (when assumptions violated)',
                'Raw + log₁₀ (concentrations, totals, ratios); raw only (percentages)'
            ] + lod_values_tw
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
        current_row += len(summary_df) + 3

        # LOD excluded analytes detail table
        if self.lod_excluded:
            lod_header = pd.DataFrame({'': ['LOD-EXCLUDED ANALYTES']})
            lod_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
            current_row += 1
            lod_detail_rows = [{'Analyte': col, 'LOD %': f'{pct}%', 'Type': 'Category' if col in ANALYSIS_GROUPS or col == 'total_all' else 'Individual'}
                               for col, pct in sorted(self.lod_excluded.items())]
            lod_detail_df = pd.DataFrame(lod_detail_rows)
            lod_detail_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += len(lod_detail_df) + 3

        # SUMMARY TABLE: Individual Bile Acids
        section_header = pd.DataFrame({'': ['INDIVIDUAL BILE ACID RESULTS SUMMARY']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 1

        if self.results.twoway_individual_ba:
            summary = get_twoway_differences_summary(self.results.twoway_individual_ba)
            summary.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += len(summary) + 3

        # SUMMARY TABLE: Totals
        section_header = pd.DataFrame({'': ['TOTAL CATEGORIES RESULTS SUMMARY']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 1

        if self.results.twoway_totals:
            summary = get_twoway_differences_summary(self.results.twoway_totals)
            summary.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += len(summary) + 3

        # SUMMARY TABLE: Ratios
        section_header = pd.DataFrame({'': ['RATIOS RESULTS SUMMARY']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 1

        if self.results.twoway_ratios:
            summary = get_twoway_differences_summary(self.results.twoway_ratios)
            summary.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += len(summary) + 3

        # SUMMARY TABLE: Percentages
        section_header = pd.DataFrame({'': ['PERCENTAGE COMPOSITION RESULTS SUMMARY']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 1

        if self.results.twoway_percentages:
            summary = get_twoway_differences_summary(self.results.twoway_percentages)
            summary.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += len(summary) + 3

        # Auto-adjust column widths
        ws = writer.sheets[sheet_name]
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            ws.column_dimensions[column_letter].width = min(max_length + 2, 60)

    def _write_twoway_results_sheet(
        self,
        writer: pd.ExcelWriter,
        sheet_name: str,
        title: str,
        tw_results: Dict[str, FullTwoWayAnalysisResult]
    ):
        """Write a two-way ANOVA results sheet for a category of variables."""

        sheet_name = sheet_name[:31]  # Excel limit
        current_row = 0

        fa_name = self.results.factor_a_name
        fb_name = self.results.factor_b_name

        # Title
        title_df = pd.DataFrame({'': [title]})
        title_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                         index=False, header=False)
        current_row += 1
        design_df = pd.DataFrame({'': [f'Design: {fa_name} x {fb_name}']})
        design_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                         index=False, header=False)
        current_row += 2

        # SECTION 1: Summary table (all variables at once)
        section_header = pd.DataFrame({'': ['OMNIBUS RESULTS SUMMARY']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 1

        summary = get_twoway_differences_summary(tw_results)
        summary.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
        current_row += len(summary) + 3

        # SECTION 2: ANOVA tables for each variable
        section_header = pd.DataFrame({'': ['DETAILED ANOVA TABLES']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 2

        for var_name, result in tw_results.items():
            tw = result.twoway_result

            # Variable header
            var_header = pd.DataFrame({'': [f'--- {var_name} ---']})
            var_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
            current_row += 1

            # Test type
            test_label = pd.DataFrame({'': [f'Test: {tw.test_type.value}']})
            test_label.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
            current_row += 1

            # ANOVA table
            if tw.anova_table is not None and not tw.anova_table.empty:
                tw.anova_table.to_excel(writer, sheet_name=sheet_name,
                                        startrow=current_row, index=False)
                current_row += len(tw.anova_table) + 1
            else:
                # Build ANOVA table manually from results
                anova_rows = []
                for source, stat, pval, df_tup, es_key in [
                    (fa_name, tw.factor_a_stat, tw.factor_a_pvalue, tw.factor_a_df, fa_name),
                    (fb_name, tw.factor_b_stat, tw.factor_b_pvalue, tw.factor_b_df, fb_name),
                    (f'{fa_name}×{fb_name}', tw.interaction_stat, tw.interaction_pvalue,
                     tw.interaction_df, f'{fa_name}×{fb_name}'),
                ]:
                    anova_rows.append({
                        'Source': source,
                        'F': f'{stat:.4f}' if not np.isnan(stat) else 'N/A',
                        'df_num': df_tup[0] if len(df_tup) > 0 else 'N/A',
                        'df_den': df_tup[1] if len(df_tup) > 1 else 'N/A',
                        'p': f'{pval:.6f}' if not np.isnan(pval) else 'N/A',
                        'partial_η²': f'{tw.effect_sizes.get(es_key, np.nan):.4f}'
                            if not np.isnan(tw.effect_sizes.get(es_key, np.nan)) else 'N/A',
                        'Significant': 'Yes' if pval < tw.alpha else 'No'
                            if not np.isnan(pval) else 'N/A'
                    })
                anova_df = pd.DataFrame(anova_rows)
                anova_df.to_excel(writer, sheet_name=sheet_name,
                                  startrow=current_row, index=False)
                current_row += len(anova_df) + 1

            # APA formatted result
            apa_text = format_twoway_apa(result)
            apa_df = pd.DataFrame({'APA Format': [apa_text]})
            apa_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += 2

            # Cell descriptive stats
            if result.descriptive_stats is not None and not result.descriptive_stats.empty:
                desc = result.descriptive_stats.copy()
                # Filter out marginal rows for clean display
                desc = desc[desc['factor_a'] != '__MARGINAL__']
                desc = desc[desc['factor_b'] != '__MARGINAL__']
                if not desc.empty:
                    desc_header = pd.DataFrame({'': ['Cell Descriptive Statistics']})
                    desc_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                        index=False, header=False)
                    current_row += 1
                    desc.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    current_row += len(desc) + 1

            # Post-hoc results
            if tw.posthoc_results is not None and not tw.posthoc_results.empty:
                ph_header = pd.DataFrame({'': [f'Post-hoc: {tw.posthoc_type}']})
                ph_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                  index=False, header=False)
                current_row += 1
                tw.posthoc_results.to_excel(writer, sheet_name=sheet_name,
                                            startrow=current_row, index=False)
                current_row += len(tw.posthoc_results) + 1

            current_row += 1  # Extra spacing between variables

        # Auto-adjust column widths
        ws = writer.sheets[sheet_name]
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            ws.column_dimensions[column_letter].width = min(max_length + 2, 50)

    def _write_threeway_overview_sheet(self, writer: pd.ExcelWriter):
        """Write overview sheet for three-way ANOVA report."""
        current_row = 0
        sheet_name = 'Overview'

        title_df = pd.DataFrame({'': ['BILE ACID ANALYSIS REPORT \u2014 THREE-WAY ANOVA']})
        title_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                         index=False, header=False)
        current_row += 2

        fa_name = self.results.factor_a_name
        fb_name = self.results.factor_b_name
        fc_name = self.results.factor_c_name
        fa_col = self.results.factor_a_col
        fb_col = self.results.factor_b_col
        fc_col = self.results.factor_c_col

        valid_data = self.data[self.data[self.group_col].notna()].copy()
        valid_data = valid_data[valid_data[self.group_col].astype(str).str.lower() != 'nan']

        fa_levels = sorted(valid_data[fa_col].unique().astype(str))
        fb_levels = sorted(valid_data[fb_col].unique().astype(str))
        fc_levels = sorted(valid_data[fc_col].unique().astype(str))

        _excl_ba_3w = {k: v for k, v in self.lod_excluded.items() if k in self.ba_cols}
        _excl_cat_3w = {k: v for k, v in self.lod_excluded.items() if k not in self.ba_cols}
        lod_params_3w = ['LOD Exclusion Threshold', 'Analytes Excluded (LOD)', 'Categories Excluded (LOD)'] if self.lod_threshold > 0 else []
        lod_values_3w = [
            f'{self.lod_threshold}% replacement',
            f'{len(_excl_ba_3w)} of {len(self.ba_cols)} ({len(_excl_ba_3w)/len(self.ba_cols)*100:.1f}%)' if self.ba_cols else '0',
            f'{len(_excl_cat_3w)}' if _excl_cat_3w else 'None'
        ] if self.lod_threshold > 0 else []
        summary_data = {
            'Parameter': [
                'Total Samples', 'Experimental Design',
                f'Factor A: {fa_name}', f'Factor B: {fb_name}', f'Factor C: {fc_name}',
                'Bile Acids Measured', 'Significance Level', 'Non-parametric Method',
                'Data Transformation'
            ] + lod_params_3w,
            'Value': [
                len(valid_data), f'{fa_name} x {fb_name} x {fc_name} factorial',
                ', '.join(fa_levels), ', '.join(fb_levels), ', '.join(fc_levels),
                len(self.ba_cols), f'α = {self.alpha}', 'ART ANOVA (when assumptions violated)',
                'Raw + log₁₀ (concentrations, totals, ratios); raw only (percentages)'
            ] + lod_values_3w
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
        current_row += len(summary_df) + 3

        # LOD excluded analytes detail table
        if self.lod_excluded:
            lod_header = pd.DataFrame({'': ['LOD-EXCLUDED ANALYTES']})
            lod_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
            current_row += 1
            lod_detail_rows = [{'Analyte': col, 'LOD %': f'{pct}%', 'Type': 'Category' if col in ANALYSIS_GROUPS or col == 'total_all' else 'Individual'}
                               for col, pct in sorted(self.lod_excluded.items())]
            lod_detail_df = pd.DataFrame(lod_detail_rows)
            lod_detail_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += len(lod_detail_df) + 3

        for section_title, results_dict in [
            ('INDIVIDUAL BILE ACID RESULTS SUMMARY', self.results.threeway_individual_ba),
            ('TOTAL CATEGORIES RESULTS SUMMARY', self.results.threeway_totals),
            ('RATIOS RESULTS SUMMARY', self.results.threeway_ratios),
            ('PERCENTAGE COMPOSITION RESULTS SUMMARY', self.results.threeway_percentages),
        ]:
            section_header = pd.DataFrame({'': [section_title]})
            section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                   index=False, header=False)
            current_row += 1
            if results_dict:
                summary = get_threeway_differences_summary(results_dict)
                summary.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                current_row += len(summary) + 3
            else:
                current_row += 2

        # Auto-adjust column widths
        ws = writer.sheets[sheet_name]
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            ws.column_dimensions[column_letter].width = min(max_length + 2, 60)

    def _write_threeway_results_sheet(
        self,
        writer: pd.ExcelWriter,
        sheet_name: str,
        title: str,
        tw_results: Dict[str, 'FullThreeWayAnalysisResult']
    ):
        """Write a three-way ANOVA results sheet for a category of variables."""
        sheet_name = sheet_name[:31]
        current_row = 0

        fa_name = self.results.factor_a_name
        fb_name = self.results.factor_b_name
        fc_name = self.results.factor_c_name

        title_df = pd.DataFrame({'': [title]})
        title_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                         index=False, header=False)
        current_row += 1
        design_df = pd.DataFrame({'': [f'Design: {fa_name} x {fb_name} x {fc_name}']})
        design_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                         index=False, header=False)
        current_row += 2

        # Summary table
        section_header = pd.DataFrame({'': ['OMNIBUS RESULTS SUMMARY']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 1

        summary = get_threeway_differences_summary(tw_results)
        summary.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
        current_row += len(summary) + 3

        # Detailed ANOVA tables
        section_header = pd.DataFrame({'': ['DETAILED ANOVA TABLES']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 2

        for var_name, result in tw_results.items():
            tw = result.threeway_result

            var_header = pd.DataFrame({'': [f'--- {var_name} ---']})
            var_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
            current_row += 1

            test_label = pd.DataFrame({'': [f'Test: {tw.test_type.value}']})
            test_label.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
            current_row += 1

            if tw.anova_table is not None and not tw.anova_table.empty:
                tw.anova_table.to_excel(writer, sheet_name=sheet_name,
                                        startrow=current_row, index=False)
                current_row += len(tw.anova_table) + 1

            # APA formatted result
            apa_text = format_threeway_apa(result)
            apa_df = pd.DataFrame({'APA Format': [apa_text]})
            apa_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += 2

            # Cell descriptive stats
            if result.descriptive_stats is not None and not result.descriptive_stats.empty:
                desc = result.descriptive_stats.copy()
                desc = desc[(desc['factor_a'] != '__MARGINAL__') &
                            (desc['factor_b'] != '__MARGINAL__') &
                            (desc['factor_c'] != '__MARGINAL__')]
                if not desc.empty:
                    desc_header = pd.DataFrame({'': ['Cell Descriptive Statistics']})
                    desc_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                        index=False, header=False)
                    current_row += 1
                    desc.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    current_row += len(desc) + 1

            # Post-hoc results
            if tw.posthoc_results is not None and not tw.posthoc_results.empty:
                ph_header = pd.DataFrame({'': [f'Post-hoc: {tw.posthoc_type}']})
                ph_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                  index=False, header=False)
                current_row += 1
                tw.posthoc_results.to_excel(writer, sheet_name=sheet_name,
                                            startrow=current_row, index=False)
                current_row += len(tw.posthoc_results) + 1

            current_row += 1

        # Auto-adjust column widths
        ws = writer.sheets[sheet_name]
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            ws.column_dimensions[column_letter].width = min(max_length + 2, 50)

    def _write_concentrations_sheet(self, writer: pd.ExcelWriter,
                                    sheet_name: str = 'Concentrations',
                                    title: str = 'INDIVIDUAL BILE ACID CONCENTRATION COMPARISONS',
                                    results_dict: Dict = None):
        """Write individual bile acid concentration comparisons sheet."""
        if results_dict is None:
            results_dict = self.results.individual_ba_results
        current_row = 0

        # Title
        title_df = pd.DataFrame({'': [title]})
        title_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                         index=False, header=False)
        current_row += 2

        # Summary table of all analytes
        section_header = pd.DataFrame({'': ['RESULTS SUMMARY']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 1

        summary_rows = []
        for ba_name, result in results_dict.items():
            sig_comparisons = []
            if result.posthoc_test and result.posthoc_test.pairwise_results is not None:
                sig_pairs = result.posthoc_test.pairwise_results[
                    result.posthoc_test.pairwise_results['significant'] == True
                ]
                for _, row in sig_pairs.iterrows():
                    sig_comparisons.append(f"{row['group1']} vs {row['group2']}")

            summary_rows.append({
                'Analyte': ba_name,
                'Test': result.main_test.test_type.value,
                'P-value': f"{result.main_test.pvalue:.6f}",
                'Significant': 'Yes' if result.main_test.significant else 'No',
                'Effect Size': f"{result.main_test.effect_size:.3f}" if result.main_test.effect_size else 'N/A',
                'Significant Comparisons': '; '.join(sig_comparisons) if sig_comparisons else 'None'
            })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += len(summary_df) + 3

        # Detailed results per analyte
        section_header = pd.DataFrame({'': ['DETAILED RESULTS']})
        section_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
        current_row += 2

        for ba_name, result in results_dict.items():
            # Analyte header
            var_header = pd.DataFrame({'': [f'--- {ba_name} ---']})
            var_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
            current_row += 1

            # Test info
            test_label = pd.DataFrame({'': [f'Test: {result.main_test.test_type.value}']})
            test_label.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                               index=False, header=False)
            current_row += 1

            # Descriptive statistics
            if result.descriptive_stats is not None and not result.descriptive_stats.empty:
                desc_header = pd.DataFrame({'': ['Descriptive Statistics']})
                desc_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                    index=False, header=False)
                current_row += 1
                result.descriptive_stats.to_excel(writer, sheet_name=sheet_name,
                                                  startrow=current_row, index=False)
                current_row += len(result.descriptive_stats) + 1

            # Main test result
            main_row = {
                'Statistic': f"{result.main_test.statistic:.4f}",
                'P-value': f"{result.main_test.pvalue:.6f}",
                'Significant': 'Yes' if result.main_test.significant else 'No',
                'Effect Size': f"{result.main_test.effect_size:.3f}" if result.main_test.effect_size else 'N/A',
            }
            main_df = pd.DataFrame([main_row])
            main_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += 3

            # Post-hoc results
            if result.posthoc_test and result.posthoc_test.pairwise_results is not None:
                ph_header = pd.DataFrame({'': [f'Post-hoc: {result.posthoc_test.test_type.value}']})
                ph_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                  index=False, header=False)
                current_row += 1
                result.posthoc_test.pairwise_results.to_excel(
                    writer, sheet_name=sheet_name, startrow=current_row, index=False)
                current_row += len(result.posthoc_test.pairwise_results) + 1

            current_row += 1  # Spacing between analytes

        # Auto-adjust column widths
        ws = writer.sheets[sheet_name]
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            ws.column_dimensions[column_letter].width = min(max_length + 2, 50)

    def get_significant_pairs(self, category_name: str) -> List[Tuple[str, str, str]]:
        """
        Get list of significant pairwise comparisons for a category.
        
        Returns list of (group1, group2, annotation) tuples for plotting.
        """
        if category_name not in self.analysis_sheets:
            return []
        
        sheet = self.analysis_sheets[category_name]
        if not sheet.statistical_result:
            return []
        
        result = sheet.statistical_result
        if not result.posthoc_test or result.posthoc_test.pairwise_results is None:
            return []
        
        pairs = []
        posthoc = result.posthoc_test.pairwise_results
        
        for _, row in posthoc.iterrows():
            if row['significant']:
                # Determine p-value annotation
                p = row.get('pvalue_adj', row.get('pvalue', 1.0))
                if p < 0.001:
                    annotation = '***'
                elif p < 0.01:
                    annotation = '**'
                elif p < 0.05:
                    annotation = '*'
                else:
                    continue
                
                pairs.append((row['group1'], row['group2'], annotation))
        
        return pairs


class SignificancePlotter:
    """
    Create publication-quality figures with significance annotations.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (8, 6)):
        self.figsize = figsize
        self.palette = sns.color_palette("Set2")
    
    def add_significance_brackets(
        self,
        ax: plt.Axes,
        pairs: List[Tuple[str, str, str]],
        group_order: List[str],
        y_data: pd.Series,
        group_col: str,
        data: pd.DataFrame
    ):
        """Add significance brackets to a plot."""
        if not pairs:
            return
        
        # Get y-axis range
        y_max = y_data.max()
        y_range = y_data.max() - y_data.min()
        bracket_height = y_range * 0.03
        bracket_gap = y_range * 0.08
        
        group_idx = {g: i for i, g in enumerate(group_order)}
        
        # Sort pairs by distance to minimize crossing brackets
        pairs = sorted(pairs, key=lambda x: abs(group_idx.get(x[0], 0) - group_idx.get(x[1], 0)))
        
        for i, (g1, g2, annotation) in enumerate(pairs):
            if g1 not in group_idx or g2 not in group_idx:
                continue
            
            x1, x2 = group_idx[g1], group_idx[g2]
            if x1 > x2:
                x1, x2 = x2, x1
            
            # Calculate y position for this bracket
            y = y_max + (i + 1) * bracket_gap
            
            # Draw bracket
            ax.plot([x1, x1, x2, x2], 
                   [y - bracket_height, y, y, y - bracket_height],
                   color='black', linewidth=1.2)
            
            # Add annotation
            ax.text((x1 + x2) / 2, y + bracket_height * 0.5, annotation,
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Adjust y-axis to fit brackets
        current_ylim = ax.get_ylim()
        new_top = y_max + (len(pairs) + 1.5) * bracket_gap
        ax.set_ylim(current_ylim[0], max(current_ylim[1], new_top))
    
    def plot_group_comparison_with_significance(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        significant_pairs: List[Tuple[str, str, str]],
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        plot_type: str = "box",
        show_points: bool = True,
        figsize: Optional[Tuple[float, float]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create group comparison plot with significance brackets."""
        if figsize is None:
            figsize = self.figsize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        groups = data[group_col].unique()
        group_order = list(groups)
        colors = dict(zip(groups, self.palette[:len(groups)]))
        
        if plot_type == "box":
            sns.boxplot(data=data, x=group_col, y=value_col, ax=ax,
                       hue=group_col, palette=colors, order=group_order, legend=False)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=value_col, ax=ax,
                            color='black', alpha=0.5, size=5, order=group_order)
        
        elif plot_type == "bar":
            means = data.groupby(group_col)[value_col].mean()
            sems = data.groupby(group_col)[value_col].sem()
            
            x = range(len(group_order))
            ax.bar(x, [means[g] for g in group_order],
                  yerr=[sems[g] for g in group_order],
                  capsize=5, color=[colors[g] for g in group_order],
                  edgecolor='black', linewidth=0.5, alpha=0.8)
            
            if show_points:
                for i, g in enumerate(group_order):
                    group_data = data[data[group_col] == g][value_col]
                    jitter = np.random.normal(0, 0.08, len(group_data))
                    ax.scatter(i + jitter, group_data, color='black',
                              alpha=0.5, s=25, zorder=3)
            
            ax.set_xticks(x)
            ax.set_xticklabels(group_order)
        
        elif plot_type == "violin":
            sns.violinplot(data=data, x=group_col, y=value_col, ax=ax,
                          hue=group_col, palette=colors, order=group_order,
                          inner='box', legend=False)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=value_col, ax=ax,
                            color='black', alpha=0.5, size=5, order=group_order)

        elif plot_type == "strip":
            sns.stripplot(data=data, x=group_col, y=value_col, ax=ax,
                         hue=group_col, palette=colors, order=group_order,
                         size=6, alpha=0.7, legend=False)
            # Add mean ± SEM lines
            means = data.groupby(group_col)[value_col].mean()
            sems = data.groupby(group_col)[value_col].sem()
            for i, g in enumerate(group_order):
                if g in means.index:
                    ax.hlines(means[g], i - 0.25, i + 0.25, color='black', linewidth=1.5, zorder=5)
                    ax.errorbar(i, means[g], yerr=sems[g], color='black',
                               capsize=5, capthick=1.5, linewidth=1.5, fmt='none', zorder=5)

        # Add significance brackets
        self.add_significance_brackets(
            ax, significant_pairs, group_order,
            data[value_col], group_col, data
        )
        
        ax.set_xlabel('')
        ax.set_ylabel(ylabel or value_col)
        ax.set_title(title or f'{value_col} by {group_col}')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_multi_panel_with_significance(
        self,
        data: pd.DataFrame,
        group_col: str,
        report_generator: ExcelReportGenerator,
        categories: Optional[List[str]] = None,
        ncols: int = 2,
        plot_type: str = "box",
        figsize: Optional[Tuple[float, float]] = None
    ) -> plt.Figure:
        """Create multi-panel figure with significance annotations."""
        if categories is None:
            categories = list(report_generator.analysis_sheets.keys())
        
        n_plots = len(categories)
        nrows = int(np.ceil(n_plots / ncols))
        
        if figsize is None:
            figsize = (5 * ncols, 4.5 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        groups = data[group_col].unique()
        group_order = list(groups)
        colors = dict(zip(groups, self.palette[:len(groups)]))
        
        for i, category in enumerate(categories):
            ax = axes[i]
            sheet = report_generator.analysis_sheets.get(category)
            
            if sheet is None:
                ax.set_visible(False)
                continue
            
            # Get the Total column from raw_data
            plot_data = sheet.raw_data[[group_col, 'Total']].copy()
            
            # Plot
            if plot_type == "box":
                sns.boxplot(data=plot_data, x=group_col, y='Total', ax=ax,
                           hue=group_col, palette=colors, order=group_order, legend=False)
                sns.stripplot(data=plot_data, x=group_col, y='Total', ax=ax,
                            color='black', alpha=0.5, size=4, order=group_order)
            elif plot_type == "violin":
                sns.violinplot(data=plot_data, x=group_col, y='Total', ax=ax,
                              hue=group_col, palette=colors, order=group_order,
                              inner='box', legend=False)
                sns.stripplot(data=plot_data, x=group_col, y='Total', ax=ax,
                            color='black', alpha=0.5, size=4, order=group_order)
            elif plot_type == "bar":
                means = plot_data.groupby(group_col)['Total'].mean()
                sems = plot_data.groupby(group_col)['Total'].sem()
                x = range(len(group_order))
                ax.bar(x, [means[g] for g in group_order],
                      yerr=[sems[g] for g in group_order],
                      capsize=4, color=[colors[g] for g in group_order],
                      edgecolor='black', linewidth=0.5, alpha=0.8)

                for j, g in enumerate(group_order):
                    gdata = plot_data[plot_data[group_col] == g]['Total']
                    jitter = np.random.normal(0, 0.08, len(gdata))
                    ax.scatter(j + jitter, gdata, color='black', alpha=0.5, s=20, zorder=3)

                ax.set_xticks(x)
                ax.set_xticklabels(group_order)

            elif plot_type == "strip":
                sns.stripplot(data=plot_data, x=group_col, y='Total', ax=ax,
                             hue=group_col, palette=colors, order=group_order,
                             size=5, alpha=0.7, legend=False)
                # Add mean ± SEM lines
                means = plot_data.groupby(group_col)['Total'].mean()
                sems = plot_data.groupby(group_col)['Total'].sem()
                for j, g in enumerate(group_order):
                    if g in means.index:
                        ax.hlines(means[g], j - 0.25, j + 0.25, color='black', linewidth=1.5, zorder=5)
                        ax.errorbar(j, means[g], yerr=sems[g], color='black',
                                   capsize=4, capthick=1.5, linewidth=1.5, fmt='none', zorder=5)

            # Add significance brackets
            sig_pairs = report_generator.get_significant_pairs(category)
            self.add_significance_brackets(
                ax, sig_pairs, group_order,
                plot_data['Total'], group_col, plot_data
            )
            
            # Format
            ax.set_title(category.replace('_', ' '), fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(f'Concentration ({report_generator.units})', fontsize=9)
            ax.tick_params(axis='x', rotation=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Hide empty panels
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    print("Report generation module loaded.")
    print("Classes: ExcelReportGenerator, SignificancePlotter, ComprehensiveAnalysisResults")
