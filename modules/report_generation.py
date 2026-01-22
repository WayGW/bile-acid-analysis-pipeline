"""
Report Generation Module for Bile Acid Analysis
=================================================

Generates organized Excel workbooks with:
- Separate sheets for each analysis type
- Raw data → calculations → statistical results on each sheet
- Transparency in data flow

Also handles figure generation with significance annotations.
"""

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
    get_sulfated
)
from modules.statistical_tests import StatisticalAnalyzer, FullAnalysisResult


@dataclass
class ComprehensiveAnalysisResults:
    """Container for all statistical results."""
    individual_ba_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)
    totals_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)
    ratios_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)
    percentages_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)  # NEW
    category_results: Dict[str, FullAnalysisResult] = field(default_factory=dict)


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
        alpha: float = 0.05
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
        """
        self.data = data.copy()
        self.group_col = group_col
        self.sample_id_col = sample_id_col
        self.alpha = alpha
        self.totals = totals
        self.ratios = ratios
        self.percentages = percentages
        
        # Identify bile acid columns
        if bile_acid_cols:
            self.ba_cols = [c for c in bile_acid_cols if c in data.columns]
        else:
            # Auto-detect from panel
            self.ba_cols = [c for c in BILE_ACID_PANEL.keys() if c in data.columns]
        
        self.analyzer = StatisticalAnalyzer(alpha=alpha)
        self.analysis_sheets: Dict[str, AnalysisSheet] = {}
        self.results = ComprehensiveAnalysisResults()
    
    def run_all_statistics(self) -> ComprehensiveAnalysisResults:
        """Run all statistical analyses and cache results."""
        # Filter valid groups
        valid_data = self.data[self.data[self.group_col].notna()].copy()
        valid_data = valid_data[valid_data[self.group_col].astype(str).str.lower() != 'nan']
        
        # 1. Individual bile acids
        for ba in self.ba_cols:
            if ba in valid_data.columns:
                try:
                    result = self.analyzer.analyze(valid_data, ba, self.group_col)
                    self.results.individual_ba_results[ba] = result
                except Exception as e:
                    print(f"Could not analyze {ba}: {e}")
        
        # 2. Totals
        if self.totals is not None:
            combined = pd.concat([valid_data[[self.group_col]], self.totals.loc[valid_data.index]], axis=1)
            for col in self.totals.columns:
                if not combined[col].isna().all():
                    try:
                        result = self.analyzer.analyze(combined, col, self.group_col)
                        self.results.totals_results[col] = result
                    except Exception as e:
                        print(f"Could not analyze {col}: {e}")
        
        # 3. Ratios
        if self.ratios is not None:
            combined = pd.concat([valid_data[[self.group_col]], self.ratios.loc[valid_data.index]], axis=1)
            for col in self.ratios.columns:
                if not combined[col].isna().all():
                    try:
                        result = self.analyzer.analyze(combined, col, self.group_col)
                        self.results.ratios_results[col] = result
                    except Exception as e:
                        print(f"Could not analyze {col}: {e}")
        
        # 4. Percentages (NEW)
        if self.percentages is not None:
            combined = pd.concat([valid_data[[self.group_col]], self.percentages.loc[valid_data.index]], axis=1)
            for col in self.percentages.columns:
                if not combined[col].isna().all():
                    try:
                        result = self.analyzer.analyze(combined, col, self.group_col)
                        self.results.percentages_results[col] = result
                    except Exception as e:
                        print(f"Could not analyze {col}: {e}")
        
        # 5. Category totals (for Excel sheets)
        self.generate_all_sheets()
        for cat_name, sheet in self.analysis_sheets.items():
            if sheet.statistical_result:
                self.results.category_results[cat_name] = sheet.statistical_result
        
        return self.results
    
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
        
        available_ba_cols = [c for c in ba_columns if c in self.data.columns]
        raw_data = self.data[id_cols + available_ba_cols].copy()
        
        # Calculate total for each sample
        raw_data['Total'] = raw_data[available_ba_cols].sum(axis=1)
        
        # Calculate total of ALL bile acids for percentage calculation
        all_ba_total = self.data[self.ba_cols].sum(axis=1)
        raw_data['Pct_of_Total_BA'] = (raw_data['Total'] / all_ba_total * 100).round(2)
        
        # Group summary statistics
        group_totals = raw_data.groupby(self.group_col).agg({
            'Total': ['count', 'mean', 'std', 'sem', 'median', 'min', 'max'],
            'Pct_of_Total_BA': ['mean', 'std', 'sem']
        }).round(4)
        
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
            group_percentages = raw_data.groupby(self.group_col)[pct_cols].mean().round(2)
        
        # Statistical analysis
        stat_result = None
        if len(self.data[self.group_col].unique()) >= 2:
            try:
                stat_result = self.analyzer.analyze(raw_data, 'Total', self.group_col)
            except Exception as e:
                print(f"Could not analyze {category_name}: {e}")
        
        return AnalysisSheet(
            name=category_name,
            description=description,
            bile_acid_columns=available_ba_cols,
            raw_data=raw_data,
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
            # Overview sheet first
            self._write_overview_sheet(writer)
            
            # Individual analysis sheets
            for sheet_name, sheet in self.analysis_sheets.items():
                self._write_sheet(writer, sheet)
        
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
        summary_data = {
            'Parameter': ['Total Samples', 'Groups', 'Bile Acids Measured', 'Significance Level'],
            'Value': [
                len(self.data),
                ', '.join(self.data[self.group_col].unique().astype(str)),
                len(self.ba_cols),
                f"α = {self.alpha}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Overview', startrow=current_row, index=False)
        current_row += 6
        
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
            
            # Add significance brackets
            sig_pairs = report_generator.get_significant_pairs(category)
            self.add_significance_brackets(
                ax, sig_pairs, group_order,
                plot_data['Total'], group_col, plot_data
            )
            
            # Format
            ax.set_title(category.replace('_', ' '), fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Concentration (nmol/L)', fontsize=9)
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
