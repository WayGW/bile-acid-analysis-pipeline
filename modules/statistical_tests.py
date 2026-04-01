"""
Statistical Testing Module for Bile Acid Analysis
==================================================

This module handles:
1. Assumption testing (normality, homoscedasticity)
2. Automatic test selection based on assumption results
3. Multiple comparison corrections
4. Effect size calculations

Supports both two-group and multi-group comparisons.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Suppress expected RuntimeWarnings from scipy when analyzing near-constant
# or LOD-replaced data groups (divide-by-zero in Shapiro-Wilk, precision loss
# in moment calculations, etc.)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\.stats")


class TestType(Enum):
    """Types of statistical tests available."""
    # Two-group tests
    TTEST_INDEPENDENT = "independent_t_test"
    TTEST_WELCH = "welch_t_test"
    MANN_WHITNEY = "mann_whitney_u"

    # Multi-group tests (one-way)
    ANOVA_ONEWAY = "one_way_anova"
    ANOVA_WELCH = "welch_anova"
    KRUSKAL_WALLIS = "kruskal_wallis"

    # Two-way factorial tests
    ANOVA_TWOWAY = "two_way_anova"
    ART_ANOVA = "art_anova"  # Aligned Rank Transform (non-parametric two-way)

    # Post-hoc tests
    TUKEY_HSD = "tukey_hsd"
    GAMES_HOWELL = "games_howell"
    DUNN = "dunn_test"


class CorrectionMethod(Enum):
    """Multiple comparison correction methods."""
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    FDR_BH = "fdr_bh"  # Benjamini-Hochberg
    FDR_BY = "fdr_by"  # Benjamini-Yekutieli
    NONE = "none"


@dataclass
class AssumptionResults:
    """Results from assumption testing."""
    # Normality
    normality_test: str = "shapiro"
    normality_stats: Dict[str, float] = field(default_factory=dict)
    normality_pvalues: Dict[str, float] = field(default_factory=dict)
    normality_passed: Dict[str, bool] = field(default_factory=dict)
    overall_normality: bool = False
    
    # Homoscedasticity
    homoscedasticity_test: str = "levene"
    homoscedasticity_stat: float = 0.0
    homoscedasticity_pvalue: float = 0.0
    homoscedasticity_passed: bool = False
    
    # Sample size info
    group_sizes: Dict[str, int] = field(default_factory=dict)
    min_group_size: int = 0
    total_n: int = 0
    
    # Recommended test
    recommended_test: TestType = None
    recommendation_reason: str = ""
    
    alpha: float = 0.05


@dataclass 
class StatisticalResult:
    """Results from a statistical test."""
    test_type: TestType
    statistic: float
    pvalue: float
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None
    effect_size_interpretation: Optional[str] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    df: Optional[float] = None
    groups_compared: List[str] = field(default_factory=list)
    significant: bool = False
    alpha: float = 0.05
    
    # For post-hoc tests
    pairwise_results: Optional[pd.DataFrame] = None


@dataclass
class FullAnalysisResult:
    """Complete analysis results for a variable."""
    variable_name: str
    assumptions: AssumptionResults
    main_test: StatisticalResult
    posthoc_test: Optional[StatisticalResult] = None
    descriptive_stats: Optional[pd.DataFrame] = None


@dataclass
class TwoWayResult:
    """Results from a two-way ANOVA (parametric or ART)."""
    test_type: TestType  # ANOVA_TWOWAY or ART_ANOVA
    factor_a_name: str
    factor_b_name: str

    # Main effects and interaction
    factor_a_stat: float = np.nan
    factor_a_pvalue: float = np.nan
    factor_a_df: Tuple = (0, 0)  # (df_num, df_den)
    factor_b_stat: float = np.nan
    factor_b_pvalue: float = np.nan
    factor_b_df: Tuple = (0, 0)
    interaction_stat: float = np.nan
    interaction_pvalue: float = np.nan
    interaction_df: Tuple = (0, 0)
    interaction_significant: bool = False

    # Effect sizes (partial eta-squared)
    effect_sizes: Dict[str, float] = field(default_factory=dict)

    # Post-hoc results
    posthoc_results: Optional[pd.DataFrame] = None
    posthoc_type: str = ""  # "simple_main_effects" or "main_effect_marginal"

    # ANOVA table (Source, SS, df, MS, F, p, partial_eta_sq)
    anova_table: Optional[pd.DataFrame] = None

    alpha: float = 0.05


@dataclass
class FullTwoWayAnalysisResult:
    """Complete two-way analysis results for a variable."""
    variable_name: str
    twoway_result: TwoWayResult
    descriptive_stats: Optional[pd.DataFrame] = None  # Cell means (FactorA x FactorB)
    assumption_notes: str = ""


class StatisticalAnalyzer:
    """
    Automated statistical analysis with assumption checking.
    
    Usage:
        analyzer = StatisticalAnalyzer(alpha=0.05)
        results = analyzer.analyze(data, value_col='concentration', group_col='group')
    """
    
    # Minimum sample size for parametric tests (per group)
    MIN_PARAMETRIC_N = 5
    
    def __init__(
        self, 
        alpha: float = 0.05,
        normality_test: str = "shapiro",
        min_sample_for_normality: int = 3,
        correction_method: CorrectionMethod = CorrectionMethod.FDR_BH
    ):
        """
        Initialize the analyzer.
        
        Args:
            alpha: Significance level for all tests
            normality_test: 'shapiro' or 'dagostino' or 'anderson'
            min_sample_for_normality: Minimum samples needed for normality test
            correction_method: Method for multiple comparison correction
        """
        self.alpha = alpha
        self.normality_test = normality_test
        self.min_sample_for_normality = min_sample_for_normality
        self.correction_method = correction_method
    
    def check_normality(
        self, 
        data: Union[np.ndarray, pd.Series],
        group_name: str = "data"
    ) -> Tuple[float, float, bool]:
        """
        Test normality of a single group.
        
        Returns:
            Tuple of (statistic, p-value, passed)
        """
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        if len(data) < self.min_sample_for_normality:
            # Not enough data to test
            return np.nan, np.nan, True  # Assume normal if can't test

        if np.ptp(data) == 0 or np.std(data) < 1e-10:
            # Constant or near-constant values - normality is undefined, skip test
            return np.nan, np.nan, True

        if len(data) > 5000:
            # For large samples, use D'Agostino-Pearson
            try:
                stat, p = stats.normaltest(data)
            except:
                stat, p = np.nan, 1.0
        else:
            # Shapiro-Wilk for smaller samples
            try:
                stat, p = stats.shapiro(data)
            except:
                stat, p = np.nan, 1.0
        
        passed = p > self.alpha
        return stat, p, passed
    
    def check_homoscedasticity(
        self, 
        groups: List[np.ndarray]
    ) -> Tuple[float, float, bool]:
        """
        Test homogeneity of variances using Levene's test.
        
        Returns:
            Tuple of (statistic, p-value, passed)
        """
        # Remove NaN values from each group
        clean_groups = [g[~np.isnan(g)] for g in groups]
        
        # Need at least 2 groups with data
        valid_groups = [g for g in clean_groups if len(g) >= 2]
        
        if len(valid_groups) < 2:
            return np.nan, np.nan, True
        
        try:
            stat, p = stats.levene(*valid_groups, center='median')  # median is more robust
        except:
            stat, p = np.nan, 1.0
        
        passed = p > self.alpha
        return stat, p, passed
    
    def check_assumptions(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str
    ) -> AssumptionResults:
        """
        Check all assumptions and recommend appropriate test.
        
        Args:
            data: DataFrame with values and group labels
            value_col: Column name for the numeric values
            group_col: Column name for group labels
            
        Returns:
            AssumptionResults with all assumption tests and recommendations
        """
        results = AssumptionResults(alpha=self.alpha)
        
        # Get groups
        groups = data[group_col].unique()
        group_data = {g: data[data[group_col] == g][value_col].values for g in groups}
        
        # Record group sizes
        results.group_sizes = {g: len(d[~np.isnan(d)]) for g, d in group_data.items()}
        results.min_group_size = min(results.group_sizes.values()) if results.group_sizes else 0
        results.total_n = sum(results.group_sizes.values())
        
        # Test normality for each group
        for group_name, values in group_data.items():
            stat, p, passed = self.check_normality(values, group_name)
            results.normality_stats[group_name] = stat
            results.normality_pvalues[group_name] = p
            results.normality_passed[group_name] = passed
        
        # Overall normality passes only if all groups pass
        results.overall_normality = all(results.normality_passed.values())
        
        # Test homoscedasticity
        group_arrays = [np.array(v) for v in group_data.values()]
        homo_stat, homo_p, homo_passed = self.check_homoscedasticity(group_arrays)
        results.homoscedasticity_stat = homo_stat
        results.homoscedasticity_pvalue = homo_p
        results.homoscedasticity_passed = homo_passed
        
        # Recommend test based on assumptions and number of groups
        n_groups = len(groups)
        
        if n_groups == 2:
            results.recommended_test, results.recommendation_reason = \
                self._recommend_two_group_test(results)
        else:
            results.recommended_test, results.recommendation_reason = \
                self._recommend_multi_group_test(results)
        
        return results
    
    def _get_failed_normality_groups(self, assumptions: AssumptionResults) -> List[str]:
        """Get list of groups that failed normality test."""
        return [g for g, passed in assumptions.normality_passed.items() if not passed]
    
    def _recommend_two_group_test(
        self, 
        assumptions: AssumptionResults
    ) -> Tuple[TestType, str]:
        """Recommend appropriate two-group test."""
        
        # PRIORITY 1: Check normality first
        if not assumptions.overall_normality:
            failed_groups = self._get_failed_normality_groups(assumptions)
            return (
                TestType.MANN_WHITNEY,
                f"Normality violated in: {', '.join(failed_groups)}"
            )
        
        # PRIORITY 2: Very small samples (normality passed but sample too small for robust parametric)
        if assumptions.min_group_size < self.MIN_PARAMETRIC_N:
            return (
                TestType.MANN_WHITNEY,
                f"Small sample size (n={assumptions.min_group_size}); using non-parametric for robustness"
            )
        
        # PRIORITY 3: Check homoscedasticity (normality passed, adequate sample)
        if assumptions.homoscedasticity_passed:
            return (
                TestType.TTEST_INDEPENDENT,
                "Normality and homoscedasticity assumptions met"
            )
        else:
            return (
                TestType.TTEST_WELCH,
                f"Normality met but unequal variances (Levene p={assumptions.homoscedasticity_pvalue:.4f})"
            )
    
    def _recommend_multi_group_test(
        self, 
        assumptions: AssumptionResults
    ) -> Tuple[TestType, str]:
        """Recommend appropriate multi-group test."""
        
        # PRIORITY 1: Check normality first
        if not assumptions.overall_normality:
            failed_groups = self._get_failed_normality_groups(assumptions)
            return (
                TestType.KRUSKAL_WALLIS,
                f"Normality violated in: {', '.join(failed_groups)}"
            )
        
        # PRIORITY 2: Very small samples (normality passed but sample too small for robust parametric)
        if assumptions.min_group_size < self.MIN_PARAMETRIC_N:
            return (
                TestType.KRUSKAL_WALLIS,
                f"Small sample size (n={assumptions.min_group_size}); using non-parametric for robustness"
            )
        
        # PRIORITY 3: Check homoscedasticity (normality passed, adequate sample)
        if assumptions.homoscedasticity_passed:
            return (
                TestType.ANOVA_ONEWAY,
                "Normality and homoscedasticity assumptions met"
            )
        else:
            return (
                TestType.ANOVA_WELCH,
                f"Normality met but unequal variances (Levene p={assumptions.homoscedasticity_pvalue:.4f})"
            )
    
    def calculate_effect_size(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        test_type: TestType
    ) -> Tuple[float, str, str]:
        """
        Calculate appropriate effect size for the test type.
        
        Returns:
            Tuple of (effect_size, effect_type, interpretation)
        """
        groups = data[group_col].unique()
        group_data = [data[data[group_col] == g][value_col].dropna().values 
                      for g in groups]
        
        if test_type in [TestType.TTEST_INDEPENDENT, TestType.TTEST_WELCH, TestType.MANN_WHITNEY]:
            # Cohen's d for two groups
            g1, g2 = group_data[0], group_data[1]
            
            # Pooled standard deviation
            n1, n2 = len(g1), len(g2)
            s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
            pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            
            if pooled_sd == 0:
                return 0.0, "cohens_d", "undefined"
            
            d = (np.mean(g1) - np.mean(g2)) / pooled_sd
            
            # Interpret Cohen's d
            abs_d = abs(d)
            if abs_d < 0.2:
                interp = "negligible"
            elif abs_d < 0.5:
                interp = "small"
            elif abs_d < 0.8:
                interp = "medium"
            else:
                interp = "large"
            
            return d, "cohens_d", interp
        
        elif test_type in [TestType.ANOVA_ONEWAY, TestType.ANOVA_WELCH, TestType.KRUSKAL_WALLIS]:
            # Eta-squared for multi-group
            all_data = np.concatenate(group_data)
            grand_mean = np.mean(all_data)
            
            # Between-group sum of squares
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
            
            # Total sum of squares
            ss_total = np.sum((all_data - grand_mean)**2)
            
            if ss_total == 0:
                return 0.0, "eta_squared", "undefined"
            
            eta_sq = ss_between / ss_total
            
            # Interpret eta-squared
            if eta_sq < 0.01:
                interp = "negligible"
            elif eta_sq < 0.06:
                interp = "small"
            elif eta_sq < 0.14:
                interp = "medium"
            else:
                interp = "large"
            
            return eta_sq, "eta_squared", interp
        
        return None, None, None
    
    def run_test(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        test_type: TestType
    ) -> StatisticalResult:
        """
        Run the specified statistical test.
        """
        groups = data[group_col].unique().tolist()
        group_data = [data[data[group_col] == g][value_col].dropna().values 
                      for g in groups]
        
        result = StatisticalResult(
            test_type=test_type,
            statistic=np.nan,
            pvalue=np.nan,
            groups_compared=groups,
            alpha=self.alpha
        )
        
        try:
            if test_type == TestType.TTEST_INDEPENDENT:
                stat, p = stats.ttest_ind(group_data[0], group_data[1])
                df = len(group_data[0]) + len(group_data[1]) - 2
                result.statistic = stat
                result.pvalue = p
                result.df = df
                
            elif test_type == TestType.TTEST_WELCH:
                stat, p = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
                result.statistic = stat
                result.pvalue = p
                
            elif test_type == TestType.MANN_WHITNEY:
                stat, p = stats.mannwhitneyu(group_data[0], group_data[1], 
                                            alternative='two-sided')
                result.statistic = stat
                result.pvalue = p
                
            elif test_type == TestType.ANOVA_ONEWAY:
                stat, p = stats.f_oneway(*group_data)
                result.statistic = stat
                result.pvalue = p
                
            elif test_type == TestType.ANOVA_WELCH:
                # Welch's ANOVA using scipy (pingouin alternative)
                stat, p = self._welch_anova(group_data)
                result.statistic = stat
                result.pvalue = p
                
            elif test_type == TestType.KRUSKAL_WALLIS:
                stat, p = stats.kruskal(*group_data)
                result.statistic = stat
                result.pvalue = p
                
        except Exception as e:
            warnings.warn(f"Test failed: {str(e)}")
        
        result.significant = result.pvalue < self.alpha
        
        # Calculate effect size
        effect_size, effect_type, effect_interp = self.calculate_effect_size(
            data, value_col, group_col, test_type
        )
        result.effect_size = effect_size
        result.effect_size_type = effect_type
        result.effect_size_interpretation = effect_interp
        
        return result
    
    def _welch_anova(self, groups: List[np.ndarray]) -> Tuple[float, float]:
        """Perform Welch's ANOVA for unequal variances."""
        k = len(groups)
        ns = np.array([len(g) for g in groups])
        means = np.array([np.mean(g) for g in groups])
        variances = np.array([np.var(g, ddof=1) for g in groups])
        
        # Weights
        w = ns / variances
        
        # Weighted grand mean
        grand_mean = np.sum(w * means) / np.sum(w)
        
        # F statistic numerator
        f_num = np.sum(w * (means - grand_mean)**2) / (k - 1)
        
        # Lambda term for denominator
        lambda_term = 3 * np.sum((1 - w/np.sum(w))**2 / (ns - 1)) / (k**2 - 1)
        
        # F statistic denominator
        f_denom = 1 + 2 * (k - 2) / (k**2 - 1) * lambda_term
        
        F = f_num / f_denom
        
        # Degrees of freedom
        df1 = k - 1
        df2 = 1 / lambda_term
        
        # P-value
        p = 1 - stats.f.cdf(F, df1, df2)
        
        return F, p
    
    def run_posthoc(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        assumptions: AssumptionResults
    ) -> Optional[StatisticalResult]:
        """
        Run appropriate post-hoc tests for multi-group comparisons.
        """
        groups = data[group_col].unique()
        if len(groups) <= 2:
            return None
        
        # Select post-hoc test based on assumptions
        if assumptions.overall_normality and assumptions.homoscedasticity_passed:
            return self._tukey_hsd(data, value_col, group_col)
        elif assumptions.overall_normality:
            return self._games_howell(data, value_col, group_col)
        else:
            return self._dunn_test(data, value_col, group_col)
    
    def _tukey_hsd(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str
    ) -> StatisticalResult:
        """Tukey's HSD test."""
        from scipy.stats import tukey_hsd
        
        groups = data[group_col].unique()
        group_data = [data[data[group_col] == g][value_col].dropna().values 
                      for g in groups]
        
        result = tukey_hsd(*group_data)
        
        # Build pairwise results DataFrame
        pairwise = []
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i < j:
                    pairwise.append({
                        'group1': g1,
                        'group2': g2,
                        'mean_diff': result.statistic[i, j],
                        'pvalue': result.pvalue[i, j],
                        'significant': result.pvalue[i, j] < self.alpha
                    })
        
        return StatisticalResult(
            test_type=TestType.TUKEY_HSD,
            statistic=np.nan,
            pvalue=np.nan,
            groups_compared=list(groups),
            pairwise_results=pd.DataFrame(pairwise),
            alpha=self.alpha
        )
    
    def _games_howell(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str
    ) -> StatisticalResult:
        """Games-Howell test for unequal variances."""
        groups = data[group_col].unique()
        group_data = {g: data[data[group_col] == g][value_col].dropna().values 
                      for g in groups}
        
        pairwise = []
        group_list = list(groups)
        
        for i in range(len(group_list)):
            for j in range(i + 1, len(group_list)):
                g1, g2 = group_list[i], group_list[j]
                x1, x2 = group_data[g1], group_data[g2]
                
                n1, n2 = len(x1), len(x2)
                m1, m2 = np.mean(x1), np.mean(x2)
                v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
                
                se = np.sqrt(v1/n1 + v2/n2)
                mean_diff = m1 - m2
                t_stat = mean_diff / se if se > 0 else 0
                
                # Welch-Satterthwaite degrees of freedom
                df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
                
                # p-value (two-tailed)
                p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                pairwise.append({
                    'group1': g1,
                    'group2': g2,
                    'mean_diff': mean_diff,
                    't_stat': t_stat,
                    'df': df,
                    'pvalue': p,
                    'significant': p < self.alpha
                })
        
        # Apply correction
        pairwise_df = pd.DataFrame(pairwise)
        pairwise_df['pvalue_adj'] = self._adjust_pvalues(pairwise_df['pvalue'].values)
        pairwise_df['significant'] = pairwise_df['pvalue_adj'] < self.alpha
        
        return StatisticalResult(
            test_type=TestType.GAMES_HOWELL,
            statistic=np.nan,
            pvalue=np.nan,
            groups_compared=list(groups),
            pairwise_results=pairwise_df,
            alpha=self.alpha
        )
    
    def _dunn_test(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str
    ) -> StatisticalResult:
        """Dunn's test for non-parametric post-hoc comparisons."""
        groups = data[group_col].unique()
        
        # Rank all data
        data = data.copy()
        data['_rank'] = data[value_col].rank()
        
        n_total = len(data)
        group_data = {g: data[data[group_col] == g] for g in groups}
        
        pairwise = []
        group_list = list(groups)
        
        for i in range(len(group_list)):
            for j in range(i + 1, len(group_list)):
                g1, g2 = group_list[i], group_list[j]
                
                n1 = len(group_data[g1])
                n2 = len(group_data[g2])
                
                mean_rank1 = group_data[g1]['_rank'].mean()
                mean_rank2 = group_data[g2]['_rank'].mean()
                
                # Z statistic
                se = np.sqrt((n_total * (n_total + 1) / 12) * (1/n1 + 1/n2))
                z = (mean_rank1 - mean_rank2) / se if se > 0 else 0
                
                # Two-tailed p-value
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                
                pairwise.append({
                    'group1': g1,
                    'group2': g2,
                    'mean_rank_diff': mean_rank1 - mean_rank2,
                    'z_stat': z,
                    'pvalue': p,
                    'significant': p < self.alpha
                })
        
        # Apply correction
        pairwise_df = pd.DataFrame(pairwise)
        pairwise_df['pvalue_adj'] = self._adjust_pvalues(pairwise_df['pvalue'].values)
        pairwise_df['significant'] = pairwise_df['pvalue_adj'] < self.alpha
        
        return StatisticalResult(
            test_type=TestType.DUNN,
            statistic=np.nan,
            pvalue=np.nan,
            groups_compared=list(groups),
            pairwise_results=pairwise_df,
            alpha=self.alpha
        )
    
    def _adjust_pvalues(self, pvalues: np.ndarray) -> np.ndarray:
        """Apply multiple comparison correction."""
        from scipy.stats import false_discovery_control
        
        if self.correction_method == CorrectionMethod.NONE:
            return pvalues
        
        elif self.correction_method == CorrectionMethod.BONFERRONI:
            return np.minimum(pvalues * len(pvalues), 1.0)
        
        elif self.correction_method == CorrectionMethod.HOLM:
            n = len(pvalues)
            sorted_idx = np.argsort(pvalues)
            sorted_p = pvalues[sorted_idx]
            adjusted = np.zeros_like(pvalues)
            
            for i, idx in enumerate(sorted_idx):
                adjusted[idx] = min(sorted_p[i] * (n - i), 1.0)
            
            # Ensure monotonicity
            for i in range(1, n):
                adjusted[sorted_idx[i]] = max(adjusted[sorted_idx[i]], 
                                              adjusted[sorted_idx[i-1]])
            return adjusted
        
        elif self.correction_method == CorrectionMethod.FDR_BH:
            return false_discovery_control(pvalues, method='bh')
        
        elif self.correction_method == CorrectionMethod.FDR_BY:
            return false_discovery_control(pvalues, method='by')
        
        return pvalues
    
    def get_descriptive_stats(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str
    ) -> pd.DataFrame:
        """Calculate descriptive statistics by group."""
        stats_df = data.groupby(group_col)[value_col].agg([
            ('n', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('sem', 'sem'),
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('min', 'min'),
            ('max', 'max')
        ]).round(4)
        
        # Add 95% CI
        stats_df['ci95_lower'] = stats_df['mean'] - 1.96 * stats_df['sem']
        stats_df['ci95_upper'] = stats_df['mean'] + 1.96 * stats_df['sem']
        
        return stats_df
    
    def analyze(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        run_posthoc: bool = True
    ) -> FullAnalysisResult:
        """
        Perform complete statistical analysis.
        
        This is the main entry point - checks assumptions, selects appropriate
        test, runs analysis, and optionally performs post-hoc tests.
        
        Args:
            data: DataFrame with values and group labels
            value_col: Column name for numeric values
            group_col: Column name for group labels
            run_posthoc: Whether to run post-hoc tests for >2 groups
            
        Returns:
            FullAnalysisResult with all analysis components
        """
        # Skip analysis if data has near-zero variance (e.g. all LOD-replaced)
        clean = data[[value_col, group_col]].dropna().copy()
        clean[value_col] = pd.to_numeric(clean[value_col], errors='coerce')
        clean = clean.dropna()
        if clean[value_col].std() < 1e-10:
            assumptions = AssumptionResults(
                recommendation_reason="Skipped: near-zero variance (all values identical, e.g. LOD-replaced)"
            )
            main_result = StatisticalResult(
                test_type=TestType.MANN_WHITNEY,
                statistic=np.nan,
                pvalue=np.nan,
                groups_compared=clean[group_col].unique().tolist(),
                alpha=self.alpha
            )
            descriptives = self.get_descriptive_stats(data, value_col, group_col)
            return FullAnalysisResult(
                variable_name=value_col,
                assumptions=assumptions,
                main_test=main_result,
                descriptive_stats=descriptives
            )

        # Check assumptions
        assumptions = self.check_assumptions(data, value_col, group_col)

        # Run main test
        main_result = self.run_test(
            data, value_col, group_col,
            assumptions.recommended_test
        )

        # Descriptive stats
        descriptives = self.get_descriptive_stats(data, value_col, group_col)

        # Post-hoc if needed
        posthoc = None
        n_groups = len(data[group_col].unique())
        if run_posthoc and n_groups > 2 and main_result.significant:
            posthoc = self.run_posthoc(data, value_col, group_col, assumptions)

        return FullAnalysisResult(
            variable_name=value_col,
            assumptions=assumptions,
            main_test=main_result,
            posthoc_test=posthoc,
            descriptive_stats=descriptives
        )

    # ================================================================
    # TWO-WAY ANOVA METHODS (fully additive — no changes to one-way)
    # ================================================================

    def analyze_twoway(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None
    ) -> FullTwoWayAnalysisResult:
        """
        Perform complete two-way factorial ANOVA analysis.

        Automatically selects between parametric two-way ANOVA and
        non-parametric ART (Aligned Rank Transform) ANOVA based on
        assumption checks.
        """
        fa_name = factor_a_name or factor_a_col
        fb_name = factor_b_name or factor_b_col

        # Clean data
        df = data[[value_col, factor_a_col, factor_b_col]].dropna().copy()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()

        # Ensure factors are categorical strings
        df[factor_a_col] = df[factor_a_col].astype(str).str.strip()
        df[factor_b_col] = df[factor_b_col].astype(str).str.strip()

        # Skip analysis if data has near-zero variance (e.g. all LOD-replaced)
        if df[value_col].std() < 1e-10:
            skip_result = TwoWayResult(
                test_type=TestType.ANOVA_TWOWAY,
                factor_a_name=fa_name,
                factor_b_name=fb_name,
                alpha=self.alpha
            )
            descriptives = self._get_twoway_descriptive_stats(
                df, value_col, factor_a_col, factor_b_col
            )
            return FullTwoWayAnalysisResult(
                variable_name=value_col,
                twoway_result=skip_result,
                descriptive_stats=descriptives,
                assumption_notes="Skipped: near-zero variance (all values identical, e.g. LOD-replaced)"
            )

        # Check assumptions and select test
        use_parametric, assumption_notes = self._recommend_twoway_test(
            df, value_col, factor_a_col, factor_b_col
        )

        # Run the appropriate two-way test
        if use_parametric:
            twoway_result = self._run_twoway_parametric(
                df, value_col, factor_a_col, factor_b_col, fa_name, fb_name
            )
        else:
            twoway_result = self._run_art_anova(
                df, value_col, factor_a_col, factor_b_col, fa_name, fb_name
            )

        # Run post-hoc tests based on interaction significance
        twoway_result = self._run_twoway_posthoc(
            df, value_col, factor_a_col, factor_b_col, twoway_result
        )

        # Cell descriptive statistics (Factor A x Factor B means)
        descriptives = self._get_twoway_descriptive_stats(
            df, value_col, factor_a_col, factor_b_col
        )

        return FullTwoWayAnalysisResult(
            variable_name=value_col,
            twoway_result=twoway_result,
            descriptive_stats=descriptives,
            assumption_notes=assumption_notes
        )

    def _recommend_twoway_test(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str
    ) -> Tuple[bool, str]:
        """Check assumptions for two-way ANOVA and recommend parametric vs ART."""
        notes = []
        normality_ok = True
        homogeneity_ok = True

        # Check normality within each cell (Factor A x Factor B combination)
        cells = data.groupby([factor_a_col, factor_b_col])
        group_arrays = []

        for (a, b), group_df in cells:
            values = group_df[value_col].dropna().values
            group_arrays.append(values)

            if len(values) >= self.min_sample_for_normality:
                _, p, passed = self.check_normality(values, f"{a}_{b}")
                if not passed:
                    normality_ok = False
                    notes.append(f"Normality violated in cell ({a}, {b}), p={p:.4f}")
            elif len(values) < 3:
                notes.append(f"Cell ({a}, {b}) has n={len(values)}, too small for normality test")

        # Check homogeneity of variances across cells
        valid_groups = [g for g in group_arrays if len(g) >= 2]
        if len(valid_groups) >= 2:
            _, homo_p, homo_passed = self.check_homoscedasticity(
                [np.array(g) for g in valid_groups]
            )
            if not homo_passed:
                homogeneity_ok = False
                notes.append(f"Homogeneity of variances violated (Levene's p={homo_p:.4f})")

        # Check minimum cell sizes
        min_cell_n = min(len(g) for g in group_arrays) if group_arrays else 0
        if min_cell_n < self.MIN_PARAMETRIC_N:
            notes.append(f"Minimum cell size n={min_cell_n} is below threshold ({self.MIN_PARAMETRIC_N})")

        use_parametric = (
            normality_ok and
            homogeneity_ok and
            min_cell_n >= self.MIN_PARAMETRIC_N
        )

        if use_parametric:
            notes.insert(0, "Using parametric two-way ANOVA: assumptions met")
        else:
            notes.insert(0, "Using ART ANOVA (non-parametric): one or more assumptions violated")

        return use_parametric, "; ".join(notes)

    def _run_twoway_parametric(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        fa_name: str,
        fb_name: str
    ) -> TwoWayResult:
        """Run parametric two-way ANOVA using Type II sum of squares."""
        result = TwoWayResult(
            test_type=TestType.ANOVA_TWOWAY,
            factor_a_name=fa_name,
            factor_b_name=fb_name,
            alpha=self.alpha
        )

        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            from statsmodels.stats.anova import anova_lm

            # Sanitize column names for formula
            df = data.copy()
            safe_val = 'Y_value'
            safe_a = 'Factor_A'
            safe_b = 'Factor_B'
            df[safe_val] = df[value_col]
            df[safe_a] = df[factor_a_col]
            df[safe_b] = df[factor_b_col]

            # Fit the full model with interaction
            formula = f'{safe_val} ~ C({safe_a}) * C({safe_b})'
            model = ols(formula, data=df).fit()

            # Type II ANOVA table
            anova_tbl = anova_lm(model, typ=2)

            for idx in anova_tbl.index:
                idx_str = str(idx)
                if 'Factor_A' in idx_str and 'Factor_B' not in idx_str:
                    result.factor_a_stat = anova_tbl.loc[idx, 'F']
                    result.factor_a_pvalue = anova_tbl.loc[idx, 'PR(>F)']
                    result.factor_a_df = (anova_tbl.loc[idx, 'df'], anova_tbl.loc['Residual', 'df'])
                elif 'Factor_B' in idx_str and 'Factor_A' not in idx_str:
                    result.factor_b_stat = anova_tbl.loc[idx, 'F']
                    result.factor_b_pvalue = anova_tbl.loc[idx, 'PR(>F)']
                    result.factor_b_df = (anova_tbl.loc[idx, 'df'], anova_tbl.loc['Residual', 'df'])
                elif 'Factor_A' in idx_str and 'Factor_B' in idx_str:
                    result.interaction_stat = anova_tbl.loc[idx, 'F']
                    result.interaction_pvalue = anova_tbl.loc[idx, 'PR(>F)']
                    result.interaction_df = (anova_tbl.loc[idx, 'df'], anova_tbl.loc['Residual', 'df'])

            result.interaction_significant = result.interaction_pvalue < self.alpha

            # Calculate partial eta-squared
            ss_residual = anova_tbl.loc['Residual', 'sum_sq']
            for idx in anova_tbl.index:
                if idx == 'Residual':
                    continue
                idx_str = str(idx)
                ss_effect = anova_tbl.loc[idx, 'sum_sq']
                partial_eta_sq = ss_effect / (ss_effect + ss_residual) if (ss_effect + ss_residual) > 0 else 0

                if 'Factor_A' in idx_str and 'Factor_B' not in idx_str:
                    result.effect_sizes[fa_name] = round(partial_eta_sq, 4)
                elif 'Factor_B' in idx_str and 'Factor_A' not in idx_str:
                    result.effect_sizes[fb_name] = round(partial_eta_sq, 4)
                elif 'Factor_A' in idx_str and 'Factor_B' in idx_str:
                    result.effect_sizes[f'{fa_name}\u00d7{fb_name}'] = round(partial_eta_sq, 4)

            # Build clean ANOVA table for reporting
            clean_tbl = pd.DataFrame({
                'Source': [fa_name, fb_name, f'{fa_name}\u00d7{fb_name}', 'Residual'],
                'SS': [np.nan]*4, 'df': [np.nan]*4, 'MS': [np.nan]*4,
                'F': [np.nan]*4, 'p': [np.nan]*4, 'partial_\u03b7\u00b2': [np.nan]*4,
            })

            row_map = {}
            for idx in anova_tbl.index:
                idx_str = str(idx)
                if idx_str == 'Residual':
                    row_map[3] = idx
                elif 'Factor_A' in idx_str and 'Factor_B' not in idx_str:
                    row_map[0] = idx
                elif 'Factor_B' in idx_str and 'Factor_A' not in idx_str:
                    row_map[1] = idx
                elif 'Factor_A' in idx_str and 'Factor_B' in idx_str:
                    row_map[2] = idx

            for clean_row, sm_idx in row_map.items():
                clean_tbl.loc[clean_row, 'SS'] = round(anova_tbl.loc[sm_idx, 'sum_sq'], 4)
                clean_tbl.loc[clean_row, 'df'] = int(anova_tbl.loc[sm_idx, 'df'])
                ms = anova_tbl.loc[sm_idx, 'sum_sq'] / anova_tbl.loc[sm_idx, 'df'] if anova_tbl.loc[sm_idx, 'df'] > 0 else np.nan
                clean_tbl.loc[clean_row, 'MS'] = round(ms, 4)
                if 'F' in anova_tbl.columns and sm_idx != 'Residual':
                    clean_tbl.loc[clean_row, 'F'] = round(anova_tbl.loc[sm_idx, 'F'], 4)
                    clean_tbl.loc[clean_row, 'p'] = anova_tbl.loc[sm_idx, 'PR(>F)']
                if clean_row < 3:
                    source_name = clean_tbl.loc[clean_row, 'Source']
                    clean_tbl.loc[clean_row, 'partial_\u03b7\u00b2'] = result.effect_sizes.get(source_name, np.nan)

            result.anova_table = clean_tbl

        except ImportError:
            warnings.warn("statsmodels not available; falling back to ART ANOVA")
            return self._run_art_anova(data, value_col, factor_a_col, factor_b_col, fa_name, fb_name)
        except Exception as e:
            warnings.warn(f"Parametric two-way ANOVA failed: {e}; falling back to ART ANOVA")
            return self._run_art_anova(data, value_col, factor_a_col, factor_b_col, fa_name, fb_name)

        return result

    def _run_art_anova(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        fa_name: str,
        fb_name: str
    ) -> TwoWayResult:
        """
        Aligned Rank Transform (ART) ANOVA — non-parametric two-way test.

        ART procedure (Wobbrock et al., 2011):
        1. For each effect (A, B, AxB), strip out all OTHER effects from the data
        2. Rank the aligned (stripped) residuals
        3. Run standard ANOVA on the aligned ranks
        """
        result = TwoWayResult(
            test_type=TestType.ART_ANOVA,
            factor_a_name=fa_name,
            factor_b_name=fb_name,
            alpha=self.alpha
        )

        df = data.copy()
        Y = df[value_col].values.astype(float)

        levels_a = sorted(df[factor_a_col].unique())
        levels_b = sorted(df[factor_b_col].unique())

        grand_mean = np.mean(Y)

        marginal_a = {}
        for a in levels_a:
            mask = df[factor_a_col] == a
            marginal_a[a] = np.mean(Y[mask])

        marginal_b = {}
        for b in levels_b:
            mask = df[factor_b_col] == b
            marginal_b[b] = np.mean(Y[mask])

        cell_means = {}
        for a in levels_a:
            for b in levels_b:
                mask = (df[factor_a_col] == a) & (df[factor_b_col] == b)
                vals = Y[mask]
                cell_means[(a, b)] = np.mean(vals) if len(vals) > 0 else grand_mean

        n = len(Y)
        from scipy.stats import rankdata

        # === ART for Factor A ===
        aligned_a = np.zeros(n)
        for i in range(n):
            a_i = df.iloc[i][factor_a_col]
            b_i = df.iloc[i][factor_b_col]
            aligned_a[i] = Y[i] - (marginal_b[b_i] - grand_mean) - (cell_means[(a_i, b_i)] - marginal_a[a_i] - marginal_b[b_i] + grand_mean)

        ranks_a = rankdata(aligned_a)
        groups_a = [ranks_a[df[factor_a_col] == a] for a in levels_a]
        if all(len(g) > 0 for g in groups_a):
            f_a, p_a = stats.f_oneway(*groups_a)
            result.factor_a_stat = f_a
            result.factor_a_pvalue = p_a
            result.factor_a_df = (len(levels_a) - 1, n - len(levels_a) * len(levels_b))

        # === ART for Factor B ===
        aligned_b = np.zeros(n)
        for i in range(n):
            a_i = df.iloc[i][factor_a_col]
            b_i = df.iloc[i][factor_b_col]
            aligned_b[i] = Y[i] - (marginal_a[a_i] - grand_mean) - (cell_means[(a_i, b_i)] - marginal_a[a_i] - marginal_b[b_i] + grand_mean)

        ranks_b = rankdata(aligned_b)
        groups_b = [ranks_b[df[factor_b_col] == b] for b in levels_b]
        if all(len(g) > 0 for g in groups_b):
            f_b, p_b = stats.f_oneway(*groups_b)
            result.factor_b_stat = f_b
            result.factor_b_pvalue = p_b
            result.factor_b_df = (len(levels_b) - 1, n - len(levels_a) * len(levels_b))

        # === ART for Interaction AxB ===
        aligned_ab = np.zeros(n)
        for i in range(n):
            a_i = df.iloc[i][factor_a_col]
            b_i = df.iloc[i][factor_b_col]
            aligned_ab[i] = Y[i] - (marginal_a[a_i] - grand_mean) - (marginal_b[b_i] - grand_mean)

        ranks_ab = rankdata(aligned_ab)

        cell_labels = df[factor_a_col].astype(str) + '_' + df[factor_b_col].astype(str)
        unique_cells = sorted(cell_labels.unique())
        groups_ab = [ranks_ab[cell_labels == cell] for cell in unique_cells]
        if all(len(g) > 0 for g in groups_ab) and len(groups_ab) > 1:
            f_ab, p_ab = stats.f_oneway(*groups_ab)
            n_a = len(levels_a)
            n_b = len(levels_b)
            df_interaction = (n_a - 1) * (n_b - 1)
            df_residual = n - n_a * n_b

            try:
                import statsmodels.api as sm
                from statsmodels.formula.api import ols as sm_ols
                from statsmodels.stats.anova import anova_lm as sm_anova_lm

                art_df = pd.DataFrame({
                    'rank_ab': ranks_ab,
                    'A': df[factor_a_col].values,
                    'B': df[factor_b_col].values
                })
                model_ab = sm_ols('rank_ab ~ C(A) * C(B)', data=art_df).fit()
                art_table = sm_anova_lm(model_ab, typ=2)

                for idx in art_table.index:
                    idx_str = str(idx)
                    if ':' in idx_str or '*' in idx_str:
                        result.interaction_stat = art_table.loc[idx, 'F']
                        result.interaction_pvalue = art_table.loc[idx, 'PR(>F)']
                        result.interaction_df = (int(art_table.loc[idx, 'df']),
                                                  int(art_table.loc['Residual', 'df']))
                        break
            except (ImportError, Exception):
                result.interaction_stat = f_ab
                result.interaction_pvalue = p_ab
                result.interaction_df = (df_interaction, df_residual)

        result.interaction_significant = result.interaction_pvalue < self.alpha

        # Calculate approximate partial eta-squared from F and df
        for label, f_val, df_tup in [
            (fa_name, result.factor_a_stat, result.factor_a_df),
            (fb_name, result.factor_b_stat, result.factor_b_df),
            (f'{fa_name}\u00d7{fb_name}', result.interaction_stat, result.interaction_df)
        ]:
            denom = f_val * df_tup[0] + df_tup[1]
            if not np.isnan(f_val) and np.isfinite(f_val) and df_tup[1] > 0 and denom > 0:
                eta = (f_val * df_tup[0]) / denom
                result.effect_sizes[label] = round(max(0, eta), 4)

        # Build ANOVA table
        result.anova_table = pd.DataFrame({
            'Source': [fa_name, fb_name, f'{fa_name}\u00d7{fb_name}'],
            'F': [result.factor_a_stat, result.factor_b_stat, result.interaction_stat],
            'df_num': [result.factor_a_df[0], result.factor_b_df[0], result.interaction_df[0]],
            'df_den': [result.factor_a_df[1], result.factor_b_df[1], result.interaction_df[1]],
            'p': [result.factor_a_pvalue, result.factor_b_pvalue, result.interaction_pvalue],
            'partial_\u03b7\u00b2': [
                result.effect_sizes.get(fa_name, np.nan),
                result.effect_sizes.get(fb_name, np.nan),
                result.effect_sizes.get(f'{fa_name}\u00d7{fb_name}', np.nan),
            ]
        })

        return result

    def _run_twoway_posthoc(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        twoway_result: TwoWayResult
    ) -> TwoWayResult:
        """
        Run post-hoc tests for two-way ANOVA.

        Strategy depends on interaction significance:
        - Significant interaction -> simple main effects
        - No significant interaction -> main effect post-hocs
        """
        pairwise_rows = []

        if twoway_result.interaction_significant:
            twoway_result.posthoc_type = "simple_main_effects"

            # Compare Factor A levels within each level of Factor B
            for b_level in sorted(data[factor_b_col].unique()):
                subset = data[data[factor_b_col] == b_level]
                a_levels = sorted(subset[factor_a_col].unique())
                if len(a_levels) < 2:
                    continue
                for i in range(len(a_levels)):
                    for j in range(i + 1, len(a_levels)):
                        g1 = subset[subset[factor_a_col] == a_levels[i]][value_col].dropna()
                        g2 = subset[subset[factor_a_col] == a_levels[j]][value_col].dropna()
                        if len(g1) < 2 or len(g2) < 2:
                            continue
                        if twoway_result.test_type == TestType.ART_ANOVA:
                            stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                            test_used = 'Mann-Whitney U'
                        else:
                            stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                            test_used = "Welch's t-test"
                        pairwise_rows.append({
                            'comparison_type': f'{twoway_result.factor_a_name} within {twoway_result.factor_b_name}={b_level}',
                            'group1': f'{a_levels[i]}', 'group2': f'{a_levels[j]}',
                            'within_level': str(b_level),
                            'factor_compared': twoway_result.factor_a_name,
                            'statistic': stat, 'pvalue': p, 'test': test_used,
                        })

            # Compare Factor B levels within each level of Factor A
            for a_level in sorted(data[factor_a_col].unique()):
                subset = data[data[factor_a_col] == a_level]
                b_levels = sorted(subset[factor_b_col].unique())
                if len(b_levels) < 2:
                    continue
                for i in range(len(b_levels)):
                    for j in range(i + 1, len(b_levels)):
                        g1 = subset[subset[factor_b_col] == b_levels[i]][value_col].dropna()
                        g2 = subset[subset[factor_b_col] == b_levels[j]][value_col].dropna()
                        if len(g1) < 2 or len(g2) < 2:
                            continue
                        if twoway_result.test_type == TestType.ART_ANOVA:
                            stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                            test_used = 'Mann-Whitney U'
                        else:
                            stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                            test_used = "Welch's t-test"
                        pairwise_rows.append({
                            'comparison_type': f'{twoway_result.factor_b_name} within {twoway_result.factor_a_name}={a_level}',
                            'group1': f'{b_levels[i]}', 'group2': f'{b_levels[j]}',
                            'within_level': str(a_level),
                            'factor_compared': twoway_result.factor_b_name,
                            'statistic': stat, 'pvalue': p, 'test': test_used,
                        })
        else:
            twoway_result.posthoc_type = "main_effect_marginal"

            # Post-hoc for Factor A if significant
            if twoway_result.factor_a_pvalue < self.alpha:
                a_levels = sorted(data[factor_a_col].unique())
                for i in range(len(a_levels)):
                    for j in range(i + 1, len(a_levels)):
                        g1 = data[data[factor_a_col] == a_levels[i]][value_col].dropna()
                        g2 = data[data[factor_a_col] == a_levels[j]][value_col].dropna()
                        if len(g1) < 2 or len(g2) < 2:
                            continue
                        if twoway_result.test_type == TestType.ART_ANOVA:
                            stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                            test_used = 'Mann-Whitney U'
                        else:
                            stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                            test_used = "Welch's t-test"
                        pairwise_rows.append({
                            'comparison_type': f'{twoway_result.factor_a_name} (marginal)',
                            'group1': f'{a_levels[i]}', 'group2': f'{a_levels[j]}',
                            'within_level': 'all',
                            'factor_compared': twoway_result.factor_a_name,
                            'statistic': stat, 'pvalue': p, 'test': test_used,
                        })

            # Post-hoc for Factor B if significant
            if twoway_result.factor_b_pvalue < self.alpha:
                b_levels = sorted(data[factor_b_col].unique())
                for i in range(len(b_levels)):
                    for j in range(i + 1, len(b_levels)):
                        g1 = data[data[factor_b_col] == b_levels[i]][value_col].dropna()
                        g2 = data[data[factor_b_col] == b_levels[j]][value_col].dropna()
                        if len(g1) < 2 or len(g2) < 2:
                            continue
                        if twoway_result.test_type == TestType.ART_ANOVA:
                            stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                            test_used = 'Mann-Whitney U'
                        else:
                            stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                            test_used = "Welch's t-test"
                        pairwise_rows.append({
                            'comparison_type': f'{twoway_result.factor_b_name} (marginal)',
                            'group1': f'{b_levels[i]}', 'group2': f'{b_levels[j]}',
                            'within_level': 'all',
                            'factor_compared': twoway_result.factor_b_name,
                            'statistic': stat, 'pvalue': p, 'test': test_used,
                        })

        # Apply multiple comparison correction
        if pairwise_rows:
            posthoc_df = pd.DataFrame(pairwise_rows)
            posthoc_df['pvalue_adj'] = self._adjust_pvalues(posthoc_df['pvalue'].values)
            posthoc_df['significant'] = posthoc_df['pvalue_adj'] < self.alpha
            twoway_result.posthoc_results = posthoc_df

        return twoway_result

    def _get_twoway_descriptive_stats(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str
    ) -> pd.DataFrame:
        """Calculate descriptive statistics for each cell of the two-way design."""
        rows = []
        for (a, b), group_df in data.groupby([factor_a_col, factor_b_col]):
            vals = group_df[value_col].dropna()
            rows.append({
                'factor_a': a, 'factor_b': b, 'n': len(vals),
                'mean': vals.mean(), 'std': vals.std(),
                'sem': vals.sem() if len(vals) > 1 else np.nan,
                'median': vals.median(), 'min': vals.min(), 'max': vals.max(),
            })

        # Marginal means for Factor A
        for a, group_df in data.groupby(factor_a_col):
            vals = group_df[value_col].dropna()
            rows.append({
                'factor_a': a, 'factor_b': '__MARGINAL__', 'n': len(vals),
                'mean': vals.mean(), 'std': vals.std(),
                'sem': vals.sem() if len(vals) > 1 else np.nan,
                'median': vals.median(), 'min': vals.min(), 'max': vals.max(),
            })

        # Marginal means for Factor B
        for b, group_df in data.groupby(factor_b_col):
            vals = group_df[value_col].dropna()
            rows.append({
                'factor_a': '__MARGINAL__', 'factor_b': b, 'n': len(vals),
                'mean': vals.mean(), 'std': vals.std(),
                'sem': vals.sem() if len(vals) > 1 else np.nan,
                'median': vals.median(), 'min': vals.min(), 'max': vals.max(),
            })

        return pd.DataFrame(rows).round(4)


def format_analysis_report(result: FullAnalysisResult) -> str:
    """Generate a human-readable report from analysis results."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"STATISTICAL ANALYSIS: {result.variable_name}")
    lines.append(f"{'='*60}")
    
    # Assumptions
    lines.append("\n--- ASSUMPTION TESTS ---")
    lines.append(f"Normality (α={result.assumptions.alpha}):")
    for group, passed in result.assumptions.normality_passed.items():
        p = result.assumptions.normality_pvalues.get(group, np.nan)
        status = "✓ PASSED" if passed else "✗ FAILED"
        lines.append(f"  {group}: p={p:.4f} {status}")
    
    lines.append(f"\nHomoscedasticity (Levene's test):")
    lines.append(f"  p={result.assumptions.homoscedasticity_pvalue:.4f} "
                f"{'✓ PASSED' if result.assumptions.homoscedasticity_passed else '✗ FAILED'}")
    
    lines.append(f"\nRecommended test: {result.assumptions.recommended_test.value}")
    lines.append(f"Reason: {result.assumptions.recommendation_reason}")
    
    # Main test
    lines.append("\n--- MAIN TEST RESULTS ---")
    lines.append(f"Test: {result.main_test.test_type.value}")
    lines.append(f"Statistic: {result.main_test.statistic:.4f}")
    lines.append(f"P-value: {result.main_test.pvalue:.4f}")
    lines.append(f"Significant (α={result.main_test.alpha}): "
                f"{'YES' if result.main_test.significant else 'NO'}")
    
    if result.main_test.effect_size is not None:
        lines.append(f"\nEffect size ({result.main_test.effect_size_type}): "
                    f"{result.main_test.effect_size:.4f} ({result.main_test.effect_size_interpretation})")
    
    # Descriptive stats
    if result.descriptive_stats is not None:
        lines.append("\n--- DESCRIPTIVE STATISTICS ---")
        lines.append(result.descriptive_stats.to_string())
    
    # Post-hoc
    if result.posthoc_test is not None:
        lines.append("\n--- POST-HOC COMPARISONS ---")
        lines.append(f"Test: {result.posthoc_test.test_type.value}")
        lines.append(result.posthoc_test.pairwise_results.to_string(index=False))
    
    return "\n".join(lines)


def format_twoway_apa(result: FullTwoWayAnalysisResult) -> str:
    """
    Generate APA-formatted text for two-way ANOVA results.

    Example output:
    "A two-way ANOVA revealed a significant main effect of Age,
    F(1,36) = 12.4, p = .001, partial η² = .26, ..."
    """
    tw = result.twoway_result
    test_name = "two-way ANOVA" if tw.test_type == TestType.ANOVA_TWOWAY else "ART ANOVA"

    parts = [f"A {test_name} was conducted to examine the effects of "
             f"{tw.factor_a_name} and {tw.factor_b_name} on {result.variable_name}."]

    def _format_effect(name, f_val, p_val, df_tup, eta):
        """Format a single effect in APA style."""
        if np.isnan(f_val):
            return f"The effect of {name} could not be computed."

        sig = "significant" if p_val < tw.alpha else "not significant"
        p_str = f"p < .001" if p_val < 0.001 else f"p = {p_val:.3f}"
        eta_str = f", partial η² = {eta:.2f}" if eta and not np.isnan(eta) else ""

        df1 = int(df_tup[0]) if not np.isnan(df_tup[0]) else '?'
        df2 = int(df_tup[1]) if not np.isnan(df_tup[1]) else '?'

        return (f"The main effect of {name} was {sig}, "
                f"F({df1},{df2}) = {f_val:.2f}, {p_str}{eta_str}.")

    # Main effects
    eta_a = tw.effect_sizes.get(tw.factor_a_name, np.nan)
    parts.append(_format_effect(tw.factor_a_name, tw.factor_a_stat,
                                tw.factor_a_pvalue, tw.factor_a_df, eta_a))

    eta_b = tw.effect_sizes.get(tw.factor_b_name, np.nan)
    parts.append(_format_effect(tw.factor_b_name, tw.factor_b_stat,
                                tw.factor_b_pvalue, tw.factor_b_df, eta_b))

    # Interaction
    eta_int = tw.effect_sizes.get(f'{tw.factor_a_name}×{tw.factor_b_name}', np.nan)
    int_sig = "significant" if tw.interaction_significant else "not significant"
    int_p = f"p < .001" if tw.interaction_pvalue < 0.001 else f"p = {tw.interaction_pvalue:.3f}"
    int_eta = f", partial η² = {eta_int:.2f}" if eta_int and not np.isnan(eta_int) else ""
    df1_int = int(tw.interaction_df[0]) if not np.isnan(tw.interaction_df[0]) else '?'
    df2_int = int(tw.interaction_df[1]) if not np.isnan(tw.interaction_df[1]) else '?'

    parts.append(
        f"The {tw.factor_a_name} × {tw.factor_b_name} interaction was {int_sig}, "
        f"F({df1_int},{df2_int}) = {tw.interaction_stat:.2f}, {int_p}{int_eta}."
    )

    # Post-hoc summary
    if tw.posthoc_results is not None and len(tw.posthoc_results) > 0:
        sig_comparisons = tw.posthoc_results[tw.posthoc_results['significant']]
        if len(sig_comparisons) > 0:
            if tw.posthoc_type == "simple_main_effects":
                parts.append(f"Simple main effects analysis revealed {len(sig_comparisons)} "
                           f"significant pairwise comparison(s) (corrected for multiple comparisons).")
            else:
                parts.append(f"Post-hoc pairwise comparisons on marginal means revealed "
                           f"{len(sig_comparisons)} significant difference(s) "
                           f"(corrected for multiple comparisons).")

    return " ".join(parts)


def format_twoway_report(result: FullTwoWayAnalysisResult) -> str:
    """Generate a detailed human-readable report for two-way ANOVA results."""
    lines = []
    tw = result.twoway_result

    lines.append(f"{'='*60}")
    lines.append(f"TWO-WAY ANOVA: {result.variable_name}")
    lines.append(f"{'='*60}")
    lines.append(f"Test: {tw.test_type.value}")
    lines.append(f"Factors: {tw.factor_a_name} × {tw.factor_b_name}")

    if result.assumption_notes:
        lines.append(f"\n--- ASSUMPTIONS ---")
        lines.append(result.assumption_notes)

    lines.append(f"\n--- ANOVA TABLE ---")
    if tw.anova_table is not None:
        lines.append(tw.anova_table.to_string(index=False))

    lines.append(f"\n--- SIGNIFICANCE SUMMARY ---")
    def _sig_stars(p):
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return 'n.s.'

    lines.append(f"  {tw.factor_a_name}: p={tw.factor_a_pvalue:.4f} {_sig_stars(tw.factor_a_pvalue)}")
    lines.append(f"  {tw.factor_b_name}: p={tw.factor_b_pvalue:.4f} {_sig_stars(tw.factor_b_pvalue)}")
    lines.append(f"  {tw.factor_a_name}×{tw.factor_b_name}: p={tw.interaction_pvalue:.4f} {_sig_stars(tw.interaction_pvalue)}")

    if result.descriptive_stats is not None:
        cell_stats = result.descriptive_stats[result.descriptive_stats['factor_b'] != '__MARGINAL__']
        cell_stats = cell_stats[cell_stats['factor_a'] != '__MARGINAL__']
        lines.append(f"\n--- CELL DESCRIPTIVE STATISTICS ---")
        lines.append(cell_stats.to_string(index=False))

    if tw.posthoc_results is not None:
        lines.append(f"\n--- POST-HOC ({tw.posthoc_type}) ---")
        lines.append(tw.posthoc_results.to_string(index=False))

    lines.append(f"\n--- APA SUMMARY ---")
    lines.append(format_twoway_apa(result))

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    
    # Create sample data
    demo_data = pd.DataFrame({
        'concentration': np.concatenate([
            np.random.normal(100, 20, 15),  # Group A
            np.random.normal(120, 25, 15),  # Group B
            np.random.normal(90, 18, 15),   # Group C
        ]),
        'group': ['A']*15 + ['B']*15 + ['C']*15
    })
    
    # Run analysis
    analyzer = StatisticalAnalyzer(alpha=0.05)
    results = analyzer.analyze(demo_data, 'concentration', 'group')
    
    # Print report
    print(format_analysis_report(results))
