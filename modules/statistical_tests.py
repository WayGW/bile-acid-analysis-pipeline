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


class TestType(Enum):
    """Types of statistical tests available."""
    # Two-group tests
    TTEST_INDEPENDENT = "independent_t_test"
    TTEST_WELCH = "welch_t_test"
    MANN_WHITNEY = "mann_whitney_u"
    
    # Multi-group tests
    ANOVA_ONEWAY = "one_way_anova"
    ANOVA_WELCH = "welch_anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    
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
