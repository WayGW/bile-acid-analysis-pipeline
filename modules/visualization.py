"""
Visualization Module for Bile Acid Analysis
=============================================

Generates publication-quality figures including:
1. Bar plots with error bars (group comparisons)
2. Stacked bar plots (composition)
3. Heatmaps (correlation, group patterns)
4. Box/violin plots
5. Pie charts (percentage breakdown)
6. Ratio comparison plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palettes
CONJUGATION_COLORS = {
    'unconjugated': '#1f77b4',
    'glycine': '#2ca02c',
    'taurine': '#d62728',
    'sulfated': '#9467bd',
    'n-acyl': '#8c564b',
}

ORIGIN_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
}

# Default palette - can be overridden
DEFAULT_PALETTE = "Set2"


def get_group_palette(palette_name: str = "Set2", n_colors: int = 10):
    """Get a color palette by name."""
    import seaborn as sns
    return sns.color_palette(palette_name, n_colors)


class BileAcidVisualizer:
    """Generate visualizations for bile acid data analysis."""
    
    def __init__(
        self,
        figsize_single: Tuple[float, float] = (8, 6),
        figsize_wide: Tuple[float, float] = (12, 6),
        figsize_tall: Tuple[float, float] = (8, 10),
        color_palette: str = "Set2",
        style: str = "whitegrid"
    ):
        self.figsize_single = figsize_single
        self.figsize_wide = figsize_wide
        self.figsize_tall = figsize_tall
        self.palette_name = color_palette
        self.group_palette = get_group_palette(color_palette, 10)
        sns.set_style(style)
        sns.set_palette(color_palette)
    
    def plot_group_comparison(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        plot_type: str = "bar",
        show_points: bool = True,
        significance_pairs: Optional[List[Tuple[str, str, str]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        colors: Optional[Dict[str, str]] = None,
        ax: Optional[plt.Axes] = None,
        log_scale: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create group comparison plot."""
        # Filter out NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        if figsize is None:
            figsize = self.figsize_single
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Get valid groups only
        groups = [g for g in data[group_col].unique() if pd.notna(g)]
        palette = colors if colors else dict(zip(groups, self.group_palette[:len(groups)]))
        
        # Prepare plot data - apply log10 transform if requested
        plot_col = value_col
        plot_ylabel = ylabel or value_col
        
        if log_scale:
            log_col = f'{value_col}_log10'
            min_positive = data[value_col][data[value_col] > 0].min()
            floor_val = min_positive / 10 if pd.notna(min_positive) else 0.001
            data[log_col] = np.log10(data[value_col].clip(lower=floor_val))
            plot_col = log_col
            plot_ylabel = f'log₁₀({ylabel or value_col})'
        
        if plot_type == "bar":
            means = data.groupby(group_col)[plot_col].mean()
            sems = data.groupby(group_col)[plot_col].sem()
            
            x = range(len(groups))
            bars = ax.bar(x, [means[g] for g in groups], 
                         yerr=[sems[g] for g in groups],
                         capsize=5, color=[palette.get(g, '#888888') for g in groups],
                         edgecolor='black', linewidth=0.5, alpha=0.8)
            
            if show_points:
                for i, g in enumerate(groups):
                    group_data = data[data[group_col] == g][plot_col]
                    jitter = np.random.normal(0, 0.05, len(group_data))
                    ax.scatter(i + jitter, group_data, color='black', 
                              alpha=0.5, s=20, zorder=3)
            
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            
        elif plot_type == "box":
            sns.boxplot(data=data, x=group_col, y=plot_col, ax=ax,
                       hue=group_col, palette=palette, order=groups, legend=False)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                            color='black', alpha=0.5, size=4, order=groups)
                
        elif plot_type == "violin":
            sns.violinplot(data=data, x=group_col, y=plot_col, ax=ax,
                          hue=group_col, palette=palette, order=groups,
                          inner='box', legend=False, cut=0)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                            color='black', alpha=0.5, size=4, order=groups)
        
        if significance_pairs:
            self._add_significance_bars(ax, groups, significance_pairs, data, plot_col, group_col)
        
        ax.set_xlabel('')
        ax.set_ylabel(plot_ylabel)
        ax.set_title(title or f'{value_col} by {group_col}')
        
        return fig, ax
    
    def _add_significance_bars(self, ax, groups, pairs, data, value_col, group_col):
        """Add significance bars to plot."""
        y_max = data[value_col].max()
        y_range = data[value_col].max() - data[value_col].min()
        bar_height = y_range * 0.05
        
        group_idx = {g: i for i, g in enumerate(groups)}
        
        for i, (g1, g2, annotation) in enumerate(pairs):
            if g1 not in group_idx or g2 not in group_idx:
                continue
            
            x1, x2 = group_idx[g1], group_idx[g2]
            y = y_max + (i + 1) * bar_height * 2
            
            ax.plot([x1, x1, x2, x2], [y - bar_height/2, y, y, y - bar_height/2],
                   color='black', linewidth=1)
            ax.text((x1 + x2) / 2, y + bar_height/4, annotation,
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(top=y_max + (len(pairs) + 1) * bar_height * 2.5)
    
    def plot_composition_stacked(
        self,
        data: pd.DataFrame,
        group_col: str,
        bile_acid_cols: List[str],
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        as_percentage: bool = True,
        legend_outside: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create stacked bar plot showing bile acid composition."""
        # Filter out NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        if figsize is None:
            figsize = self.figsize_wide
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        available_cols = [c for c in bile_acid_cols if c in data.columns]
        
        if not available_cols:
            ax.text(0.5, 0.5, 'No bile acid data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig, ax
        
        group_means = data.groupby(group_col)[available_cols].mean()
        
        if as_percentage:
            row_sums = group_means.sum(axis=1)
            # Avoid division by zero
            row_sums = row_sums.replace(0, np.nan)
            group_means = group_means.div(row_sums, axis=0) * 100
            group_means = group_means.fillna(0)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(available_cols)))
        group_means.plot(kind='bar', stacked=True, ax=ax, color=colors,
                        edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('')
        ax.set_ylabel('Percentage (%)' if as_percentage else 'Concentration')
        ax.set_title(title or 'Bile Acid Composition by Group')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        if legend_outside:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                     borderaxespad=0, fontsize=8)
        else:
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: Optional[str] = None,
        method: str = 'spearman',
        figsize: Optional[Tuple[float, float]] = None,
        annot: bool = True,
        mask_upper: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create correlation heatmap."""
        if figsize is None:
            figsize = self.figsize_single
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        corr = data[columns].corr(method=method)
        
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                   annot=annot, fmt='.2f', square=True, ax=ax,
                   cbar_kws={'shrink': 0.8},
                   annot_kws={'size': 8} if annot else {})
        
        ax.set_title(title or f'{method.capitalize()} Correlation Matrix')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_ratio_comparison(
        self,
        data: pd.DataFrame,
        ratio_col: str,
        group_col: str,
        title: Optional[str] = None,
        log_scale: bool = False,
        reference_line: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        plot_type: str = "violin",
        show_points: bool = True,
        significance_pairs: Optional[List[Tuple[str, str, str]]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create comparison plot for a bile acid ratio.
        
        Args:
            data: DataFrame with ratio and group columns
            ratio_col: Column name for the ratio values
            group_col: Column name for grouping
            title: Plot title
            log_scale: Whether to log10 transform the data
            reference_line: Y value for reference line (default None)
            figsize: Figure size
            plot_type: 'violin', 'box', or 'bar'
            show_points: Whether to show individual data points
            significance_pairs: List of (group1, group2, annotation) for significance brackets
        """
        if figsize is None:
            figsize = self.figsize_single
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        groups = [g for g in data[group_col].unique() if pd.notna(g)]
        colors = dict(zip(groups, self.group_palette[:len(groups)]))
        
        # Prepare plot data - apply log10 transform if requested
        plot_col = ratio_col
        ylabel = ratio_col
        
        if log_scale:
            # Create log10 transformed column, handling zeros
            log_col = f'{ratio_col}_log10'
            min_positive = data[ratio_col][data[ratio_col] > 0].min()
            floor_val = min_positive / 10 if pd.notna(min_positive) else 0.001
            data[log_col] = np.log10(data[ratio_col].clip(lower=floor_val))
            plot_col = log_col
            ylabel = f'log₁₀({ratio_col})'
        
        # Plot based on type
        if plot_type == "violin":
            sns.violinplot(data=data, x=group_col, y=plot_col, ax=ax,
                          hue=group_col, palette=colors, order=groups,
                          inner=None, alpha=0.3, legend=False, cut=0)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                             hue=group_col, palette=colors, order=groups, 
                             size=6, alpha=0.7, legend=False)
            # Add mean lines
            means = data.groupby(group_col)[plot_col].mean()
            for i, g in enumerate(groups):
                if g in means.index:
                    ax.hlines(means[g], i-0.3, i+0.3, color='black', linewidth=2)
                    
        elif plot_type == "box":
            sns.boxplot(data=data, x=group_col, y=plot_col, ax=ax,
                       hue=group_col, palette=colors, order=groups, legend=False)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                             hue=group_col, palette=colors, order=groups,
                             size=4, alpha=0.6, legend=False)
                             
        elif plot_type == "bar":
            sns.barplot(data=data, x=group_col, y=plot_col, ax=ax,
                       hue=group_col, palette=colors, order=groups, 
                       capsize=0.1, legend=False)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                             hue=group_col, palette=colors, order=groups,
                             size=4, alpha=0.6, legend=False)
        
        # Add reference line (only makes sense if not log transformed)
        if reference_line is not None and not log_scale:
            ax.axhline(y=reference_line, color='gray', linestyle='--', 
                      linewidth=1, label=f'Reference ({reference_line})')
            ax.legend()
        
        # Add significance brackets if provided
        if significance_pairs:
            self._add_significance_bars(ax, groups, significance_pairs, data, plot_col, group_col)
        
        ax.set_xlabel('')
        ax.set_ylabel(ylabel)
        ax.set_title(title or f'{ratio_col} by {group_col}')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_multi_panel_groups(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        group_col: str,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        plot_type: str = "box",
        sharey: bool = False,
        ylabel: str = 'Concentration'
    ) -> plt.Figure:
        """Create multi-panel figure comparing groups across multiple variables."""
        # Filter out NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        n_plots = len(value_cols)
        nrows = int(np.ceil(n_plots / ncols))

        if figsize is None:
            col_width = 4
            row_height = 4.5 if plot_type == "box" else 5
            figsize = (ncols * col_width, nrows * row_height)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
        axes = axes.flatten() if n_plots > 1 else [axes]

        # Get valid groups only (no NaN)
        groups = [g for g in data[group_col].unique() if pd.notna(g)]
        palette = dict(zip(groups, self.group_palette[:len(groups)]))
        
        for i, col in enumerate(value_cols):
            if col not in data.columns:
                continue
            
            if plot_type == "box":
                sns.boxplot(data=data, x=group_col, y=col, ax=axes[i],
                          hue=group_col, palette=palette, order=groups, legend=False)
            elif plot_type == "violin":
                sns.violinplot(data=data, x=group_col, y=col, ax=axes[i],
                             hue=group_col, palette=palette, order=groups,
                             inner='box', legend=False, cut=0)
            elif plot_type == "bar":
                means = data.groupby(group_col)[col].mean()
                sems = data.groupby(group_col)[col].sem()
                x = range(len(groups))
                axes[i].bar(x, [means[g] for g in groups],
                          yerr=[sems[g] for g in groups],
                          capsize=3, color=[palette[g] for g in groups],
                          edgecolor='black', linewidth=0.5)
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(groups)
            
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel('')
            if i % ncols == 0:
                axes[i].set_ylabel(ylabel)
            else:
                axes[i].set_ylabel('')
            axes[i].tick_params(axis='x', rotation=45)

        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        fig.subplots_adjust(hspace=0.45, wspace=0.35)
        return fig

    def plot_multi_panel_groups_with_stats(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        group_col: str,
        stats_results: Dict,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        plot_type: str = "box",
        sharey: bool = False,
        max_sig_bars: int = 3,
        log_scale: bool = False,
        show_points: bool = True,
        ylabel: str = 'Concentration'
    ) -> plt.Figure:
        """
        Create multi-panel figure with significance annotations.
        
        Args:
            data: DataFrame with data
            value_cols: Columns to plot
            group_col: Grouping column
            stats_results: Dict of statistical results keyed by column name
            ncols: Number of columns in grid
            figsize: Figure size
            plot_type: 'box', 'violin', or 'bar'
            sharey: Share y-axis across panels
            max_sig_bars: Maximum significance bars to show per panel
            log_scale: Whether to log10 transform the data for plotting
            show_points: Whether to overlay individual data points
        """
        # Filter out NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        n_plots = len(value_cols)
        nrows = int(np.ceil(n_plots / ncols))
        
        if figsize is None:
            col_width = 4
            row_height = 4.5 if plot_type == "box" else 5
            figsize = (ncols * col_width, nrows * row_height)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        # Get valid groups only
        groups = [g for g in data[group_col].unique() if pd.notna(g)]
        palette = dict(zip(groups, self.group_palette[:len(groups)]))
        
        for i, col in enumerate(value_cols):
            if col not in data.columns:
                continue
            
            ax = axes[i]
            
            # Prepare plot data - apply log10 transform if requested
            plot_data = data.copy()
            plot_col = col
            ax_ylabel = ylabel

            if log_scale:
                # Create log10 transformed column, handling zeros
                log_col = f'{col}_log10'
                # Replace zeros/negatives with small positive value before log
                min_positive = plot_data[col][plot_data[col] > 0].min()
                floor_val = min_positive / 10 if pd.notna(min_positive) else 0.001
                plot_data[log_col] = np.log10(plot_data[col].clip(lower=floor_val))
                plot_col = log_col
                ax_ylabel = f'log₁₀({ylabel})'
            
            if plot_type == "box":
                sns.boxplot(data=plot_data, x=group_col, y=plot_col, ax=ax,
                          hue=group_col, palette=palette, order=groups, legend=False)
                if show_points:
                    sns.stripplot(data=plot_data, x=group_col, y=plot_col, ax=ax,
                                color='black', alpha=0.7, size=6, order=groups, 
                                jitter=0.2, zorder=10, edgecolor='white', linewidth=0.5)
            elif plot_type == "violin":
                sns.violinplot(data=plot_data, x=group_col, y=plot_col, ax=ax,
                             hue=group_col, palette=palette, order=groups,
                             inner=None, legend=False, cut=0)
                if show_points:
                    sns.stripplot(data=plot_data, x=group_col, y=plot_col, ax=ax,
                                color='black', alpha=0.7, size=6, order=groups,
                                jitter=0.15, zorder=10, edgecolor='white', linewidth=0.5)
            elif plot_type == "bar":
                means = plot_data.groupby(group_col)[plot_col].mean()
                sems = plot_data.groupby(group_col)[plot_col].sem()
                x = range(len(groups))
                ax.bar(x, [means[g] for g in groups],
                      yerr=[sems[g] for g in groups],
                      capsize=3, color=[palette[g] for g in groups],
                      edgecolor='black', linewidth=0.5)
                if show_points:
                    for j, g in enumerate(groups):
                        group_vals = plot_data[plot_data[group_col] == g][plot_col]
                        jitter = np.random.normal(0, 0.1, len(group_vals))
                        ax.scatter(j + jitter, group_vals, color='black', alpha=0.7, s=30, 
                                  zorder=10, edgecolors='white', linewidths=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(groups)
            
            # Add significance annotations (use original data for stats)
            if col in stats_results:
                result = stats_results[col]
                sig_pairs = self._get_significant_pairs(result, max_pairs=max_sig_bars)
                if sig_pairs:
                    self._add_significance_bars(ax, groups, sig_pairs, plot_data, plot_col, group_col)
                
                # Add p-value to title
                p = result.main_test.pvalue
                sig_marker = '*' if result.main_test.significant else ''
                ax.set_title(f"{col} (p={p:.3f}){sig_marker}", fontsize=10)
            else:
                ax.set_title(col, fontsize=10)
            
            ax.set_xlabel('')
            ax.set_ylabel(ax_ylabel)
            ax.tick_params(axis='x', rotation=45)

        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        fig.subplots_adjust(hspace=0.45, wspace=0.35)
        return fig

    def _get_significant_pairs(self, result, max_pairs: int = 3) -> List[Tuple[str, str, str]]:
        """Extract significant pairs from statistical result."""
        if not result.posthoc_test or result.posthoc_test.pairwise_results is None:
            return []
        
        pairs = []
        posthoc_df = result.posthoc_test.pairwise_results
        sig_rows = posthoc_df[posthoc_df['significant']].copy()
        
        # Sort by p-value to get most significant first
        p_col = 'pvalue_adj' if 'pvalue_adj' in sig_rows.columns else 'pvalue'
        sig_rows = sig_rows.sort_values(p_col)
        
        for _, row in sig_rows.head(max_pairs).iterrows():
            p = row.get('pvalue_adj', row.get('pvalue', 1.0))
            if p < 0.001:
                annot = '***'
            elif p < 0.01:
                annot = '**'
            elif p < 0.05:
                annot = '*'
            else:
                continue
            pairs.append((row['group1'], row['group2'], annot))
        
        return pairs
    
    def plot_grouped_comparison_with_stats(
        self,
        data: pd.DataFrame,
        group_col: str,
        value_cols: List[str],
        value_labels: List[str],
        stats_results: Dict,
        title: str = '',
        ylabel: str = 'Concentration',
        plot_type: str = 'bar',
        figsize: Optional[Tuple[float, float]] = None,
        log_scale: bool = False
    ) -> Tuple[plt.Figure, plt.Axes, List[str]]:
        """
        Create grouped comparison plot with significance annotations for each value type.
        
        Shows multiple value columns (e.g., Primary/Secondary) grouped by
        the group column, with significance brackets comparing groups within each type.
        
        Args:
            data: DataFrame with group column and value columns
            group_col: Column name for grouping (x-axis groups)
            value_cols: List of column names to plot
            value_labels: Display labels for each value column
            stats_results: Dict mapping value_col names to StatisticalResult objects
            title: Plot title
            ylabel: Y-axis label
            plot_type: 'bar', 'box', or 'violin'
            figsize: Figure size
            log_scale: Whether to log10 transform the data
            
        Returns:
            Tuple of (figure, axes, stats_text_list)
        """
        if figsize is None:
            figsize = (10, 7)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        # Apply log10 transform if requested
        plot_ylabel = ylabel
        if log_scale:
            for col in value_cols:
                min_positive = data[col][data[col] > 0].min()
                floor_val = min_positive / 10 if pd.notna(min_positive) else 0.001
                data[col] = np.log10(data[col].clip(lower=floor_val))
            plot_ylabel = f'log₁₀({ylabel})'
        
        groups = data[group_col].unique().tolist()
        n_groups = len(groups)
        n_types = len(value_cols)
        
        # Colors for each type
        type_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'][:n_types]
        
        # Reshape data for seaborn
        # Rename columns to avoid conflicts with 'Type' var_name
        plot_data = data[[group_col] + value_cols].copy()
        plot_data = plot_data.rename(columns={group_col: '_Group_'})
        plot_data = plot_data.melt(
            id_vars='_Group_', 
            value_vars=value_cols,
            var_name='_Type_', 
            value_name='Value'
        )
        # Replace column names with labels
        label_map = dict(zip(value_cols, value_labels))
        plot_data['_Type_'] = plot_data['_Type_'].replace(label_map)
        
        # Create palette mapping labels to colors
        palette = dict(zip(value_labels, type_colors))
        
        # Plot based on type
        if plot_type == 'box':
            sns.boxplot(data=plot_data, x='_Group_', y='Value', hue='_Type_',
                       ax=ax, palette=palette, order=groups)
        elif plot_type == 'violin':
            sns.violinplot(data=plot_data, x='_Group_', y='Value', hue='_Type_',
                          ax=ax, palette=palette, order=groups, inner='box', cut=0)
        else:  # bar
            sns.barplot(data=plot_data, x='_Group_', y='Value', hue='_Type_',
                       ax=ax, palette=palette, order=groups, capsize=0.1)
        
        ax.set_xlabel('')
        ax.set_ylabel(plot_ylabel)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Type', loc='upper right')
        
        # Add significance annotations for each type
        y_max = ax.get_ylim()[1]
        annotation_y_start = y_max * 1.02
        bar_height = y_max * 0.04
        
        stats_text_parts = []
        
        # Calculate x positions for each type within groups
        # For grouped plots, positions depend on number of types
        x = np.arange(n_groups)
        width = 0.8 / n_types
        
        for type_idx, (col, label) in enumerate(zip(value_cols, value_labels)):
            if col not in stats_results:
                continue
            
            result = stats_results[col]
            sig_pairs = self._get_significant_pairs(result, max_pairs=2)
            
            # Calculate offset for this type
            offset = (type_idx - n_types/2 + 0.5) * width
            
            # Add significance brackets
            for pair_idx, (g1, g2, annot) in enumerate(sig_pairs):
                if g1 in groups and g2 in groups:
                    idx1, idx2 = groups.index(g1), groups.index(g2)
                    x1, x2 = x[idx1] + offset, x[idx2] + offset
                    
                    # Stagger heights for multiple comparisons
                    y = annotation_y_start + (type_idx * 2 + pair_idx) * bar_height * 1.5
                    
                    # Draw bracket
                    ax.plot([x1, x1, x2, x2], [y, y + bar_height*0.3, y + bar_height*0.3, y],
                           color=type_colors[type_idx], linewidth=1.5)
                    ax.text((x1 + x2) / 2, y + bar_height*0.4, annot,
                           ha='center', va='bottom', fontsize=10, fontweight='bold',
                           color=type_colors[type_idx])
            
            # Build stats text
            test_name = result.main_test.test_type.value
            p_val = result.main_test.pvalue
            sig_marker = "âœ“" if result.main_test.significant else ""
            stats_text_parts.append(f"{label}: {test_name} p={p_val:.4f} {sig_marker}")
        
        # Adjust y limit to accommodate annotations
        new_y_max = annotation_y_start + (n_types * 2 + 1) * bar_height * 1.5
        ax.set_ylim(top=new_y_max)
        
        plt.tight_layout()
        
        return fig, ax, stats_text_parts
    
    # Keep old name as alias for backwards compatibility
    def plot_grouped_bar_with_stats(self, *args, **kwargs):
        """Alias for plot_grouped_comparison_with_stats (backwards compatibility)."""
        return self.plot_grouped_comparison_with_stats(*args, **kwargs)
    
    def plot_composition_pie_charts(
        self,
        data: pd.DataFrame,
        group_col: str,
        pct_cols: List[str],
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        top_n: int = 10,
        other_threshold: float = 2.0
    ) -> plt.Figure:
        """
        Create pie charts showing bile acid pool composition for each group.
        
        Args:
            data: DataFrame with group column and percentage columns
            group_col: Column name for grouping
            pct_cols: List of percentage column names (with _pct suffix)
            title: Overall figure title
            figsize: Figure size (auto-calculated if None)
            top_n: Maximum number of slices to show (rest grouped as "Other")
            other_threshold: Minimum percentage to show individually (below this goes to "Other")
            
        Returns:
            Matplotlib figure with pie charts
        """
        # Filter NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.lower() != 'nan']
        
        groups = data[group_col].unique().tolist()
        n_groups = len(groups)
        
        # Calculate figure size
        if figsize is None:
            figsize = (5 * n_groups, 5)
        
        fig, axes = plt.subplots(1, n_groups, figsize=figsize)
        if n_groups == 1:
            axes = [axes]
        
        # Use a nice color palette
        all_bas = [col.replace('_pct', '') for col in pct_cols]
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(all_bas) + 1, 20)))
        color_map = dict(zip(all_bas + ['Other'], colors))
        
        for idx, (group, ax) in enumerate(zip(groups, axes)):
            group_data = data[data[group_col] == group][pct_cols].mean()
            
            # Clean up names (remove _pct suffix)
            group_data.index = [col.replace('_pct', '') for col in group_data.index]
            
            # Sort by value and handle small slices
            group_data = group_data.sort_values(ascending=False)
            
            # Group small values into "Other"
            main_slices = group_data[group_data >= other_threshold].head(top_n)
            other_value = group_data.sum() - main_slices.sum()
            
            if other_value > 0.1:  # Only add "Other" if meaningful
                plot_data = pd.concat([main_slices, pd.Series({'Other': other_value})])
            else:
                plot_data = main_slices
            
            # Get colors for this pie
            pie_colors = [color_map.get(ba, '#999999') for ba in plot_data.index]
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                plot_data.values,
                labels=None,  # We'll add a legend instead
                autopct=lambda pct: f'{pct:.1f}%' if pct >= 3 else '',
                colors=pie_colors,
                pctdistance=0.75,
                startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            
            # Style the percentage labels
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_fontweight('bold')
            
            ax.set_title(f'{group}', fontsize=12, fontweight='bold')
        
        # Add a single legend for the whole figure
        # Get unique labels across all pies
        all_labels = set()
        for group in groups:
            group_data = data[data[group_col] == group][pct_cols].mean()
            group_data.index = [col.replace('_pct', '') for col in group_data.index]
            group_data = group_data.sort_values(ascending=False)
            main_slices = group_data[group_data >= other_threshold].head(top_n)
            all_labels.update(main_slices.index.tolist())
        
        # Sort labels by overall mean
        overall_means = data[pct_cols].mean()
        overall_means.index = [col.replace('_pct', '') for col in overall_means.index]
        sorted_labels = [l for l in overall_means.sort_values(ascending=False).index if l in all_labels]
        if 'Other' not in sorted_labels:
            sorted_labels.append('Other')
        
        # Create legend patches
        legend_patches = [mpatches.Patch(color=color_map.get(label, '#999999'), label=label) 
                         for label in sorted_labels if label in all_labels or label == 'Other']
        
        fig.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.0, 0.5),
                  fontsize=9, title='Bile Acids')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return fig
    
    def plot_composition_horizontal_bars(
        self,
        data: pd.DataFrame,
        group_col: str,
        pct_cols: List[str],
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        top_n: int = 15,
        show_values: bool = True
    ) -> plt.Figure:
        """
        Create horizontal bar charts showing bile acid percentages by group.
        
        Args:
            data: DataFrame with group column and percentage columns
            group_col: Column name for grouping
            pct_cols: List of percentage column names (with _pct suffix)
            title: Overall figure title
            figsize: Figure size (auto-calculated if None)
            top_n: Maximum number of bile acids to show
            show_values: Whether to show percentage values on bars
            
        Returns:
            Matplotlib figure with horizontal bar charts
        """
        # Filter NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.lower() != 'nan']
        
        groups = data[group_col].unique().tolist()
        n_groups = len(groups)
        
        # Calculate mean percentages for each group
        group_means = data.groupby(group_col)[pct_cols].mean()
        
        # Clean column names (remove _pct suffix)
        group_means.columns = [col.replace('_pct', '') for col in group_means.columns]
        
        # Get top N bile acids by overall mean
        overall_means = group_means.mean().sort_values(ascending=False)
        top_bas = overall_means.head(top_n).index.tolist()
        
        # Calculate figure size
        if figsize is None:
            figsize = (12, max(6, len(top_bas) * 0.4))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up positions
        y_pos = np.arange(len(top_bas))
        bar_height = 0.8 / n_groups
        
        # Get colors for groups
        group_colors = dict(zip(groups, self.group_palette[:n_groups]))
        
        # Plot bars for each group
        for i, group in enumerate(groups):
            values = [group_means.loc[group, ba] for ba in top_bas]
            offset = (i - n_groups/2 + 0.5) * bar_height
            
            bars = ax.barh(y_pos + offset, values, bar_height * 0.9,
                          label=group, color=group_colors[group],
                          edgecolor='white', linewidth=0.5)
            
            # Add value labels
            if show_values:
                for bar, val in zip(bars, values):
                    if val >= 1:  # Only show if >= 1%
                        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                               f'{val:.1f}%', va='center', ha='left', fontsize=8)
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_bas)
        ax.invert_yaxis()  # Top bile acid at top
        ax.set_xlabel('% of Total Bile Acid Pool')
        ax.set_xlim(0, ax.get_xlim()[1] * 1.15)  # Add space for labels
        
        ax.legend(title='Group', loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_composition_stacked_horizontal(
        self,
        data: pd.DataFrame,
        group_col: str,
        pct_cols: List[str],
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        top_n: int = 12
    ) -> plt.Figure:
        """
        Create stacked horizontal bar chart showing bile acid pool composition.
        Each bar represents a group, segments show bile acid proportions.
        
        Args:
            data: DataFrame with group column and percentage columns
            group_col: Column name for grouping
            pct_cols: List of percentage column names (with _pct suffix)
            title: Overall figure title
            figsize: Figure size
            top_n: Maximum number of bile acids to show (rest grouped as "Other")
            
        Returns:
            Matplotlib figure
        """
        # Filter NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.lower() != 'nan']
        
        groups = data[group_col].unique().tolist()
        n_groups = len(groups)
        
        # Calculate mean percentages for each group
        group_means = data.groupby(group_col)[pct_cols].mean()
        group_means.columns = [col.replace('_pct', '') for col in group_means.columns]
        
        # Get top N bile acids by overall mean
        overall_means = group_means.mean().sort_values(ascending=False)
        top_bas = overall_means.head(top_n).index.tolist()
        
        # Create "Other" category
        other_bas = [ba for ba in group_means.columns if ba not in top_bas]
        if other_bas:
            group_means['Other'] = group_means[other_bas].sum(axis=1)
            plot_bas = top_bas + ['Other']
        else:
            plot_bas = top_bas
        
        # Calculate figure size
        if figsize is None:
            figsize = (12, max(4, n_groups * 1.2))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(plot_bas)))
        
        # Plot stacked horizontal bars
        y_pos = np.arange(n_groups)
        left = np.zeros(n_groups)
        
        for i, ba in enumerate(plot_bas):
            values = group_means.loc[groups, ba].values
            bars = ax.barh(y_pos, values, left=left, label=ba, color=colors[i],
                          edgecolor='white', linewidth=0.5, height=0.6)
            
            # Add labels for segments >= 5%
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val >= 5:
                    ax.text(left[j] + val/2, bar.get_y() + bar.get_height()/2,
                           f'{val:.0f}%', va='center', ha='center', 
                           fontsize=8, fontweight='bold', color='white')
            
            left += values
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups)
        ax.set_xlabel('% of Total Bile Acid Pool')
        ax.set_xlim(0, 100)
        
        # Legend outside
        ax.legend(title='Bile Acids', bbox_to_anchor=(1.02, 1), loc='upper left',
                 fontsize=8)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    # ===================================================================
    # TWO-WAY ANOVA VISUALIZATION METHODS
    # ===================================================================

    def plot_twoway_bar(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        twoway_result=None,
        ax: Optional[plt.Axes] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        ylabel: str = "Concentration",
        show_points: bool = True,
        title: Optional[str] = None,
        show_anova_text: bool = True,
        plot_type: str = "bar",
    ) -> plt.Axes:
        """
        Create grouped plot for two-way ANOVA.

        Factor A on x-axis, Factor B as hue/color with legend.
        Supports bar, box, strip, and violin plot types.
        """
        import math

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_single)

        fa_name = factor_a_name or factor_a_col
        fb_name = factor_b_name or factor_b_col

        # Clean data
        df = data[[value_col, factor_a_col, factor_b_col]].dropna().copy()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()

        # Get ordered levels
        a_levels = sorted(df[factor_a_col].unique())
        b_levels = sorted(df[factor_b_col].unique())
        n_a = len(a_levels)
        n_b = len(b_levels)

        # Bar positioning
        bar_width = 0.8 / n_b
        x_base = np.arange(n_a)
        palette = self.group_palette[:n_b]
        palette_dict = dict(zip(b_levels, palette))

        if plot_type == "box":
            sns.boxplot(data=df, x=factor_a_col, y=value_col, hue=factor_b_col,
                       ax=ax, palette=palette_dict, order=a_levels, hue_order=b_levels)
            if show_points:
                sns.stripplot(data=df, x=factor_a_col, y=value_col, hue=factor_b_col,
                             ax=ax, dodge=True, color='black', alpha=0.6, size=4,
                             order=a_levels, hue_order=b_levels, legend=False,
                             jitter=0.15, zorder=10, edgecolor='white', linewidth=0.3)
        elif plot_type == "violin":
            sns.violinplot(data=df, x=factor_a_col, y=value_col, hue=factor_b_col,
                          ax=ax, palette=palette_dict, order=a_levels, hue_order=b_levels,
                          inner=None)
            if show_points:
                sns.stripplot(data=df, x=factor_a_col, y=value_col, hue=factor_b_col,
                             ax=ax, dodge=True, color='black', alpha=0.6, size=4,
                             order=a_levels, hue_order=b_levels, legend=False,
                             jitter=0.15, zorder=10, edgecolor='white', linewidth=0.3)
        elif plot_type == "strip":
            sns.stripplot(data=df, x=factor_a_col, y=value_col, hue=factor_b_col,
                         ax=ax, dodge=True, palette=palette_dict, order=a_levels, hue_order=b_levels,
                         size=4, alpha=0.6)
            # Add mean +/- SEM lines
            for j, b_level in enumerate(b_levels):
                x_pos = x_base + (j - (n_b - 1) / 2) * bar_width
                for i_a, a_level in enumerate(a_levels):
                    cell = df[(df[factor_a_col] == a_level) & (df[factor_b_col] == b_level)][value_col]
                    if len(cell) > 0:
                        m, s = cell.mean(), cell.sem() if len(cell) > 1 else 0
                        ax.hlines(m, x_pos[i_a] - bar_width * 0.35, x_pos[i_a] + bar_width * 0.35,
                                 color='black', linewidth=1.5, zorder=5)
                        ax.errorbar(x_pos[i_a], m, yerr=s, color='black',
                                   capsize=4, capthick=1.5, linewidth=1.5, fmt='none', zorder=5)
        else:  # "bar"
            for j, b_level in enumerate(b_levels):
                x_pos = x_base + (j - (n_b - 1) / 2) * bar_width
                means = []
                sems = []
                for a_level in a_levels:
                    cell = df[(df[factor_a_col] == a_level) & (df[factor_b_col] == b_level)][value_col]
                    means.append(cell.mean() if len(cell) > 0 else 0)
                    sems.append(cell.sem() if len(cell) > 1 else 0)

                ax.bar(x_pos, means, width=bar_width * 0.9, yerr=sems,
                       capsize=3, color=palette[j], edgecolor='black', linewidth=0.5,
                       label=str(b_level), zorder=2)

                if show_points:
                    for i, a_level in enumerate(a_levels):
                        cell_vals = df[(df[factor_a_col] == a_level) & (df[factor_b_col] == b_level)][value_col]
                        if len(cell_vals) > 0:
                            jitter = np.random.normal(0, bar_width * 0.08, len(cell_vals))
                            ax.scatter(x_pos[i] + jitter, cell_vals, color='black', alpha=0.6,
                                       s=20, zorder=10, edgecolors='white', linewidths=0.3)

            ax.set_xticks(x_base)
            ax.set_xticklabels(a_levels)

        ax.set_xlabel(fa_name)
        ax.set_ylabel(ylabel)

        # Add post-hoc significance brackets FIRST (they expand y-axis)
        if twoway_result is not None and twoway_result.posthoc_results is not None:
            self._add_twoway_significance_bars(
                ax, df, value_col, factor_a_col, factor_b_col,
                a_levels, b_levels, bar_width, twoway_result
            )

        # Add ANOVA p-value text annotation AFTER brackets
        if twoway_result is not None and show_anova_text:
            tw = twoway_result
            def _stars(p):
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    return 'N/A'
                if p < 0.001: return '***'
                if p < 0.01: return '**'
                if p < 0.05: return '*'
                return 'n.s.'
            def _fmt_p(p):
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    return 'N/A'
                return f"{p:.3f}"

            anova_text = (
                f"{tw.factor_a_name}: p={_fmt_p(tw.factor_a_pvalue)} {_stars(tw.factor_a_pvalue)}\n"
                f"{tw.factor_b_name}: p={_fmt_p(tw.factor_b_pvalue)} {_stars(tw.factor_b_pvalue)}\n"
                f"{tw.factor_a_name}\u00d7{tw.factor_b_name}: p={_fmt_p(tw.interaction_pvalue)} {_stars(tw.interaction_pvalue)}"
            )
            y_lo, y_hi = ax.get_ylim()
            ax.set_ylim(top=y_hi + (y_hi - y_lo) * 0.15)
            ax.text(0.02, 0.98, anova_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

        # Legend
        ax.legend(title=fb_name, loc='upper right', framealpha=0.9)

        if title:
            ax.set_title(title, fontsize=10)
        elif twoway_result is not None:
            p_int = twoway_result.interaction_pvalue
            sig_marker = '*' if twoway_result.interaction_significant else ''
            p_str = 'N/A' if (p_int is None or (isinstance(p_int, float) and math.isnan(p_int))) else f"{p_int:.3f}"
            ax.set_title(f"{value_col} (interaction p={p_str}){sig_marker}", fontsize=10)
        else:
            ax.set_title(value_col, fontsize=10)

        return ax

    def _add_twoway_significance_bars(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        a_levels: list,
        b_levels: list,
        bar_width: float,
        twoway_result,
        max_bars: int = 4
    ):
        """Add significance bracket annotations for two-way post-hoc comparisons."""
        import math

        posthoc = twoway_result.posthoc_results
        if posthoc is None or len(posthoc) == 0:
            return

        sig_rows = posthoc[posthoc['significant']].copy()
        p_col = 'pvalue_adj' if 'pvalue_adj' in sig_rows.columns else 'pvalue'
        sig_rows = sig_rows.sort_values(p_col).head(max_bars)

        if len(sig_rows) == 0:
            return

        # Calculate y positions for brackets
        y_max = ax.get_ylim()[1]
        y_step = (y_max - ax.get_ylim()[0]) * 0.06
        current_y = y_max

        n_b = len(b_levels)
        x_base = np.arange(len(a_levels))

        for _, row in sig_rows.iterrows():
            p = row.get('pvalue_adj', row.get('pvalue', 1.0))

            g1, g2 = str(row['group1']), str(row['group2'])
            within_level = str(row.get('within_level', 'all'))
            factor_compared = row.get('factor_compared', '')

            if twoway_result.posthoc_type == "main_effect_marginal":
                if factor_compared == twoway_result.factor_a_name:
                    anova_p = twoway_result.factor_a_pvalue
                    if anova_p is not None and not (isinstance(anova_p, float) and math.isnan(anova_p)):
                        p = anova_p
                elif factor_compared == twoway_result.factor_b_name:
                    anova_p = twoway_result.factor_b_pvalue
                    if anova_p is not None and not (isinstance(anova_p, float) and math.isnan(anova_p)):
                        p = anova_p

            if p < 0.001:
                annot = '***'
            elif p < 0.01:
                annot = '**'
            elif p < 0.05:
                annot = '*'
            else:
                continue

            # Determine x positions for the bracket
            try:
                if twoway_result.posthoc_type == "simple_main_effects":
                    if factor_compared == twoway_result.factor_a_name:
                        b_idx = b_levels.index(within_level) if within_level in b_levels else 0
                        a_idx1 = a_levels.index(g1) if g1 in a_levels else None
                        a_idx2 = a_levels.index(g2) if g2 in a_levels else None
                        if a_idx1 is None or a_idx2 is None:
                            continue
                        x1 = x_base[a_idx1] + (b_idx - (n_b - 1) / 2) * bar_width
                        x2 = x_base[a_idx2] + (b_idx - (n_b - 1) / 2) * bar_width
                    else:
                        a_idx = a_levels.index(within_level) if within_level in a_levels else 0
                        b_idx1 = b_levels.index(g1) if g1 in b_levels else None
                        b_idx2 = b_levels.index(g2) if g2 in b_levels else None
                        if b_idx1 is None or b_idx2 is None:
                            continue
                        x1 = x_base[a_idx] + (b_idx1 - (n_b - 1) / 2) * bar_width
                        x2 = x_base[a_idx] + (b_idx2 - (n_b - 1) / 2) * bar_width
                else:
                    if factor_compared == twoway_result.factor_a_name:
                        a_idx1 = a_levels.index(g1) if g1 in a_levels else None
                        a_idx2 = a_levels.index(g2) if g2 in a_levels else None
                        if a_idx1 is None or a_idx2 is None:
                            continue
                        x1 = x_base[a_idx1]
                        x2 = x_base[a_idx2]
                    else:
                        continue
            except (ValueError, IndexError):
                continue

            # Draw bracket
            bracket_y = current_y
            ax.plot([x1, x1, x2, x2],
                    [bracket_y, bracket_y + y_step * 0.3, bracket_y + y_step * 0.3, bracket_y],
                    color='black', linewidth=0.8)
            ax.text((x1 + x2) / 2, bracket_y + y_step * 0.35, annot,
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            current_y += y_step

        # Expand y-axis to fit brackets
        ax.set_ylim(top=current_y + y_step)

    def plot_twoway_interaction(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        twoway_result=None,
        ax: Optional[plt.Axes] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        ylabel: str = "Concentration",
    ) -> plt.Axes:
        """
        Create interaction plot (line plot) for two-way ANOVA.

        Factor A on x-axis, separate lines for each Factor B level.
        Shows means +/- SEM. Parallel lines = no interaction,
        crossing/diverging lines = interaction present.
        """
        import math

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_single)

        fa_name = factor_a_name or factor_a_col
        fb_name = factor_b_name or factor_b_col

        df = data[[value_col, factor_a_col, factor_b_col]].dropna().copy()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()

        a_levels = sorted(df[factor_a_col].unique())
        b_levels = sorted(df[factor_b_col].unique())
        palette = self.group_palette[:len(b_levels)]
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

        x_pos = np.arange(len(a_levels))

        for j, b_level in enumerate(b_levels):
            means = []
            sems = []
            for a_level in a_levels:
                cell = df[(df[factor_a_col] == a_level) & (df[factor_b_col] == b_level)][value_col]
                means.append(cell.mean() if len(cell) > 0 else np.nan)
                sems.append(cell.sem() if len(cell) > 1 else 0)

            marker = markers[j % len(markers)]
            ax.errorbar(x_pos, means, yerr=sems,
                       marker=marker, markersize=8,
                       color=palette[j], linewidth=2,
                       capsize=4, capthick=1.5,
                       label=str(b_level), zorder=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(a_levels)
        ax.set_xlabel(fa_name)
        ax.set_ylabel(ylabel)
        ax.legend(title=fb_name, loc='best')

        title = f"{value_col} — Interaction Plot"
        if twoway_result is not None:
            p_int = twoway_result.interaction_pvalue
            sig = "SIGNIFICANT" if twoway_result.interaction_significant else "n.s."
            p_str = 'N/A' if (p_int is None or (isinstance(p_int, float) and math.isnan(p_int))) else f"{p_int:.3f}"
            title += f"\n(interaction p={p_str}, {sig})"
        ax.set_title(title, fontsize=10)

        return ax

    def plot_twoway_multi_panel(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        factor_a_col: str,
        factor_b_col: str,
        twoway_results: Dict = None,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        ylabel: str = "Concentration",
        show_points: bool = True,
        plot_type: str = "bar",
    ) -> plt.Figure:
        """
        Create multi-panel two-way grouped plots with significance annotations.

        Each panel shows one analyte with Factor A on x-axis, Factor B as hue.
        """
        n_plots = len(value_cols)
        nrows = int(np.ceil(n_plots / ncols))

        if figsize is None:
            figsize = (ncols * 4.5, nrows * 4.5)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col in enumerate(value_cols):
            if col not in data.columns:
                axes[i].set_visible(False)
                continue

            tw_result = None
            if twoway_results and col in twoway_results:
                result_obj = twoway_results[col]
                if hasattr(result_obj, 'twoway_result'):
                    tw_result = result_obj.twoway_result
                else:
                    tw_result = result_obj

            self.plot_twoway_bar(
                data=data, value_col=col,
                factor_a_col=factor_a_col, factor_b_col=factor_b_col,
                twoway_result=tw_result, ax=axes[i],
                factor_a_name=factor_a_name, factor_b_name=factor_b_name,
                ylabel=ylabel, show_points=show_points,
                plot_type=plot_type,
            )

        # Hide empty panels
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_twoway_interaction_multi_panel(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        factor_a_col: str,
        factor_b_col: str,
        twoway_results: Dict = None,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        ylabel: str = "Concentration",
    ) -> plt.Figure:
        """Multi-panel interaction (line) plots for two-way ANOVA."""
        n_plots = len(value_cols)
        nrows = int(np.ceil(n_plots / ncols))

        if figsize is None:
            figsize = (ncols * 4.5, nrows * 4)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col in enumerate(value_cols):
            if col not in data.columns:
                axes[i].set_visible(False)
                continue

            tw_result = None
            if twoway_results and col in twoway_results:
                result_obj = twoway_results[col]
                if hasattr(result_obj, 'twoway_result'):
                    tw_result = result_obj.twoway_result
                else:
                    tw_result = result_obj

            self.plot_twoway_interaction(
                data=data, value_col=col,
                factor_a_col=factor_a_col, factor_b_col=factor_b_col,
                twoway_result=tw_result, ax=axes[i],
                factor_a_name=factor_a_name, factor_b_name=factor_b_name,
                ylabel=ylabel,
            )

        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        filepath: Union[str, Path],
        formats: List[str] = ['png', 'pdf'],
        dpi: int = 300
    ):
        """Save figure in multiple formats."""
        filepath = Path(filepath)
        
        for fmt in formats:
            save_path = filepath.with_suffix(f'.{fmt}')
            fig.savefig(save_path, format=fmt, dpi=dpi, bbox_inches='tight')


def create_summary_figure(
    data: pd.DataFrame,
    bile_acid_cols: List[str],
    group_col: str,
    totals: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Create a comprehensive summary figure with multiple panels."""
    fig = plt.figure(figsize=(16, 12))
    
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Total bile acids by group
    ax1 = fig.add_subplot(gs[0, 0])
    combined = pd.concat([data[[group_col]], totals], axis=1)
    if 'total_all' in combined.columns:
        sns.boxplot(data=combined, x=group_col, y='total_all', ax=ax1)
        ax1.set_title('Total Bile Acids')
        ax1.set_ylabel('Concentration (nmol/L)')
    
    # 2. Primary vs Secondary
    ax2 = fig.add_subplot(gs[0, 1])
    if 'total_primary' in totals.columns and 'total_secondary' in totals.columns:
        plot_data = pd.DataFrame({
            'Primary': totals['total_primary'],
            'Secondary': totals['total_secondary'],
            'Group': data[group_col]
        }).melt(id_vars='Group', var_name='Type', value_name='Concentration')
        sns.barplot(data=plot_data, x='Group', y='Concentration', hue='Type', ax=ax2)
        ax2.set_title('Primary vs Secondary')
        ax2.legend(title='')
    
    # 3. Conjugated vs Unconjugated
    ax3 = fig.add_subplot(gs[0, 2])
    if 'total_conjugated' in totals.columns and 'total_unconjugated' in totals.columns:
        plot_data = pd.DataFrame({
            'Conjugated': totals['total_conjugated'],
            'Unconjugated': totals['total_unconjugated'],
            'Group': data[group_col]
        }).melt(id_vars='Group', var_name='Type', value_name='Concentration')
        sns.barplot(data=plot_data, x='Group', y='Concentration', hue='Type', ax=ax3)
        ax3.set_title('Conjugated vs Unconjugated')
        ax3.legend(title='')
    
    # 4. Glycine vs Taurine
    ax4 = fig.add_subplot(gs[1, 0])
    if 'glycine_conjugated' in totals.columns and 'taurine_conjugated' in totals.columns:
        plot_data = pd.DataFrame({
            'Glycine': totals['glycine_conjugated'],
            'Taurine': totals['taurine_conjugated'],
            'Group': data[group_col]
        }).melt(id_vars='Group', var_name='Conjugation', value_name='Concentration')
        sns.barplot(data=plot_data, x='Group', y='Concentration', hue='Conjugation', ax=ax4)
        ax4.set_title('Glycine vs Taurine Conjugation')
        ax4.legend(title='')
    
    # 5. Composition stacked bar
    ax5 = fig.add_subplot(gs[1, 1:])
    available_cols = [c for c in bile_acid_cols if c in data.columns][:15]
    if available_cols:
        group_means = data.groupby(group_col)[available_cols].mean()
        row_sums = group_means.sum(axis=1)
        pct_data = group_means.div(row_sums, axis=0) * 100
        pct_data.plot(kind='bar', stacked=True, ax=ax5, 
                     colormap='tab20', edgecolor='white', linewidth=0.3)
        ax5.set_title('Bile Acid Composition (%)')
        ax5.set_ylabel('Percentage')
        ax5.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # 6. Top individual BAs
    ax6 = fig.add_subplot(gs[2, :])
    top_bas = data[available_cols].mean().nlargest(8).index.tolist()
    if top_bas:
        plot_data = data[[group_col] + top_bas].melt(
            id_vars=group_col, var_name='Bile Acid', value_name='Concentration'
        )
        sns.boxplot(data=plot_data, x='Bile Acid', y='Concentration', 
                   hue=group_col, ax=ax6)
        ax6.set_title('Top 8 Most Abundant Bile Acids')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
        ax6.legend(title='Group')
    
    plt.suptitle('Bile Acid Analysis Summary', fontsize=14, fontweight='bold', y=1.02)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
