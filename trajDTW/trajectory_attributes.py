#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Attribute Functions

This module provides functionality for extracting and visualizing trajectory attributes
such as gene expression correlation with pseudotime, peak expression timing, and 
expression levels from 3D trajectory data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy.stats import pearsonr
from pathlib import Path

class TrajectoryAttributes:
    """
    Container for trajectory attribute data including correlation, expression, and peak timing.
    Provides a simplified alternative to MuData for storing and manipulating trajectory attribute data.
    
    Attributes
    ----------
    correlation : pd.DataFrame
        DataFrame of correlation values (genes × batches)
    expression : pd.DataFrame
        DataFrame of normalized expression values (genes × batches)
    peak : pd.DataFrame
        DataFrame of peak timing values (genes × batches)
    gene_names : list
        List of gene names
    batch_names : list
        List of batch names
    """
    
    def __init__(
        self,
        correlation: pd.DataFrame,
        expression: pd.DataFrame,
        peak: pd.DataFrame
    ):
        """
        Initialize TrajectoryAttributes with correlation, expression, and peak DataFrames.
        
        Parameters
        ----------
        correlation : pd.DataFrame
            DataFrame of correlation values (genes × batches)
        expression : pd.DataFrame
            DataFrame of normalized expression values (genes × batches)
        peak : pd.DataFrame
            DataFrame of peak timing values (genes × batches)
        """
        self.correlation = correlation
        self.expression = expression
        self.peak = peak
        self.gene_names = correlation.index.tolist()
        self.batch_names = correlation.columns.tolist()
        
    def __getitem__(self, key):
        """
        Allow dictionary-like access to data attributes.
        
        Parameters
        ----------
        key : str
            One of 'correlation', 'expression', 'peak'
            
        Returns
        -------
        pd.DataFrame
            The requested DataFrame
        """
        if key in ['correlation', 'corr']:
            return self.correlation
        elif key in ['expression', 'expr']:
            return self.expression
        elif key in ['peak']:
            return self.peak
        else:
            raise KeyError(f"Invalid key: {key}. Valid keys are 'correlation', 'expression', and 'peak'.")
    
    def subset(self, genes=None, batches=None):
        """
        Create a new TrajectoryAttributes object with a subset of genes and/or batches.
        
        Parameters
        ----------
        genes : list of str, optional
            Subset of genes to include. If None, all genes are used.
        batches : list of str, optional
            Subset of batches to include. If None, all batches are used.
            
        Returns
        -------
        TrajectoryAttributes
            A new TrajectoryAttributes object with the specified subset of data
        """
        corr_df = self.correlation
        expr_df = self.expression
        peak_df = self.peak
        
        if genes is not None:
            corr_df = corr_df.loc[genes]
            expr_df = expr_df.loc[genes]
            peak_df = peak_df.loc[genes]
            
        if batches is not None:
            corr_df = corr_df[batches]
            expr_df = expr_df[batches]
            peak_df = peak_df[batches]
            
        return TrajectoryAttributes(corr_df, expr_df, peak_df)
    
    def to_dot_table(self, genes=None, samples=None):
        """
        Create a DataFrame suitable for dotplot visualization.
        
        Parameters
        ----------
        genes : list of str, optional
            Subset of genes to include. If None, all genes are used.
        samples : list of str, optional
            Subset of samples/batches to include. If None, all samples are used.
            
        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns for gene, sample, correlation, expression, and peak
        """
        # Use the existing make_dot_table function
        return make_dot_table(
            self.correlation, 
            self.expression, 
            self.peak,
            genes=genes,
            samples=samples
        )

def calculate_trajectory_attributes(
    reshaped_data: np.ndarray,
    gene_names: List[str],
    batch_names: Optional[List[str]] = None,
    num_bins: int = 10
) -> Union[Dict[str, pd.DataFrame], TrajectoryAttributes]:
    """
    Calculate correlation, expression, and peak attributes from 3D trajectory data.
    
    Parameters
    ----------
    reshaped_data : np.ndarray
        3D array with shape (batch, pseudotime, gene)
    gene_names : list of str
        Names of genes corresponding to the third dimension of reshaped_data
    batch_names : list of str, optional
        Names of batches corresponding to the first dimension of reshaped_data
        If None, batches will be numbered as "Batch_1", "Batch_2", etc.
    num_bins : int, default=10
        Number of bins to divide pseudotime into for peak determination
        
    Returns
    -------
    TrajectoryAttributes
        Class containing correlation, expression, and peak DataFrames
    """
    n_batches, n_timepoints, n_genes = reshaped_data.shape
    
    # Default batch names if not provided
    if batch_names is None:
        batch_names = [f"Batch_{i+1}" for i in range(n_batches)]
    
    # Create time points array
    time_points = np.linspace(0, 1, n_timepoints)
    
    # Initialize dictionaries to store results
    corr_data = {}
    expr_data = {}
    peak_data = {}
    
    # Process each batch
    for batch_idx, batch_name in enumerate(batch_names):
        # Get data for this batch
        batch_data = reshaped_data[batch_idx, :, :]
        
        # Create bins for peak determination
        bin_edges = np.linspace(0, 1, num_bins + 1)
        
        # Calculate correlation, expression, and peak for each gene
        corr_values = []
        expr_values = []
        peak_values = []
        
        for gene_idx in range(n_genes):
            gene_expression = batch_data[:, gene_idx]
            
            # Calculate correlation with pseudotime
            if np.sum(gene_expression) > 0 and np.std(gene_expression) > 0:
                corr, _ = pearsonr(gene_expression, time_points)
            else:
                corr = 0
                
            # Calculate normalized expression (max normalized)
            max_expr = np.max(gene_expression)
            if max_expr > 0:
                norm_expr = np.mean(gene_expression) / max_expr
            else:
                norm_expr = 0
                
            # Calculate peak timing
            # 1. Bin the data
            binned_data = np.zeros(num_bins)
            for i in range(num_bins):
                mask = (time_points >= bin_edges[i]) & (time_points < bin_edges[i+1])
                if np.any(mask):
                    binned_data[i] = np.mean(gene_expression[mask])
            
            # 2. Find the peak bin
            if np.sum(binned_data) > 0:
                peak_bin = np.argmax(binned_data)
                # Convert to a 1-10 scale
                peak_value = peak_bin + 1
            else:
                peak_value = 0
                
            corr_values.append(corr)
            expr_values.append(norm_expr)
            peak_values.append(peak_value)
            
        # Add batch results to dictionaries
        corr_data[batch_name] = corr_values
        expr_data[batch_name] = expr_values
        peak_data[batch_name] = peak_values
    
    # Convert to DataFrames
    corr_df = pd.DataFrame(corr_data, index=gene_names)
    expr_df = pd.DataFrame(expr_data, index=gene_names)
    peak_df = pd.DataFrame(peak_data, index=gene_names)
    
    # Convert peak values to categorical
    peak_df = peak_df.applymap(lambda x: 'Early' if x <= num_bins/3 else 
                                         'Middle' if x <= 2*num_bins/3 else 
                                         'Late' if x > 0 else 'None')
    
    # Return a TrajectoryAttributes object
    return TrajectoryAttributes(corr_df, expr_df, peak_df)

def make_dot_table(
    corr_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    genes: Optional[List[str]] = None,
    samples: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a DataTable suitable for dotplot visualization.
    
    Parameters
    ----------
    corr_df : pd.DataFrame
        DataFrame of correlation values (genes × batches)
    expr_df : pd.DataFrame
        DataFrame of expression values (genes × batches)
    peak_df : pd.DataFrame
        DataFrame of peak timing values (genes × batches)
    genes : list of str, optional
        Subset of genes to include. If None, all genes are used.
    samples : list of str, optional
        Subset of samples/batches to include. If None, all samples are used.
        
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns for gene, sample, correlation, expression, and peak
    """
    # Filter genes and samples if specified
    if genes is not None:
        corr_df = corr_df.loc[genes]
        expr_df = expr_df.loc[genes]
        peak_df = peak_df.loc[genes]
        
    if samples is not None:
        corr_df = corr_df[samples]
        expr_df = expr_df[samples]
        peak_df = peak_df[samples]
    
    # Convert to long format
    corr_long = corr_df.stack().reset_index()
    corr_long.columns = ['Gene', 'Sample', 'Correlation']
    
    expr_long = expr_df.stack().reset_index()
    expr_long.columns = ['Gene', 'Sample', 'Expression']
    
    peak_long = peak_df.stack().reset_index()
    peak_long.columns = ['Gene', 'Sample', 'Peak']
    
    # Merge the DataFrames
    dot_table = pd.merge(corr_long, expr_long, on=['Gene', 'Sample'])
    dot_table = pd.merge(dot_table, peak_long, on=['Gene', 'Sample'])
    
    return dot_table

def plot_trajectory_dotplot(
    dot_table: pd.DataFrame,
    col_split: Optional[Union[int, List, pd.Series, pd.DataFrame]] = None,
    row_split: Optional[Union[int, List, pd.Series, pd.DataFrame]] = None,
    figsize: Tuple[int, int] = (10, 10),
    dot_scale: float = 100,
    col_spacing: float = 1.0,
    dot_width: float = 0.45,
    subplot_spacing: float = 0.2,
    colormap: str = "coolwarm",
    show_rownames: bool = True,
    show_colnames: bool = True,
    grid: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Create a dotplot visualization of trajectory attributes with support for splitting by rows and columns.
    
    Parameters
    ----------
    dot_table : pd.DataFrame
        DataFrame with columns Gene, Sample, Correlation, Expression, and Peak
    col_split : int, list, Series, or DataFrame, optional
        How to group the columns (samples):
        - If int: Split into that many evenly sized groups
        - If list: List of category labels for each sample in the order they appear
        - If Series/DataFrame: Grouping information where index corresponds to sample names
          and values are group identifiers
    row_split : int, list, Series, or DataFrame, optional
        How to group the rows (genes):
        - If int: Split into that many evenly sized groups
        - If list: List of category labels for each gene in the order they appear
        - If Series/DataFrame: Grouping information where index corresponds to gene names
          and values are group identifiers
    figsize : tuple, default=(10, 10)
        Base figure size in inches (width, height) for a single subplot. The actual figure size 
        will be scaled based on the number of row and column groups and col_spacing.
    dot_scale : float, default=100
        Scale factor for dot sizes
    col_spacing : float, default=1.0
        Spacing between dots (columns) in the dotplot. Higher values increase spacing.
    dot_width : float, default=0.45
        Width of each dot in inches. Controls the physical width of each subplot to ensure 
        consistent dot widths across all subplots regardless of the number of samples.
    subplot_spacing : float, default=0.2
        Spacing between subplots as a fraction of the subplot size.
    colormap : str, default="coolwarm"
        Matplotlib colormap name for correlation values
    show_rownames : bool, default=True
        Whether to show gene names on the y-axis. When col_split is used, row names will only 
        be shown in the leftmost box of each row.
    show_colnames : bool, default=True
        Whether to show sample names on the x-axis
    grid : bool, default=True
        Whether to show grid lines
    save_path : str or Path, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Get unique genes and samples
    genes = dot_table['Gene'].unique()
    samples = dot_table['Sample'].unique()
    
    # Handle column grouping
    if col_split is not None:
        if isinstance(col_split, int):
            # Create n evenly sized groups of samples
            n_groups = col_split
            group_size = len(samples) // n_groups
            remainder = len(samples) % n_groups
            
            col_groups = {}
            start_idx = 0
            for i in range(n_groups):
                # Distribute remainder across groups
                group_len = group_size + (1 if i < remainder else 0)
                end_idx = start_idx + group_len
                col_groups[f'Group {i+1}'] = samples[start_idx:end_idx]
                start_idx = end_idx
        elif isinstance(col_split, list):
            # Handle list input where each item corresponds to a sample in order
            if len(col_split) != len(samples):
                raise ValueError(f"Length of col_split list ({len(col_split)}) must match number of samples ({len(samples)})")
            
            # Create mapping from sample to category
            category_map = {sample: category for sample, category in zip(samples, col_split)}
            
            # Group samples by category
            col_groups = {}
            for category in sorted(set(col_split)):
                category_samples = [s for s, c in category_map.items() if c == category]
                if category_samples:
                    col_groups[category] = np.array(category_samples)
        elif isinstance(col_split, pd.Series):
            # Use Series for grouping where index matches sample names
            # Must align with sample names in the data
            aligned_col_split = {s: col_split.get(s, "Other") for s in samples}
            col_groups = {}
            for group in sorted(set(aligned_col_split.values())):
                group_samples = [s for s in samples if aligned_col_split[s] == group]
                if group_samples:  # Only include non-empty groups
                    col_groups[group] = np.array(group_samples)
        elif isinstance(col_split, pd.DataFrame):
            # Extract first column of DataFrame for grouping
            first_col = col_split.iloc[:, 0]
            aligned_col_split = {s: first_col.get(s, "Other") for s in samples}
            col_groups = {}
            for group in sorted(set(aligned_col_split.values())):
                group_samples = [s for s in samples if aligned_col_split[s] == group]
                if group_samples:  # Only include non-empty groups
                    col_groups[group] = np.array(group_samples)
        else:
            raise ValueError("col_split must be an int, list, pd.Series, or pd.DataFrame")
    else:
        # No column splitting
        col_groups = {'All Samples': samples}
    
    # Handle row grouping
    if row_split is not None:
        if isinstance(row_split, int):
            # Create n evenly sized groups of genes
            n_groups = row_split
            group_size = len(genes) // n_groups
            remainder = len(genes) % n_groups
            
            row_groups = {}
            start_idx = 0
            for i in range(n_groups):
                # Distribute remainder across groups
                group_len = group_size + (1 if i < remainder else 0)
                end_idx = start_idx + group_len
                row_groups[f'Group {i+1}'] = genes[start_idx:end_idx]
                start_idx = end_idx
        elif isinstance(row_split, list):
            # Handle list input where each item corresponds to a gene in order
            if len(row_split) != len(genes):
                raise ValueError(f"Length of row_split list ({len(row_split)}) must match number of genes ({len(genes)})")
            
            # Create mapping from gene to category
            category_map = {gene: category for gene, category in zip(genes, row_split)}
            
            # Group genes by category
            row_groups = {}
            for category in sorted(set(row_split)):
                category_genes = [g for g, c in category_map.items() if c == category]
                if category_genes:
                    row_groups[category] = np.array(category_genes)
        elif isinstance(row_split, pd.Series):
            # Use Series for grouping where index matches gene names
            aligned_row_split = {g: row_split.get(g, "Other") for g in genes}
            row_groups = {}
            for group in sorted(set(aligned_row_split.values())):
                group_genes = [g for g in genes if aligned_row_split[g] == group]
                if group_genes:  # Only include non-empty groups
                    row_groups[group] = np.array(group_genes)
        elif isinstance(row_split, pd.DataFrame):
            # Extract first column of DataFrame for grouping
            first_col = row_split.iloc[:, 0]
            aligned_row_split = {g: first_col.get(g, "Other") for g in genes}
            row_groups = {}
            for group in sorted(set(aligned_row_split.values())):
                group_genes = [g for g in genes if aligned_row_split[g] == group]
                if group_genes:  # Only include non-empty groups
                    row_groups[group] = np.array(group_genes)
        else:
            raise ValueError("row_split must be an int, list, pd.Series, or pd.DataFrame")
    else:
        # No row splitting
        row_groups = {'All Genes': genes}

    # Calculate subplot widths based on the actual number of samples in each group
    # This ensures each subplot's width is proportional to its sample count
    subplot_widths = {}
    for col_group_name, col_group_samples in col_groups.items():
        subplot_widths[col_group_name] = dot_width * len(col_group_samples) * col_spacing
    
    # Calculate total figure width based on sum of all subplot widths
    total_width = sum(subplot_widths.values()) * (1 + subplot_spacing)
    
    # Calculate heights based on original aspect ratio
    base_height = figsize[1]
    max_genes_per_group = max(len(group) for group in row_groups.values())
    subplot_height = base_height * (max_genes_per_group / len(genes))
    fig_height = subplot_height * len(row_groups) * (1 + subplot_spacing)
    
    # Create figure with calculated dimensions
    fig = plt.figure(figsize=(total_width, fig_height))
    
    # Create a GridSpec with width ratios proportional to the number of samples in each group
    width_ratios = [subplot_widths[col_group_name] for col_group_name in col_groups.keys()]
    gs = plt.GridSpec(
        len(row_groups), 
        len(col_groups), 
        figure=fig, 
        width_ratios=width_ratios
    )
    
    # Create the axes objects
    axes = np.empty((len(row_groups), len(col_groups)), dtype=object)
    
    for i, row_group_name in enumerate(row_groups.keys()):
        for j, col_group_name in enumerate(col_groups.keys()):
            # Create the subplot
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
    
    # Define marker shapes for peak timing
    markers = {
        'Early': 'o',    # Circle for early peak
        'Middle': 'D',   # Diamond for middle peak
        'Late': 's',     # Square for late peak
        'None': 'x'      # X for no peak
    }
    
    # Define colormap for correlation
    cmap = plt.cm.get_cmap(colormap)
    
    # Plot each subplot
    for i, (row_group_name, row_group_genes) in enumerate(row_groups.items()):
        for j, (col_group_name, col_group_samples) in enumerate(col_groups.items()):
            ax = axes[i, j]
            
            # Filter data for this subplot
            subplot_data = dot_table[
                dot_table['Gene'].isin(row_group_genes) & 
                dot_table['Sample'].isin(col_group_samples)
            ]
            
            # Get unique genes and samples in this subplot
            subplot_genes = np.array([g for g in row_group_genes if g in subplot_data['Gene'].unique()])
            subplot_samples = np.array([s for s in col_group_samples if s in subplot_data['Sample'].unique()])
            
            if len(subplot_genes) == 0 or len(subplot_samples) == 0:
                # No data for this subplot - hide the axes
                ax.axis('off')
                continue
            
            # Create mapping from gene/sample name to position in this subplot
            gene_to_pos = {g: i for i, g in enumerate(subplot_genes)}
            sample_to_pos = {s: j for j, s in enumerate(subplot_samples)}
            
            # Plot data points for this subplot
            for _, row in subplot_data.iterrows():
                gene = row['Gene']
                sample = row['Sample']
                
                # Calculate positions
                y_pos = gene_to_pos[gene]
                x_pos = sample_to_pos[sample] * col_spacing
                
                correlation = row['Correlation']
                expression = row['Expression']
                peak = row['Peak']
                
                # Calculate marker size based on expression
                size = expression * dot_scale
                
                # Calculate color based on correlation
                color = cmap((correlation + 1) / 2)  # Map from [-1, 1] to [0, 1]
                
                # Get marker based on peak
                marker = markers.get(peak, 'o')
                
                # Plot the point
                ax.scatter(x_pos, y_pos, s=size, c=[color], marker=marker, edgecolors='black', linewidths=0.5)
            
            # Set titles and labels
            if i == 0:
                ax.set_title(col_group_name)
            
            if j == 0:
                ax.set_ylabel(row_group_name)
            
            # Set ticks and tick labels - only show row names in the leftmost box
            if show_rownames and j == 0:  # Only show in the leftmost box
                ax.set_yticks(range(len(subplot_genes)))
                ax.set_yticklabels(subplot_genes)
            else:
                ax.set_yticks([])
            
            if show_colnames:
                ax.set_xticks([j * col_spacing for j in range(len(subplot_samples))])
                ax.set_xticklabels(subplot_samples, rotation=90)
            else:
                ax.set_xticks([])
            
            # Create custom grid that respects column spacing
            if grid:
                # Turn off the default grid
                ax.grid(False)
                
                # Add vertical grid lines at positions between columns
                for k in range(len(subplot_samples) + 1):
                    x = (k - 0.5) * col_spacing
                    ax.axvline(x, linestyle='--', color='gray', alpha=0.3)
                
                # Add horizontal grid lines at positions between rows
                for k in range(len(subplot_genes) + 1):
                    y = k - 0.5
                    ax.axhline(y, linestyle='--', color='gray', alpha=0.3)
            
            # Set limits to ensure all points are visible
            ax.set_xlim(-0.5 * col_spacing, (len(subplot_samples) - 0.5) * col_spacing)
            ax.set_ylim(-0.5, len(subplot_genes) - 0.5)
    
    # Add shared legend for markers
    legend_elements = [
        plt.Line2D([0], [0], marker=markers['Early'], color='w', markerfacecolor='gray', 
                  markersize=10, label='Early Peak'),
        plt.Line2D([0], [0], marker=markers['Middle'], color='w', markerfacecolor='gray', 
                  markersize=10, label='Middle Peak'),
        plt.Line2D([0], [0], marker=markers['Late'], color='w', markerfacecolor='gray', 
                  markersize=10, label='Late Peak')
    ]
    
    # Add a common colorbar for the whole figure
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), label='Correlation with Pseudotime', shrink=0.7)
    
    # Add legend to the figure
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
    
    # Add overall title
    fig.suptitle('Gene Trajectory Dotplot', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust rect to make room for the legend
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def attribute_to_dotplot(
    reshaped_data: np.ndarray,
    gene_names: List[str],
    genes_to_plot: Optional[List[str]] = None,
    batch_names: Optional[List[str]] = None,
    samples_to_plot: Optional[List[str]] = None,
    num_bins: int = 10,
    col_split: Optional[Union[int, List, pd.Series, pd.DataFrame]] = None,
    row_split: Optional[Union[int, List, pd.Series, pd.DataFrame]] = None,
    dot_width: float = 0.2,
    **dotplot_kwargs
) -> Figure:
    """
    Convenience function to directly create a dotplot from reshaped_data.
    
    Parameters
    ----------
    reshaped_data : np.ndarray
        3D array with shape (batch, pseudotime, gene)
    gene_names : list of str
        Names of genes corresponding to the third dimension of reshaped_data
    genes_to_plot : list of str, optional
        Subset of genes to include in the dotplot. If None, all genes are used.
    batch_names : list of str, optional
        Names of batches corresponding to the first dimension of reshaped_data.
        If None, batches will be numbered as "Batch_1", "Batch_2", etc.
    samples_to_plot : list of str, optional
        Subset of samples/batches to include in the dotplot. If None, all samples are used.
    num_bins : int, default=10
        Number of bins to divide pseudotime into for peak determination
    col_split : int, list, Series, or DataFrame, optional
        How to group the columns (samples):
        - If int: Split into that many evenly sized groups
        - If list: List of category labels for each sample in the order they appear
        - If Series/DataFrame: Grouping information where index corresponds to sample names
    row_split : int, list, Series, or DataFrame, optional
        How to group the rows (genes):
        - If int: Split into that many evenly sized groups
        - If list: List of category labels for each gene in the order they appear
        - If Series/DataFrame: Grouping information where index corresponds to gene names
    dot_width : float, default=0.2
        Width of each dot in inches. Controls the physical width of each subplot to ensure 
        consistent dot widths across all subplots regardless of the number of samples.
    **dotplot_kwargs
        Additional keyword arguments to pass to plot_trajectory_dotplot
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Calculate attributes
    attributes = calculate_trajectory_attributes(
        reshaped_data=reshaped_data,
        gene_names=gene_names,
        batch_names=batch_names,
        num_bins=num_bins
    )
    
    # Create dot table using the TrajectoryAttributes instance
    dot_table = attributes.to_dot_table(
        genes=genes_to_plot,
        samples=samples_to_plot
    )
    
    # Create dotplot with row and column splitting
    fig = plot_trajectory_dotplot(
        dot_table=dot_table, 
        col_split=col_split,
        row_split=row_split,
        dot_width=dot_width,
        **dotplot_kwargs
    )
    
    return fig 