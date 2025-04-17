#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization functions for the trajDTW package.

This module contains all visualization-related functions used throughout the package,
centralizing the plotting code to enhance readability and maintainability.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as mpatches

# Import needed for type hints only
from scipy import interpolate

# Define default color schemes
DEFAULT_COLORS = plt.cm.tab10.colors
BATCH_COLORS = plt.cm.tab20.colors

#######################
# Interpolation Plots #
#######################

def plot_interpolation_example(
    adata: "anndata.AnnData",
    gene_name: str,
    pseudo_col: str,
    batch_col: str,
    interpolator=None
) -> Figure:
    """
    Plot an example of Gaussian kernel interpolation for a gene.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the single-cell data
    gene_name : str
        Name of gene to plot
    pseudo_col : str
        Column in adata.obs containing pseudotime values
    batch_col : str
        Column in adata.obs containing batch labels
    interpolator : GaussianTrajectoryInterpolator, optional
        Interpolator instance. If None, a new one will be created.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with plot
    """
    # Import here to avoid circular imports
    from .cell_interpolation import GaussianTrajectoryInterpolator

    # If interpolator not provided, create a new one
    if interpolator is None:
        interpolator = GaussianTrajectoryInterpolator()

    # Get gene data
    if gene_name not in adata.var_names:
        raise ValueError(f"Gene {gene_name} not found in adata.var_names")

    gene_idx = list(adata.var_names).index(gene_name)
    expression_vector = adata.X[:, gene_idx].toarray().flatten()

    # Get pseudotime and batch information
    pseudotime = adata.obs[pseudo_col].values
    batch_labels = adata.obs[batch_col].values
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot raw data points
    batch_point_colors = dict(zip(unique_batches, BATCH_COLORS[:n_batches]))
    
    for batch in unique_batches:
        batch_mask = batch_labels == batch
        ax.scatter(
            pseudotime[batch_mask],
            expression_vector[batch_mask],
            s=30,
            alpha=0.4,
            c=[batch_point_colors[batch]],
            label=f"Raw Data - {batch}"
        )

    # Plot interpolated curves
    time_points = np.linspace(0, 1, 100)
    
    for batch_idx, batch in enumerate(unique_batches):
        batch_mask = batch_labels == batch
        batch_cells = np.where(batch_mask)[0]
        
        if len(batch_cells) > 0:
            # Compute the interpolated values across the time points
            batch_expression = interpolator.interpolate_gene_expression(
                expression_vector[batch_cells],
                interpolator.compute_weight_matrix(
                    interpolator.compute_abs_timediff_mat(pseudotime[batch_cells]),
                    interpolator.compute_adaptive_window_denominator(
                        interpolator.compute_cell_densities(pseudotime[batch_cells])
                    )
                ),
                time_points=time_points
            )
            
            # Plot means with std shading
            ax.plot(
                time_points,
                batch_expression,
                '-',
                lw=2,
                alpha=0.8,
                color=BATCH_COLORS[batch_idx],
                label=f"Interpolated - {batch}"
            )

    ax.set_xlabel("Pseudotime")
    ax.set_ylabel(f"Expression - {gene_name}")
    ax.set_title(f"Gaussian Kernel Interpolation for {gene_name}")
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    
    return fig

#############################
# Conservation Score Plots #
#############################

def plot_conservation_scores(
    conservation_scores: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
    prefix: str = "conservation",
    normalize: str = "zscore",
    variation_threshold: float = 0.1,
    variation_metric: str = "max",
    top_n: int = 20
) -> Figure:
    """
    Create a bar plot of gene conservation scores.

    Parameters
    ----------
    conservation_scores : pd.DataFrame
        DataFrame with conservation scores from calculate_trajectory_conservation
    save_dir : str or Path, optional
        Directory to save the figure. If None, the figure is not saved.
    prefix : str, default="conservation"
        Prefix for the saved file
    normalize : str, default="zscore"
        Normalization method used
    variation_threshold : float, default=0.1
        Variation threshold used for filtering
    variation_metric : str, default="max"
        Variation metric used for filtering
    top_n : int, default=20
        Number of top genes to display

    Returns
    -------
    matplotlib.figure.Figure
        Figure with plot
    """
    # Get top N genes
    top_genes_df = conservation_scores.head(top_n)
    
    # Create the figure
    fig = plt.figure(figsize=(12, 10))
    
    # Create a bar plot
    plt.barh(range(len(top_genes_df)), top_genes_df['normalized_score'], color='skyblue')
    plt.yticks(range(len(top_genes_df)), top_genes_df['gene'])
    plt.title(f'Gene Conservation Scores (Top {top_n} Genes, Normalization: {normalize})')
    plt.xlabel('Conservation Score (higher = more conserved)')
    
    if variation_threshold > 0:
        plt.ylabel(f'Gene (Variation Threshold: {variation_threshold}, Metric: {variation_metric})')
    else:
        plt.ylabel('Gene')
        
    plt.tight_layout()
    
    # Save if requested
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{prefix}_conservation_scores.png", dpi=300)
    
    return fig


######################
# Fitting Result Plots #
######################

def visualize_fitting_results(
    standard_results: Dict[str, Any],
    optimized_results: Dict[str, Any],
    top_genes_data: List[np.ndarray],
    top_gene_names: List[str],
    time_points: np.ndarray,
    output_dir: Union[str, Path],
    max_genes_to_plot: int = 10
) -> None:
    """
    Visualize the results of spline fitting, comparing standard and optimized approaches.

    Parameters
    ----------
    standard_results : dict
        Results from standard spline fitting
    optimized_results : dict
        Results from DTW-optimized spline fitting
    top_genes_data : list of arrays
        Data for each gene (list of 3D arrays)
    top_gene_names : list of str
        Names of genes
    time_points : array
        Time points for plotting
    output_dir : str or Path
        Directory to save plots
    max_genes_to_plot : int, optional (default=10)
        Maximum number of genes to plot

    Returns
    -------
    None
        Saves plots to output_dir
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a subdirectory for gene plots
    gene_plots_dir = output_dir / "gene_plots"
    gene_plots_dir.mkdir(exist_ok=True)
    
    # Only visualize up to max_genes_to_plot genes for clarity
    vis_genes = min(max_genes_to_plot, len(top_gene_names))
    
    # Get fine time points for plotting
    standard_fine_times = standard_results['time_points']
    optimized_fine_times = optimized_results['time_points']
    
    # Visualize fits for each top gene
    for i in range(vis_genes):
        gene_name = top_gene_names[i]
        file_path = gene_plots_dir / f"spline_comparison_{gene_name}.png"
        
        # Create a 1x2 subplot for standard vs optimized
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot standard fit
        ax = axes[0]
        for batch in range(top_genes_data[i].shape[0]):
            ax.plot(time_points, top_genes_data[i][batch, :, 0], 'o', alpha=0.3, markersize=3)
        
        # Plot fitted trajectory
        ax.plot(standard_results['time_points'], standard_results['fitted_trajectories'][:, i],
                'r-', linewidth=2, label=f'Standard Fit')
        
        ax.set_title(f"Gene {gene_name} - Standard Spline\nDTW: {standard_results['dtw_distances'][i]:.3f}, Smoothing: 0.5")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Expression")
        
        # Plot optimized fit
        ax = axes[1]
        for batch in range(top_genes_data[i].shape[0]):
            ax.plot(time_points, top_genes_data[i][batch, :, 0], 'o', alpha=0.3, markersize=3)
        
        # Plot fitted trajectory
        ax.plot(optimized_results['time_points'], optimized_results['fitted_trajectories'][:, i],
                'g-', linewidth=2, label=f'DTW Optimized')
        
        optimized_smoothing = optimized_results['smoothing_values'][i]
        ax.set_title(f"Gene {gene_name} - DTW Optimized Spline\nDTW: {optimized_results['dtw_distances'][i]:.3f}, Smoothing: {optimized_smoothing:.3f}")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Pseudotime")
        
        plt.tight_layout()
        # Save the figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot distribution of smoothing values
    file_path = output_dir / "smoothing_distribution.png"
    fig = plt.figure(figsize=(10, 6))
    plt.hist(optimized_results['smoothing_values'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0.5, color='r', linestyle='--', linewidth=2, label='Standard smoothing value')
    plt.title(f'Distribution of Optimized Smoothing Values for {len(top_gene_names)} Genes')
    plt.xlabel('Smoothing Value')
    plt.ylabel('Number of Genes')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # Save the figure
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # If there are not too many genes, also create a scatter plot for individual genes
    display_genes = min(50, len(top_gene_names))
    if display_genes > 5:  # Only create this plot if there are enough genes
        # Create a scatter plot of smoothing values (for top 50 genes max)
        file_path = output_dir / "smoothing_values_per_gene.png"
        plt.figure(figsize=(max(10, display_genes * 0.3), 6))
        plt.scatter(range(display_genes), optimized_results['smoothing_values'][:display_genes], c='g', marker='o')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Standard smoothing value')
        plt.xticks(range(display_genes), top_gene_names[:display_genes], rotation=90)
        plt.title('Optimized Smoothing Values by Gene')
        plt.xlabel('Gene')
        plt.ylabel('Smoothing Value')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        # Save the figure
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

#########################
# Trajectory Fitting Plots #
#########################

def plot_fitted_trajectories(
    fitter: Any,  # TrajectoryFitter instance
    model_type: Optional[str] = None,
    feature_indices: Optional[List[int]] = None,
    n_samples: int = 3,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot fitted trajectories against original data.

    Parameters
    ----------
    fitter : TrajectoryFitter
        Fitted TrajectoryFitter instance
    model_type : str, optional
        Model type to plot. If None, uses the best model.
    feature_indices : list of int, optional
        Indices of features to plot. If None, selects a few representative features.
    n_samples : int, default=3
        Number of sample trajectories to plot
    figsize : tuple, default=(16, 12)
        Figure size
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure
    """
    # Determine model type to plot
    if model_type is None:
        # Use the best model if available
        if hasattr(fitter, '_best_model'):
            model_type = fitter._best_model
        # Otherwise use the first model in results
        elif hasattr(fitter, 'results') and fitter.results:
            model_type = list(fitter.results.keys())[0]
        else:
            raise ValueError("No model results available in fitter.")
    
    # Get the results for this model
    if not hasattr(fitter, 'results') or model_type not in fitter.results:
        raise ValueError(f"Model type {model_type} not found in fitter results.")
    
    # Extract data
    results = fitter.results[model_type]
    fitted_trajectories = results['fitted_trajectories']
    dtw_distances = results['dtw_distances']
    n_features = fitted_trajectories.shape[1]
    
    # Determine features to plot
    if feature_indices is None:
        # Choose features based on DTW distances (best, worst, and some in between)
        sorted_indices = np.argsort(dtw_distances)
        n_to_plot = min(9, n_features)
        feature_indices = np.linspace(0, len(sorted_indices)-1, n_to_plot, dtype=int)
        feature_indices = sorted_indices[feature_indices]
    
    # Create figure
    n_rows = int(np.ceil(len(feature_indices) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    if n_rows == 1:
        axes = [axes]  # Ensure axes is a 2D array
    
    # Plot each feature
    for i, feature_idx in enumerate(feature_indices):
        ax = axes[i // 3][i % 3]
        
        # Plot fitted trajectory
        ax.plot(fitter.fine_time_points, fitted_trajectories[:, feature_idx],
                'r-', linewidth=2, label=f'Fitted ({model_type})')
        
        # Plot title with feature index and DTW distance
        if hasattr(fitter, 'feature_names') and fitter.feature_names is not None:
            feature_name = fitter.feature_names[feature_idx]
            ax.set_title(f"Feature: {feature_name}\nDTW: {dtw_distances[feature_idx]:.4f}")
        else:
            ax.set_title(f"Feature: {feature_idx}\nDTW: {dtw_distances[feature_idx]:.4f}")
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(feature_indices), n_rows * 3):
        axes[i // 3][i % 3].set_visible(False)
    
    plt.suptitle(f"Fitted Trajectories - {model_type.capitalize()} Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_clusters(
    clustering_results: Dict,
    time_points: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot feature clusters based on trajectory patterns.

    Parameters
    ----------
    clustering_results : dict
        Clustering results containing 'labels', 'cluster_centers', etc.
    time_points : array-like
        Time points for plotting
    figsize : tuple, default=(12, 5)
        Figure size
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure
    """
    # Extract clustering data
    cluster_centers = clustering_results['cluster_centers']
    labels = clustering_results['labels']
    n_clusters = len(cluster_centers)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot cluster centers
    ax = axes[0]
    for i in range(n_clusters):
        ax.plot(time_points, cluster_centers[i], label=f'Cluster {i+1}')
    
    ax.set_title(f"Cluster Centers (n={n_clusters})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot cluster sizes
    ax = axes[1]
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    bars = ax.bar(range(n_clusters), cluster_sizes)
    
    # Add count labels on bars
    for bar, count in zip(bars, cluster_sizes):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{count}',
            ha='center',
            va='bottom'
        )
    
    ax.set_title("Cluster Sizes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Features")
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f'Cluster {i+1}' for i in range(n_clusters)])
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_model_comparison(
    results_dict: Dict[str, Dict],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot comparison of different models based on their performance.

    Parameters
    ----------
    results_dict : dict
        Dictionary with model types as keys and result dictionaries as values
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure
    """
    # Extract model scores
    model_types = []
    model_scores = []
    
    for model_type, results in results_dict.items():
        if 'model_score' in results:
            model_types.append(model_type.capitalize())
            model_scores.append(results['model_score'])
    
    # Sort models by performance
    sorted_indices = np.argsort(model_scores)
    model_types = [model_types[i] for i in sorted_indices]
    model_scores = [model_scores[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.barh(model_types, model_scores, color='skyblue')
    
    # Add value labels
    for bar, score in zip(bars, model_scores):
        width = bar.get_width()
        ax.text(
            width * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va='center'
        )
    
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model Score (higher is better)")
    ax.set_ylabel("Model Type")
    ax.grid(axis='x', alpha=0.3)
    
    # Add note about what the score represents
    ax.text(
        0.01, -0.1,
        "Note: Model score = negative mean DTW distance (higher is better)",
        transform=ax.transAxes,
        fontsize=10,
        ha='left'
    )
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_feature_variation(
    data: np.ndarray,
    variation_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot feature variation across samples.

    Parameters
    ----------
    data : array-like
        3D data array (samples, time_points, features)
    variation_threshold : float, optional
        Threshold to mark in the plot
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure
    """
    # Calculate variation for each feature
    from .cell_interpolation import calculate_trajectory_variation
    
    n_samples, n_time_points, n_features = data.shape
    feature_variations = []
    
    for feature_idx in range(n_features):
        feature_data = data[:, :, feature_idx]
        variations = []
        
        for sample_idx in range(n_samples):
            sample_variation = calculate_trajectory_variation(feature_data[sample_idx])
            variations.append(sample_variation)
        
        feature_variations.append(np.mean(variations))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    ax.hist(feature_variations, bins=30, alpha=0.7, color='skyblue')
    
    # Add threshold line if provided
    if variation_threshold is not None:
        ax.axvline(
            x=variation_threshold,
            color='red',
            linestyle='--',
            label=f'Threshold: {variation_threshold}'
        )
        
        # Count features above threshold
        n_above = sum(var >= variation_threshold for var in feature_variations)
        ax.text(
            0.95, 0.95,
            f"Features above threshold: {n_above}/{n_features} ({n_above/n_features*100:.1f}%)",
            transform=ax.transAxes,
            ha='right',
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    ax.set_title("Feature Variation Distribution")
    ax.set_xlabel("Mean Variation Across Samples")
    ax.set_ylabel("Number of Features")
    ax.grid(alpha=0.3)
    
    if variation_threshold is not None:
        ax.legend()
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

############################
# Gene Trajectory Plots #
############################

def plot_gene_trajectory(
    reshaped_data: np.ndarray,
    gene_idx: int,
    gene_name: Optional[str] = None,
    batch_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "viridis",
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Plot the expression trajectory of a single gene across pseudotime for all batches.

    Parameters
    ----------
    reshaped_data : numpy.ndarray
        3D array with shape (batch, pseudotime, gene)
    gene_idx : int
        Index of the gene to plot
    gene_name : str, optional
        Name of the gene for plot title
    batch_labels : list of str, optional
        Names of the batches for the legend
    figsize : tuple, default=(10, 6)
        Figure size
    cmap : str, default="viridis"
        Colormap for the batch lines
    save_path : str or Path, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Validate inputs
    if gene_idx < 0 or gene_idx >= reshaped_data.shape[2]:
        raise ValueError(f"Gene index {gene_idx} is out of range (0-{reshaped_data.shape[2]-1})")
    
    # Set default gene name and batch labels if not provided
    if gene_name is None:
        gene_name = f"Gene {gene_idx}"
    
    if batch_labels is None:
        batch_labels = [f"Batch {i+1}" for i in range(reshaped_data.shape[0])]
    
    # Ensure we have the right number of batch labels
    if len(batch_labels) < reshaped_data.shape[0]:
        # Extend with default labels if we don't have enough
        batch_labels.extend([f"Batch {i+1}" for i in range(len(batch_labels), reshaped_data.shape[0])])
    elif len(batch_labels) > reshaped_data.shape[0]:
        # Truncate if we have too many
        batch_labels = batch_labels[:reshaped_data.shape[0]]
    
    # Extract data for the specified gene
    n_batches, n_timepoints, _ = reshaped_data.shape
    gene_data = reshaped_data[:, :, gene_idx]
    
    # Create pseudotime axis
    pseudotime = np.linspace(0, 1, n_timepoints)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.get_cmap(cmap, n_batches)
    
    # Plot each batch
    for batch_idx in range(n_batches):
        ax.plot(pseudotime, gene_data[batch_idx], 
                label=batch_labels[batch_idx],
                color=colors(batch_idx), 
                linewidth=2.5, 
                alpha=0.8)
    
    # Add plot details
    ax.set_title(f"Expression trajectory of {gene_name} across pseudotime", fontsize=14)
    ax.set_xlabel("Pseudotime", fontsize=12)
    ax.set_ylabel("Expression level", fontsize=12)
    ax.legend(title="Batches")
    ax.grid(True, alpha=0.3)
    
    # Apply some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the figure if specified
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    return fig

def plot_gene_by_name(
    gene_name: str,
    reshaped_data: np.ndarray,
    filtered_genes: Union[List[str], np.ndarray],
    batch_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 7),
    cmap: str = "plasma",
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Plot a gene expression trajectory by gene name instead of index.
    
    This function finds the gene index by name and passes it to plot_gene_trajectory.
    
    Parameters
    ----------
    gene_name : str
        Name of the gene to plot
    reshaped_data : numpy.ndarray
        3D array with shape (batch, pseudotime, gene)
    filtered_genes : list or numpy.ndarray
        List of gene names corresponding to the third dimension of reshaped_data
    batch_names : list of str, optional
        Names of the batches for the legend
    figsize : tuple, default=(12, 7)
        Figure size (width, height) in inches
    cmap : str, default="plasma"
        Colormap for the batch lines
    save_path : str or Path, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Find the gene index by name
    filtered_genes = np.array(filtered_genes)
    
    try:
        gene_index = np.where(filtered_genes == gene_name)[0][0]
    except IndexError:
        raise ValueError(f"Gene '{gene_name}' not found in filtered_genes")
    
    # Plot the gene trajectory
    fig = plot_gene_trajectory(
        reshaped_data=reshaped_data,
        gene_idx=gene_index,
        gene_name=gene_name,
        batch_labels=batch_names,
        figsize=figsize,
        cmap=cmap,
        save_path=save_path
    )
    
    return fig 

# Example usage of gene trajectory visualization functions:
'''
# Example 1: Plot a gene by index
fig1 = plot_gene_trajectory(
    reshaped_data=reshaped_data,  # 3D array (batch, pseudotime, gene)
    gene_idx=5,                   # Plot the 6th gene
    gene_name="FOXP3",            # Optional: specify a gene name for the title
    batch_labels=["Condition A", "Condition B"],  # Optional: custom batch labels
    figsize=(12, 6),              # Optional: customize figure size
    cmap="viridis",               # Optional: specify colormap
    save_path="gene5_trajectory.png"  # Optional: save the figure
)

# Example 2: Plot a gene by name
fig2 = plot_gene_by_name(
    gene_name="FOXP3",            # Gene name to plot
    reshaped_data=reshaped_data,  # 3D array (batch, pseudotime, gene)
    filtered_genes=gene_names,    # List/array of gene names matching the 3rd dimension
    batch_names=["Day 0", "Day 3", "Day 7"],  # Optional: batch names for legend
    figsize=(14, 8),              # Optional: customize figure size
    cmap="plasma",                # Optional: specify colormap
    save_path="FOXP3_trajectory.png"  # Optional: save the figure
)

# Example 3: Plot multiple genes in a loop
genes_of_interest = ["FOXP3", "IL2RA", "CTLA4"]
for gene in genes_of_interest:
    try:
        fig = plot_gene_by_name(
            gene_name=gene,
            reshaped_data=reshaped_data,
            filtered_genes=gene_names,
            save_path=f"{gene}_trajectory.png"
        )
        plt.close(fig)  # Close figure to free memory
    except ValueError as e:
        print(f"Error plotting {gene}: {e}")
''' 