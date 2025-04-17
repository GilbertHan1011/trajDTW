#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example pipeline script for trajDTW package using run_trajectory_conservation_analysis function
"""

import scanpy as sc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime

# Import the main pipeline function from trajDTW
from trajDTW import run_trajectory_conservation_analysis

# Set output directory
output_dir = Path("../../../../processed_data/toy_data/traj_conservation_results")
output_dir.mkdir(parents=True, exist_ok=True)

print("\n=== Trajectory Conservation Analysis Pipeline ===\n")
print(f"Results will be saved to: {output_dir}")

# Define the path to your data
adata_path = "../../../../processed_data/toy_data/20250412_example_trajconserve.h5ad"

# Run the complete trajectory conservation analysis pipeline
results = run_trajectory_conservation_analysis(
    adata_path=adata_path,
    output_dir=output_dir,
    pseudo_col='pseudo',     # Column containing pseudotime values
    batch_col='Sample',      # Column containing batch information
    n_bins=100,              # Number of interpolation points
    adaptive_kernel=True,    # Use adaptive kernel width
    gene_thred=0.1,          # Filter genes expressed in at least 10% of bins
    batch_thred=0.3,         # Filter batches covering at least 30% of timeline
    tail_num=0.05,           # Size of tail region
    ensure_tail=True,        # Ensure batches cover the tail region
    dtw_radius=3,            # Radius parameter for fastdtw
    use_fastdtw=True,        # Use fastdtw algorithm
    normalize='zscore',      # Normalize trajectories before DTW calculation
    variation_filter_level='basic',  # Level of filtering for sample variation
    top_n_genes=10,          # Number of top conserved genes to select for fitting
    spline_smoothing=0.5,    # Smoothing parameter for spline fitting
    interpolation_factor=2,  # Factor for increasing time point density
    n_jobs=4,                # Number of parallel jobs
    save_figures=True,       # Whether to save visualization figures
    layer=None               # Layer in anndata to use (None = default X matrix)
)

# Extract results
conservation_results = results['conservation_results']
fit_results = results['fit_results']
filtered_genes = results['filtered_genes']
selected_genes = results['selected_genes']
reshaped_data = results['reshaped_data']

print("\n=== Results Summary ===")
print(f"Number of genes after filtering: {len(filtered_genes)}")
print(f"Number of selected top conserved genes: {len(selected_genes)}")
print(f"Number of batches: {reshaped_data.shape[0]}")
print(f"Number of time points: {reshaped_data.shape[1]}")

# Print top conserved genes
print("\nTop 10 most conserved genes:")
top_conserved = conservation_results['conservation_scores'].head(10)
print(top_conserved)

# Print spline fitting performance
standard_results = fit_results['standard_results']
optimized_results = fit_results['optimized_results']

print("\nSpline Fitting Results Comparison:")
print(f"Standard approach - mean DTW distance: {standard_results['mean_dtw_distance']:.4f}")
print(f"DTW-optimized approach - mean DTW distance: {optimized_results['mean_dtw_distance']:.4f}")

improvement = standard_results['mean_dtw_distance'] - optimized_results['mean_dtw_distance']
percent_improvement = 100 * improvement / standard_results['mean_dtw_distance']
print(f"Improvement: {improvement:.4f}")
print(f"Percentage improvement: {percent_improvement:.2f}%")

print(f"\nAll results saved to: {output_dir}")
print("\n=== Analysis Pipeline Complete ===") 