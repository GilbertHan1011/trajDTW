#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple example script for using the trajDTW package
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Import from the trajDTW package
from trajDTW import (
    anndata_to_3d_matrix, 
    calculate_trajectory_conservation,
    TrajectoryFitter,
    run_trajectory_conservation_analysis
)

# Set output directory
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

print("\n=== Simple trajDTW Example ===\n")

# Sample AnnData path - replace with your own data
adata_path = "../data/example_data.h5ad"

if os.path.exists(adata_path):
    print("\n=== METHOD 1: Step-by-step approach ===\n")
    # Load AnnData
    print("Loading AnnData...")
    adata = sc.read_h5ad(adata_path)
    print(f"AnnData shape: {adata.shape}")
    
    # Print available columns in obs
    print("\nAvailable columns in adata.obs:")
    for col in adata.obs.columns:
        print(f"  - {col}")
    
    # Convert to 3D matrix - adjust column names according to your data
    print("\nConverting to 3D matrix using Gaussian kernel interpolation...")
    result = anndata_to_3d_matrix(
        adata=adata,
        pseudo_col='pseudotime',  # Column containing pseudotime
        batch_col='sample',      # Column containing batch information
        n_bins=100,              # Number of interpolation points
        adaptive_kernel=True     # Use adaptive kernel width
    )
    
    # Extract results
    reshaped_data = result['reshaped_data']  # 3D array (batch x time x gene)
    filtered_genes = result['filtered_genes']
    batch_names = result['batch_names']
    
    print(f"\n3D matrix shape: {reshaped_data.shape}")
    print(f"Number of batches: {len(batch_names)}")
    print(f"Number of genes: {len(filtered_genes)}")
    
    # Calculate conservation scores
    print("\nCalculating conservation scores...")
    conservation_results = calculate_trajectory_conservation(
        trajectory_data=reshaped_data,
        gene_names=filtered_genes,
        normalize='zscore'
    )
    
    # Print top conserved genes
    print("\nTop 10 most conserved genes:")
    print(conservation_results['conservation_scores'].head(10))
    
    # Fit trajectory for top gene
    top_gene = conservation_results['conservation_scores'].iloc[0]['gene']
    gene_idx = np.where(filtered_genes == top_gene)[0][0]
    
    print(f"\nFitting trajectory for top gene: {top_gene}")
    
    # Create time points
    time_points = np.linspace(0, 1, reshaped_data.shape[1])
    
    # Extract data for this gene
    gene_data = reshaped_data[:, :, gene_idx:gene_idx+1]
    
    # Initialize and run TrajectoryFitter
    fitter = TrajectoryFitter(time_points=time_points, verbose=True)
    
    # Fit with different models
    models = ['spline', 'polynomial', 'sine']
    
    for model_type in models:
        print(f"\nFitting {model_type} model...")
        results = fitter.fit(
            gene_data,
            model_type=model_type,
            spline_degree=3,
            polynomial_degree=4
        )
        
        print(f"  DTW distance: {np.mean(results['dtw_distances']):.4f}")
    
    print("\n=== METHOD 2: All-in-one approach with run_trajectory_conservation_analysis ===\n")
    # Create a different output directory for the all-in-one approach
    one_step_dir = output_dir / "one_step_results"
    one_step_dir.mkdir(exist_ok=True)
    
    print(f"Running the full pipeline analysis using run_trajectory_conservation_analysis...")
    print(f"Results will be saved to: {one_step_dir}")
    
    # Run the complete pipeline analysis
    results = run_trajectory_conservation_analysis(
        adata_path=adata_path,
        output_dir=one_step_dir,
        pseudo_col='pseudotime',
        batch_col='sample',
        n_bins=100,
        adaptive_kernel=True,
        gene_thred=0.1,
        batch_thred=0.3,
        top_n_genes=10,
        spline_smoothing=0.5,
        n_jobs=4,
        save_figures=True
    )
    
    # Access and display the results
    print("\nAll-in-one pipeline results:")
    print(f"Number of filtered genes: {len(results['filtered_genes'])}")
    print(f"Number of selected top genes: {len(results['selected_genes'])}")
    
    # Print top conserved genes
    print("\nTop 10 most conserved genes from full pipeline:")
    print(results['conservation_results']['conservation_scores'].head(10))
    
    # Print spline fitting performance
    standard_results = results['fit_results']['standard_results']
    optimized_results = results['fit_results']['optimized_results']
    
    print("\nSpline Fitting Results Comparison:")
    print(f"Standard approach - mean DTW distance: {standard_results['mean_dtw_distance']:.4f}")
    print(f"DTW-optimized approach - mean DTW distance: {optimized_results['mean_dtw_distance']:.4f}")
    
    improvement = standard_results['mean_dtw_distance'] - optimized_results['mean_dtw_distance']
    percent_improvement = 100 * improvement / standard_results['mean_dtw_distance']
    print(f"Improvement: {improvement:.4f}")
    print(f"Percentage improvement: {percent_improvement:.2f}%")
    
    print("\nExample completed successfully!")
    print(f"Various plots and results files saved to: {one_step_dir}")
    
else:
    print(f"Error: Could not find example data at {adata_path}")
    print("Please provide a valid h5ad file path to run this example.") 