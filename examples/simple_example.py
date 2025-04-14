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
    TrajectoryFitter
)

# Set output directory
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

print("\n=== Simple trajDTW Example ===\n")

# Sample AnnData path - replace with your own data
adata_path = "../data/example_data.h5ad"

if os.path.exists(adata_path):
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
    
    print("\nExample completed successfully!")
    
else:
    print(f"Error: Could not find example data at {adata_path}")
    print("Please provide a valid h5ad file path to run this example.") 