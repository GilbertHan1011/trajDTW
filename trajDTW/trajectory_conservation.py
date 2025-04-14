#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Conservation Module

This module provides functions for calculating conservation scores of gene expression 
trajectories across samples/batches using Dynamic Time Warping (DTW) distance metrics.
It helps identify genes with consistent temporal expression patterns across different
samples or experimental conditions.
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Import utility functions
from .trajectory_utils import (
    normalize_trajectory,
    calculate_trajectory_variation,
    compute_dtw,
    optimized_dtw
)

# Try to import fastdtw, but handle it if not available
try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None
    warnings.warn("fastdtw not available. DTW computations will be slower.")

def calculate_trajectory_conservation(trajectory_data: np.ndarray, 
                                      gene_names: Optional[Union[List[str], np.ndarray]] = None, 
                                      save_dir: Optional[Union[str, Path]] = None, 
                                      prefix: str = "conservation",
                                      dtw_radius: int = 3, 
                                      use_fastdtw: bool = True,
                                      normalize: str = 'zscore', 
                                      filter_samples_by_variation: bool = True,
                                      variation_threshold: float = 0.1,
                                      variation_metric: str = 'max',
                                      min_valid_samples: int = 2) -> Dict:
    """
    Calculate conservation of trajectories across samples using DTW distance.
    
    Parameters
    ----------
    trajectory_data : numpy.ndarray
        3D array with shape (sample, pseudotime, gene) or (sample, gene, pseudotime)
        The function will detect and handle the axis order.
    gene_names : list or numpy.ndarray, optional
        Names of genes corresponding to the gene axis. If None, uses indices.
    save_dir : str or pathlib.Path, optional
        Directory to save results. If None, results are not saved.
    prefix : str, optional
        Prefix for saved files.
    dtw_radius : int, optional
        Radius parameter for fastdtw to speed up computation.
    use_fastdtw : bool, optional
        Whether to use fastdtw (faster) or scipy's dtw (more accurate).
    normalize : str, optional
        Method to normalize trajectories before DTW: 'zscore', 'minmax', 'cv', or 'none'
        - zscore: Standardize to mean=0, std=1 (recommended)
        - minmax: Scale to range [0, 1]
        - cv: Divide by mean (coefficient of variation)
        - none: No normalization
    filter_samples_by_variation : bool, optional
        Whether to filter out samples with too little variation for each gene
    variation_threshold : float, optional
        Minimum variation required for a sample to be included
    variation_metric : str, optional
        Metric to use for variation: 'cv', 'std', 'range', 'max', or 'mad'
    min_valid_samples : int, optional
        Minimum number of valid samples required after filtering (default 2)
        
    Returns:
    -------
    dict
        Dictionary containing:
        - pairwise_distances: Dictionary of dataframes with pairwise DTW distances for each gene
        - conservation_scores: DataFrame with conservation scores for all genes
        - similarity_matrix: Similarity matrix based on pairwise distances
        - metadata: Additional information about the calculation
        - filtering_info: Information about which samples were filtered (if filtering enabled)
    """
    # Check if fastdtw is available if requested
    if use_fastdtw and fastdtw is None:
        warnings.warn("fastdtw not installed but use_fastdtw=True. Switching to slower DTW.")
        use_fastdtw = False
    
    # Detect the shape and orientation of the input data
    # Expected shape is (sample, time, gene) but we'll handle other orientations
    shape = trajectory_data.shape
    
    if len(shape) != 3:
        raise ValueError(f"Input data must be a 3D array, got shape {shape}")
        
    # Try to detect the orientation
    # If we have many more points in axis 1 than axis 2, 
    # the data might be (sample, pseudotime, gene)
    if shape[1] > shape[2] * 2:  # Heuristic: pseudotime has at least 2x more points than genes
        orientation = "sample_time_gene"
        n_samples, n_timepoints, n_genes = shape
        
        # No need to transpose, already in correct format
        data = trajectory_data
        
    # If we have more points in axis 2, it might be (sample, gene, pseudotime)    
    elif shape[2] > shape[1] * 2:  # Heuristic: pseudotime has at least 2x more points than genes
        orientation = "sample_gene_time"
        n_samples, n_genes, n_timepoints = shape
        
        # Transpose to get (sample, time, gene)
        data = np.transpose(trajectory_data, (0, 2, 1))
        
    # If the heuristic isn't clear, we'll assume (sample, time, gene)
    else:
        orientation = "assumed_sample_time_gene"
        n_samples, n_timepoints, n_genes = shape
        data = trajectory_data
        print(f"Warning: Ambiguous data orientation. Assuming shape is (sample={n_samples}, "
              f"pseudotime={n_timepoints}, gene={n_genes})")
    
    # Set up gene names
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
    elif len(gene_names) != n_genes:
        raise ValueError(f"Length of gene_names ({len(gene_names)}) doesn't match "
                         f"number of genes in data ({n_genes})")

    # Initialize results storage
    pairwise_distances = {}
    conservation_scores = np.zeros(n_genes)
    sample_pairs = [(i, j) for i in range(n_samples) for j in range(i+1, n_samples)]
    n_pairs = len(sample_pairs)
    
    print(f"Calculating pairwise DTW distances for {n_genes} genes across {n_samples} samples "
          f"({n_pairs} pairwise comparisons per gene)...")
    print(f"Using normalization method: {normalize}")
    
    if filter_samples_by_variation:
        print(f"Filtering samples by variation: threshold={variation_threshold}, metric={variation_metric}")
    
    # Dictionary to store filtering information
    samples_included = {}
    filtered_genes = []
    
    # Calculate pairwise DTW distances for each gene
    for gene_idx in range(n_genes):
        gene_name = gene_names[gene_idx]
        
        # Initialize distance matrix for this gene
        dist_matrix = np.zeros((n_samples, n_samples))
        np.fill_diagonal(dist_matrix, 0)  # Set diagonal to 0 (self-distance)
        
        # Filter samples by variation if requested
        if filter_samples_by_variation:
            # Calculate variation for each sample's trajectory
            sample_variations = np.array([
                calculate_trajectory_variation(
                    data[i, :, gene_idx], 
                    metric=variation_metric
                )
                for i in range(n_samples)
            ])
            
            # Create mask for samples with sufficient variation
            valid_samples = sample_variations >= variation_threshold
            valid_sample_indices = np.where(valid_samples)[0]
            
            # Store which samples were included
            samples_included[gene_name] = {
                'sample_indices': valid_sample_indices.tolist(),
                'variations': sample_variations.tolist(),
                'n_valid': np.sum(valid_samples)
            }
            
            # Skip gene if too few valid samples
            if len(valid_sample_indices) < min_valid_samples:
                filtered_genes.append(gene_name)
                conservation_scores[gene_idx] = np.nan  # Use NaN for filtered genes
                continue
                
            # Create list of valid sample pairs
            valid_sample_pairs = [
                (i, j) for i in valid_sample_indices 
                for j in valid_sample_indices if i < j
            ]
        else:
            # Use all samples if not filtering
            valid_sample_pairs = sample_pairs
            samples_included[gene_name] = {
                'sample_indices': list(range(n_samples)),
                'variations': None,  # Not calculated
                'n_valid': n_samples
            }
        
        # Check if we have any valid pairs
        n_valid_pairs = len(valid_sample_pairs)
        if n_valid_pairs == 0:
            conservation_scores[gene_idx] = np.nan
            filtered_genes.append(gene_name)
            continue
            
        # Calculate distances only for valid sample pairs
        pair_distances = []
        for i, j in valid_sample_pairs:
            # Extract trajectories
            traj_i = data[i, :, gene_idx]
            traj_j = data[j, :, gene_idx]
            
            # Compute DTW distance using imported utility
            distance = compute_dtw(traj_i, traj_j, radius=dtw_radius, 
                                  norm_method=normalize, use_fastdtw=use_fastdtw)
            
            # Store in the distance matrix
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
            
            # Add to pair distances for conservation score
            pair_distances.append(distance)
        
        # Store as DataFrame
        dist_df = pd.DataFrame(
            dist_matrix,
            index=[f"Sample_{i}" for i in range(n_samples)],
            columns=[f"Sample_{i}" for i in range(n_samples)]
        )
        pairwise_distances[gene_name] = dist_df
        
        # Compute conservation score (negative mean distance)
        if len(pair_distances) > 0:
            conservation_scores[gene_idx] = -np.mean(pair_distances)
        else:
            conservation_scores[gene_idx] = np.nan
            
        # Print progress
        if (gene_idx + 1) % 10 == 0 or gene_idx == n_genes - 1:
            print(f"Processed {gene_idx + 1}/{n_genes} genes")
    
    # Handle NaN values (filtered genes)
    non_nan_indices = ~np.isnan(conservation_scores)
    if np.any(non_nan_indices):
        # Normalize only non-NaN scores
        non_nan_scores = conservation_scores[non_nan_indices]
        min_score = np.min(non_nan_scores)
        
        # Shift to positive range if needed
        shifted_scores = np.full_like(conservation_scores, np.nan)
        if min_score < 0:
            shifted_scores[non_nan_indices] = non_nan_scores - min_score
        else:
            shifted_scores[non_nan_indices] = non_nan_scores
        
        # Scale to [0, 1]
        max_score = np.nanmax(shifted_scores)
        normalized_scores = np.full_like(conservation_scores, np.nan)
        if max_score > 0:  # Avoid division by zero
            normalized_scores[non_nan_indices] = shifted_scores[non_nan_indices] / max_score
        else:
            normalized_scores[non_nan_indices] = shifted_scores[non_nan_indices]
    else:
        # All genes were filtered
        normalized_scores = conservation_scores
    
    # Create conservation score DataFrame with filtering information
    conservation_df = pd.DataFrame({
        'gene': gene_names,
        'raw_score': conservation_scores,
        'normalized_score': normalized_scores,
        'n_valid_samples': [samples_included.get(g, {}).get('n_valid', 0) for g in gene_names],
        'was_filtered': [g in filtered_genes for g in gene_names]
    })
    # Sort only by non-NaN scores
    conservation_df = conservation_df.sort_values(
        'normalized_score', 
        ascending=False, 
        na_position='last'  # Put NaN values at the end
    )
    
    # Calculate overall similarity matrix across all genes (using only non-filtered genes)
    # This represents how similar the overall gene trajectories are between samples
    overall_similarity = np.zeros((n_samples, n_samples))
    non_filtered_genes = [g for g in gene_names if g not in filtered_genes]
    
    # For each pair of samples, calculate mean normalized distance across all genes
    for i, j in sample_pairs:
        gene_distances = []
        for gene_name in non_filtered_genes:
            # Only include if both samples were valid for this gene
            sample_indices = samples_included.get(gene_name, {}).get('sample_indices', [])
            if i in sample_indices and j in sample_indices:
                distance = pairwise_distances[gene_name].iloc[i, j]
                gene_distances.append(distance)
        
        # Calculate similarity only if we have distances
        if gene_distances:
            # Convert to similarity (higher is more similar)
            mean_distance = np.mean(gene_distances)
            # Simple conversion to similarity: exp(-distance)
            similarity = np.exp(-mean_distance)
            
            overall_similarity[i, j] = similarity
            overall_similarity[j, i] = similarity
    
    # Set diagonal to 1 (perfect similarity with self)
    np.fill_diagonal(overall_similarity, 1.0)
    
    # Create similarity DataFrame
    similarity_df = pd.DataFrame(
        overall_similarity,
        index=[f"Sample_{i}" for i in range(n_samples)],
        columns=[f"Sample_{i}" for i in range(n_samples)]
    )
    
    # If saving is enabled, create visualizations
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Import visualization functions
        from .visualization import (
            plot_conservation_scores, 
            plot_sample_similarity, 
            plot_variation_distribution
        )
        
        # Plot conservation scores
        plot_conservation_scores(
            conservation_scores=conservation_df,
            save_dir=save_dir,
            prefix=prefix,
            normalize=normalize,
            variation_threshold=variation_threshold,
            variation_metric=variation_metric
        )
        
        # Plot sample similarity
        plot_sample_similarity(
            similarity_matrix=overall_similarity,
            save_dir=save_dir,
            prefix=prefix,
            sample_names=[f"Sample_{i}" for i in range(n_samples)]
        )
        
        # If variation filtering was applied, plot the distribution
        if filter_samples_by_variation:
            # Collect all variations to plot distribution
            all_variations = []
            for info in samples_included.values():
                if info.get('variations') is not None:
                    all_variations.extend(info['variations'])
                    
            if all_variations:
                plot_variation_distribution(
                    all_variations=np.array(all_variations),
                    variation_threshold=variation_threshold,
                    save_dir=save_dir,
                    prefix=prefix,
                    variation_metric=variation_metric
                )
    
    # Return results
    results = {
        'pairwise_distances': pairwise_distances,
        'conservation_scores': conservation_df,
        'similarity_matrix': similarity_df,
        'metadata': {
            'orientation': orientation,
            'n_samples': n_samples,
            'n_genes': n_genes,
            'n_timepoints': n_timepoints,
            'normalization': normalize,
            'filter_samples_by_variation': filter_samples_by_variation,
            'variation_threshold': variation_threshold if filter_samples_by_variation else None,
            'variation_metric': variation_metric if filter_samples_by_variation else None,
            'min_valid_samples': min_valid_samples if filter_samples_by_variation else None,
            'n_filtered_genes': len(filtered_genes) if filter_samples_by_variation else 0,
        },
        'filtering_info': {
            'samples_included': samples_included,
            'filtered_genes': filtered_genes
        } if filter_samples_by_variation else None
    }
    
    return results


def get_most_conserved_samples(pairwise_distances: Dict, n_samples: int, fraction: float = 0.5) -> Dict:
    """
    For each gene, identify the most conserved samples based on pairwise distances.
    
    This is important because:
    1. Not all samples express a gene in the same conserved pattern
    2. Using only the most conserved samples can reduce noise and improve fitting
    3. It allows gene-specific sample selection rather than a one-size-fits-all approach
    
    Parameters:
    -----------
    pairwise_distances : dict
        Dictionary of pandas DataFrames with pairwise distances for each gene
    n_samples : int
        Total number of samples
    fraction : float, optional (default=0.5)
        Fraction of samples to select (e.g., 0.5 for half)
        
    Returns:
    --------
    conserved_samples : dict
        Dictionary with gene names as keys and lists of most conserved sample indices as values
    """
    conserved_samples = {}
    
    # Check for empty dictionary or n_samples <= 0
    if not pairwise_distances or n_samples <= 0:
        return conserved_samples
    
    for gene_name, dist_df in pairwise_distances.items():
        # Skip empty dataframes
        if dist_df.empty or dist_df.shape[0] == 0 or dist_df.shape[1] == 0:
            continue
        
        # Handle case where dataframe size doesn't match n_samples
        actual_n_samples = min(n_samples, dist_df.shape[0], dist_df.shape[1])
        if actual_n_samples == 0:
            continue
            
        # Calculate mean distance for each sample to all other samples
        mean_distances = []
        for i in range(actual_n_samples):
            try:
                # Extract distances from this sample to all others
                if i < dist_df.shape[0] and dist_df.shape[1] > 0:
                    distances = dist_df.iloc[i, :].values
                    # Calculate mean (excluding self which should be 0)
                    valid_distances = distances[distances > 0]
                    if len(valid_distances) > 0:
                        mean_distances.append((i, np.mean(valid_distances)))
                    else:
                        mean_distances.append((i, np.inf))  # If no valid distances, rank last
            except Exception as e:
                # Skip this sample if there's an error
                continue
        
        # Skip if no valid mean distances
        if not mean_distances:
            continue
            
        # Sort by mean distance (lower is better/more conserved)
        sorted_samples = sorted(mean_distances, key=lambda x: x[1])
        
        # Select the top fraction
        n_select = max(2, min(int(actual_n_samples * fraction), len(sorted_samples)))  # At least 2 samples, but not more than we have
        selected_indices = [idx for idx, _ in sorted_samples[:n_select]]
        
        if selected_indices:  # Only add non-empty lists
            conserved_samples[gene_name] = selected_indices
    
    return conserved_samples 