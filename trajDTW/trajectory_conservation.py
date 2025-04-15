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
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm

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

def _process_gene(gene_idx, gene_name, data, n_samples, sample_pairs, 
                 dtw_radius, normalize, use_fastdtw, 
                 filter_samples_by_variation, variation_threshold, 
                 variation_metric, min_valid_samples):
    """
    Helper function to process a single gene for parallel computation.
    
    Calculates the pairwise DTW distances between samples for a given gene
    and returns all necessary information for the gene.
    
    Returns a tuple with:
    - gene_idx: Index of the gene
    - gene_name: Name of the gene
    - dist_df: DataFrame with pairwise distances
    - conservation_score: Conservation score for the gene
    - is_filtered: Whether the gene was filtered
    - gene_info: Dictionary with sample filtering information
    """
    # Initialize distance matrix for this gene
    dist_matrix = np.zeros((n_samples, n_samples))
    np.fill_diagonal(dist_matrix, 0)  # Set diagonal to 0 (self-distance)
    
    # Store filtering information
    gene_info = {}
    
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
        gene_info['sample_indices'] = valid_sample_indices.tolist()
        gene_info['variations'] = sample_variations.tolist()
        gene_info['n_valid'] = np.sum(valid_samples)
        
        # Skip gene if too few valid samples
        if len(valid_sample_indices) < min_valid_samples:
            return gene_idx, gene_name, dist_matrix, None, True, gene_info
            
        # Create list of valid sample pairs
        valid_sample_pairs = [
            (i, j) for i in valid_sample_indices 
            for j in valid_sample_indices if i < j
        ]
    else:
        # Use all samples if not filtering
        valid_sample_pairs = sample_pairs
        gene_info = {
            'sample_indices': list(range(n_samples)),
            'variations': None,  # Not calculated
            'n_valid': n_samples
        }
    
    # Check if we have any valid pairs
    n_valid_pairs = len(valid_sample_pairs)
    if n_valid_pairs == 0:
        return gene_idx, gene_name, dist_matrix, None, True, gene_info
        
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
    
    # Convert distance matrix to DataFrame
    dist_df = pd.DataFrame(
        dist_matrix,
        index=[f"Sample_{i}" for i in range(n_samples)],
        columns=[f"Sample_{i}" for i in range(n_samples)]
    )
    
    # Compute conservation score (negative mean distance)
    if len(pair_distances) > 0:
        conservation_score = -np.mean(pair_distances)
    else:
        conservation_score = np.nan
        
    return gene_idx, gene_name, dist_df, conservation_score, False, gene_info

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
                                      min_valid_samples: int = 2,
                                      calculate_conserved_samples: bool = True,
                                      conserved_fraction: float = 0.5,
                                      n_jobs: int = -1,
                                      show_progress: bool = True) -> Dict:
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
    calculate_conserved_samples : bool, optional
        Whether to calculate the most conserved samples for each gene (default True)
    conserved_fraction : float, optional
        Fraction of samples to select as most conserved (default 0.5)
    n_jobs : int, optional (default=-1)
        Number of parallel jobs to run. -1 means using all processors.
    show_progress : bool, optional (default=True)
        Whether to show a progress bar during computation.
        
    Returns:
    -------
    dict
        Dictionary containing:
        - pairwise_distances: Dictionary of dataframes with pairwise DTW distances for each gene
        - conservation_scores: DataFrame with conservation scores for all genes
        - similarity_matrix: Similarity matrix based on pairwise distances
        - conserved_samples: Dictionary mapping gene names to indices of most conserved samples
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

    # Create all possible sample pairs once
    sample_pairs = [(i, j) for i in range(n_samples) for j in range(i+1, n_samples)]
    n_pairs = len(sample_pairs)
    
    if show_progress:
        print(f"Calculating pairwise DTW distances for {n_genes} genes across {n_samples} samples "
              f"({n_pairs} pairwise comparisons per gene)...")
        print(f"Using normalization method: {normalize}")
        
        if filter_samples_by_variation:
            print(f"Filtering samples by variation: threshold={variation_threshold}, metric={variation_metric}")
    
    # Determine number of jobs
    n_jobs = n_jobs if n_jobs > 0 else joblib.cpu_count()
    actual_n_jobs = min(n_jobs, n_genes)
    
    if show_progress:
        print(f"Processing {n_genes} genes using {actual_n_jobs} parallel jobs...")

    # Process all genes in parallel
    results = Parallel(n_jobs=actual_n_jobs, verbose=1 if show_progress else 0)(
        delayed(_process_gene)(
            gene_idx, gene_names[gene_idx], data, n_samples, sample_pairs,
            dtw_radius, normalize, use_fastdtw,
            filter_samples_by_variation, variation_threshold, 
            variation_metric, min_valid_samples
        )
        for gene_idx in range(n_genes)
    )
    
    # Initialize storage for results
    pairwise_distances = {}
    conservation_scores = np.zeros(n_genes)
    samples_included = {}
    filtered_genes = []
    
    # Process results
    for gene_idx, gene_name, dist_df, score, is_filtered, gene_info in results:
        if is_filtered:
            filtered_genes.append(gene_name)
            conservation_scores[gene_idx] = np.nan
        else:
            pairwise_distances[gene_name] = dist_df
            conservation_scores[gene_idx] = score
        
        samples_included[gene_name] = gene_info
    
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
            plot_conservation_scores
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

        # If variation filtering was applied, plot the distribution
        if filter_samples_by_variation:
            # Collect all variations to plot distribution
            all_variations = []
            for info in samples_included.values():
                if info.get('variations') is not None:
                    all_variations.extend(info['variations'])
                    
    # Return results
    results = {
        'pairwise_distances': pairwise_distances,
        'conservation_scores': conservation_df,
        'similarity_matrix': similarity_df,
        'conserved_samples': get_most_conserved_samples(pairwise_distances, n_samples, conserved_fraction) if calculate_conserved_samples else None,
        'metadata': {
            'n_samples': n_samples,
            'n_genes': n_genes,
            'n_timepoints': n_timepoints,
            'normalization': normalize,
            'filter_samples_by_variation': filter_samples_by_variation,
            'variation_threshold': variation_threshold if filter_samples_by_variation else None,
            'variation_metric': variation_metric if filter_samples_by_variation else None,
            'min_valid_samples': min_valid_samples if filter_samples_by_variation else None,
            'n_filtered_genes': len(filtered_genes) if filter_samples_by_variation else 0,
            'n_jobs': actual_n_jobs  # Add the number of jobs used
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


def extract_pairwise_distances(
    conservation_results: Dict,
    output_csv: Optional[Union[str, Path]] = None,
    gene_subset: Optional[List[str]] = None,
    sample_subset: Optional[List[int]] = None,
    include_filtered: bool = False
) -> pd.DataFrame:
    """
    Extract pairwise distances from conservation results into a tabular format.
    
    Parameters
    ----------
    conservation_results : dict
        Results dictionary from calculate_trajectory_conservation function
    output_csv : str or Path, optional
        Path to save CSV file. If None, CSV is not saved.
    gene_subset : list, optional
        List of genes to include. If None, all genes are included.
    sample_subset : list, optional
        List of sample indices to include. If None, all samples are included.
    include_filtered : bool, optional
        Whether to include genes that were filtered during conservation analysis
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: gene, sample1, sample2, distance
    """
    # Verify required data is in conservation results
    if not isinstance(conservation_results, dict) or 'pairwise_distances' not in conservation_results:
        raise ValueError("Invalid conservation_results provided. Must be a dictionary with 'pairwise_distances' key.")
    
    pairwise_distances = conservation_results['pairwise_distances']
    
    # Get filtered genes if needed
    filtered_genes = []
    if not include_filtered and 'filtering_info' in conservation_results:
        filtering_info = conservation_results.get('filtering_info', {})
        if filtering_info and 'filtered_genes' in filtering_info:
            filtered_genes = filtering_info['filtered_genes']
    
    # Determine which genes to process
    if gene_subset is not None:
        # Filter to only include specified genes that also exist in the results
        genes_to_process = [gene for gene in gene_subset if gene in pairwise_distances]
    else:
        # Use all genes
        genes_to_process = list(pairwise_distances.keys())
    
    # Filter out filtered genes if required
    if not include_filtered:
        genes_to_process = [gene for gene in genes_to_process if gene not in filtered_genes]
    
    # First, create an empty list to store our data
    rows = []
    
    # Loop through each gene and its distance matrix
    for gene in genes_to_process:
        distance_matrix = pairwise_distances[gene]
        
        # Convert the distance matrix to a DataFrame if it's not already
        if not isinstance(distance_matrix, pd.DataFrame):
            distance_matrix = pd.DataFrame(distance_matrix)
        
        # Filter samples if sample_subset is provided
        if sample_subset is not None:
            # Get valid sample indices that exist in the distance matrix
            valid_indices = [idx for idx in sample_subset if idx < len(distance_matrix.index)]
            
            # If no valid indices after filtering, skip this gene
            if not valid_indices:
                continue
                
            # Filter the distance matrix to only include specified samples
            distance_matrix = distance_matrix.iloc[valid_indices, valid_indices]
        
        # Extract the sample pairs and their distances (upper triangular part only)
        for i, row_idx in enumerate(distance_matrix.index):
            for j, col_idx in enumerate(distance_matrix.columns):
                if j > i:  # Only upper triangular
                    sample1 = row_idx
                    sample2 = col_idx
                    distance = distance_matrix.iloc[i, j]
                    
                    # Add this pair to our list
                    rows.append({
                        'gene': gene,
                        'sample1': sample1,
                        'sample2': sample2,
                        'distance': distance
                    })
    
    # Create a DataFrame from the list
    distance_df = pd.DataFrame(rows)
    
    # Save to CSV if output_csv is provided
    if output_csv is not None:
        output_path = Path(output_csv) if not isinstance(output_csv, Path) else output_csv
        output_path.parent.mkdir(parents=True, exist_ok=True)
        distance_df.to_csv(output_path, index=False)
        print(f"Pairwise distances saved to {output_path}")
    
    return distance_df 