import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler
import anndata
from typing import List, Dict, Tuple, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import joblib
from joblib import Parallel, delayed
import multiprocessing
import warnings
from scipy import sparse
from scipy.spatial.distance import euclidean
from pathlib import Path
import time
from tqdm.auto import tqdm  # Import tqdm for progress bars

# Try to import fastdtw, but handle it if not available
try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None
    warnings.warn("fastdtw not available. DTW computations will be slower.")

# Import utility functions
from .trajectory_utils import (
    normalize_trajectory,
    calculate_trajectory_variation,
    compute_dtw,
    optimized_dtw
)

# Import from the dedicated trajectory_conservation module
from .trajectory_conservation import (
    calculate_trajectory_conservation,
    get_most_conserved_samples
)

class GaussianTrajectoryInterpolator:
    """
    This class provides functionality to convert AnnData objects to 3D matrices (time * sample * gene)
    using Gaussian kernel interpolation similar to the Genes2Genes framework.
    """
    
    def __init__(self, n_bins: int = 100, adaptive_kernel: bool = True, 
                 kernel_window_size: float = 0.1, raising_degree: float = 1.0):
        """
        Initialize the interpolator with parameters.
        
        Parameters
        ----------
        n_bins : int
            Number of interpolation points along pseudotime
        adaptive_kernel : bool
            Whether to use adaptive kernel width based on cell density
        kernel_window_size : float
            Base window size for the Gaussian kernel
        raising_degree : float
            Degree of stretch imposed for adaptive window sizes
        """
        self.n_bins = n_bins
        self.adaptive_kernel = adaptive_kernel
        self.kernel_window_size = kernel_window_size
        self.raising_degree = raising_degree
        self.interpolation_points = np.linspace(0, 1, n_bins)
        
    def compute_abs_timediff_mat(self, cell_pseudotimes):
        """
        Compute absolute time differences between interpolation points and cell pseudotimes.
        
        Parameters
        ----------
        cell_pseudotimes : array-like
            Pseudotime values for each cell
            
        Returns
        -------
        pandas.DataFrame
            Matrix of absolute differences
        """
        df_list = []
        for t in self.interpolation_points:
            # Calculate absolute difference between each cell's pseudotime and the interpolation point
            abs_dist = np.abs(np.asarray(cell_pseudotimes) - t)
            df_list.append(abs_dist)
        
        return np.array(df_list)
    
    def compute_cell_densities(self, cell_pseudotimes):
        """
        Compute cell density estimates at each interpolation point.
        
        Parameters
        ----------
        cell_pseudotimes : array-like
            Pseudotime values for each cell
            
        Returns
        -------
        list
            Reciprocal cell density estimates for each interpolation point
        """
        cell_density_estimates = []
        interpolation_points = self.interpolation_points
        range_length_mid = interpolation_points[2] - interpolation_points[0]
        range_length_corner = interpolation_points[1] - interpolation_points[0]
        
        for i in range(len(interpolation_points)):
            if i == 0:
                logic = cell_pseudotimes <= interpolation_points[i+1]
                range_length = range_length_corner
            elif i == len(interpolation_points) - 1:
                logic = cell_pseudotimes >= interpolation_points[i-1]
                range_length = range_length_corner
            else:
                logic = np.logical_and(
                    cell_pseudotimes <= interpolation_points[i+1], 
                    cell_pseudotimes >= interpolation_points[i-1]
                )
                range_length = range_length_mid
            
            density_stat = np.count_nonzero(logic)
            density_stat = density_stat / range_length
            cell_density_estimates.append(density_stat)
        
        # Store the original density estimates
        self.cell_density_estimates_original = cell_density_estimates.copy()
        
        # Take reciprocal for weighting
        cell_density_estimates = [1/x if x > 0 else np.inf for x in cell_density_estimates]
        
        # Handle infinite values
        arr = np.array(cell_density_estimates)
        if np.any(np.isinf(arr)):
            max_w = np.max(arr[np.isfinite(arr)])
            cell_density_estimates = np.where(np.isinf(arr), max_w, arr)
            
        return cell_density_estimates
    
    def compute_adaptive_window_denominator(self, reciprocal_cell_density_estimates):
        """
        Compute adaptive window denominators for Gaussian kernel.
        
        Parameters
        ----------
        reciprocal_cell_density_estimates : array-like
            Reciprocal cell density estimates
            
        Returns
        -------
        list
            Window denominators for each interpolation point
        """
        cell_density_adaptive_weights = np.asarray(reciprocal_cell_density_estimates)
        
        # Scale weights to [0, 1] and multiply by raising_degree
        scaler = MinMaxScaler()
        cell_density_adaptive_weights = scaler.fit_transform(
            cell_density_adaptive_weights.reshape(-1, 1)
        ).flatten()
        cell_density_adaptive_weights = cell_density_adaptive_weights * self.raising_degree
        
        # Calculate adaptive window sizes
        adaptive_window_sizes = [
            cd * self.kernel_window_size for cd in cell_density_adaptive_weights
        ]
        
        # Adjust window sizes to maintain minimum window size
        temp = list(np.abs(np.array(adaptive_window_sizes) - 
                           np.repeat(self.kernel_window_size, self.n_bins)))
        least_affected_point = temp.index(max(temp))
        residue = np.abs(self.kernel_window_size - adaptive_window_sizes[least_affected_point])
        
        if self.raising_degree > 1:
            adaptive_window_sizes = [
                aws + (residue / (self.raising_degree - 1)) 
                for aws in adaptive_window_sizes
            ]
        else:
            adaptive_window_sizes = [aws + residue for aws in adaptive_window_sizes]
        
        # Store for later use
        self.adaptive_window_sizes = adaptive_window_sizes
        
        # Calculate window denominators (squared window sizes)
        window_denominators = [aws**2 for aws in adaptive_window_sizes]
        
        return window_denominators
    
    def compute_weight_matrix(self, abs_timediff_mat, adaptive_win_denoms=None):
        """
        Compute Gaussian kernel weights.
        
        Parameters
        ----------
        abs_timediff_mat : array-like
            Matrix of absolute time differences
        adaptive_win_denoms : array-like, optional
            Adaptive window denominators
            
        Returns
        -------
        numpy.ndarray
            Weight matrix
        """
        if self.adaptive_kernel and adaptive_win_denoms is not None:
            adaptive_win_denoms_mat = np.asarray([
                np.repeat(denom, abs_timediff_mat.shape[1]) 
                for denom in adaptive_win_denoms
            ])
            W_matrix = np.exp(-np.divide(abs_timediff_mat**2, adaptive_win_denoms_mat))
        else:
            W_matrix = np.exp(-np.array(abs_timediff_mat**2) / self.kernel_window_size**2)
            
        return W_matrix
    
    def bin_pseudotime(self, pseudotime):
        """
        Bin pseudotime values into discrete bins.
        
        Parameters
        ----------
        pseudotime : array-like
            Pseudotime values
            
        Returns
        -------
        array-like
            Binned pseudotime values (1 to n_bins)
        """
        #print(f"Pseudotime: {np.min(pseudotime)}, {np.max(pseudotime)}")
        bins = np.linspace(np.min(pseudotime), np.max(pseudotime), self.n_bins + 1)
        binned = np.digitize(pseudotime, bins) - 1
        binned = np.clip(binned, 0, self.n_bins - 1)  # Ensure values are within range
        return binned + 1  # 1-based indexing
    
    def filter_batches_by_coverage(self, batch_labels, binned_pseudotime, 
                                  batch_thred=0.3, ensure_tail=True, 
                                  tail_width=0.3, tail_num=0.02, verbose=True):
        """
        Filter batches based on coverage of pseudotime and presence in tail region.
        
        Parameters
        ----------
        batch_labels : array-like
            Batch labels for each cell
        binned_pseudotime : array-like
            Binned pseudotime values
        batch_thred : float
            Threshold for batch coverage (fraction of bins)
        ensure_tail : bool
            Whether to ensure batches cover the tail region
        tail_width : float
            Width of the tail region (fraction of bins)
        tail_num : float
            Minimum fraction of tail bins that must be covered
        verbose : bool
            Whether to print detailed information
            
        Returns
        -------
        list
            Names of batches that meet criteria
        """
        # Count bin coverage for each batch
        unique_batches = np.unique(batch_labels)
        batch_coverage = {}
        
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_bins = binned_pseudotime[batch_mask]
            unique_bins = np.unique(batch_bins)
            batch_coverage[batch] = len(unique_bins) / self.n_bins
        
        # Filter batches by coverage threshold
        qualified_batches = [
            batch for batch, coverage in batch_coverage.items() 
            if coverage > batch_thred
        ]
        
        if verbose:
            print(f"Batches after coverage filtering: {qualified_batches}")
        
        # If ensure_tail, check for tail coverage using unique bins
        if ensure_tail:
            # Calculate tail threshold - bins greater than this are in the tail region
            tail_threshold = (1 - tail_width) * self.n_bins
            tail_batches = []
            
            # Create metadata-like structure for tail analysis
            # This matches the R implementation's approach
            metadata = pd.DataFrame({
                'batch': batch_labels,
                'pseudotime_binned': binned_pseudotime,
                'pseudotime_binned_tail': binned_pseudotime > tail_threshold
            })
            
            # Create a unique (batch, bin) dataset
            unique_metadata = metadata.drop_duplicates(subset=['batch', 'pseudotime_binned'])
            
            # Create a contingency table of batch vs. tail status
            # Count unique bins per batch that are in the tail region
            contingency_table = pd.crosstab(
                unique_metadata['batch'], 
                unique_metadata['pseudotime_binned_tail']
            )
            
            if verbose:
                print(f"Contingency table of unique bins in tail region:")
                #print(contingency_table)
            
            # Similar to the R code:
            # return(rownames(selectTable)[selectTable[,2]> tail_num* n_bin])
            for batch in qualified_batches:
                if batch in contingency_table.index:
                    # If True column exists (some bins are in tail) and count exceeds threshold
                    if True in contingency_table.columns and contingency_table.loc[batch, True] > tail_num * self.n_bins:
                        tail_batches.append(batch)
                        if verbose:
                            print(f"Batch {batch} qualified: {contingency_table.loc[batch, True] if True in contingency_table.columns else 0} tail bins > {tail_num * self.n_bins} threshold")
                    else:
                        if verbose:
                            print(f"Batch {batch} filtered out: {contingency_table.loc[batch, True] if True in contingency_table.columns else 0} tail bins <= {tail_num * self.n_bins} threshold")
                    
            qualified_batches = tail_batches
            
        if verbose:
            print(f"Final qualified batches: {qualified_batches}")
        return qualified_batches
    
    def calculate_bin_means(self, expression_matrix, cell_indices, bin_labels, n_bins):
        """
        Calculate mean expression for each bin.
        
        Parameters
        ----------
        expression_matrix : array-like
            Gene expression matrix (genes x cells)
        cell_indices : array-like
            Indices to use from expression_matrix
        bin_labels : array-like
            Bin labels for each cell
        n_bins : int
            Number of bins
            
        Returns
        -------
        numpy.ndarray
            Mean expression per bin
        """
        unique_bins = np.arange(1, n_bins+1)
        result = np.zeros((expression_matrix.shape[0], len(unique_bins)))
        
        for i, bin_val in enumerate(unique_bins):
            bin_mask = bin_labels == bin_val
            if np.sum(bin_mask) > 0:
                cell_idx = cell_indices[bin_mask]
                if len(cell_idx) > 0:
                    if scipy.sparse.issparse(expression_matrix):
                        bin_expr = expression_matrix[:, cell_idx].toarray()
                    else:
                        bin_expr = expression_matrix[:, cell_idx]
                    result[:, i] = np.mean(bin_expr, axis=1)
                    
        return result
    
    def interpolate_gene_expression(self, expression_vector, cell_weights):
        """
        Interpolate gene expression using Gaussian kernel weights.
        
        Parameters
        ----------
        expression_vector : array-like
            Expression vector for a gene
        cell_weights : array-like
            Weight matrix from Gaussian kernel
            
        Returns
        -------
        tuple
            (interpolated_means, interpolated_stds)
        """
        interpolated_means = []
        interpolated_stds = []
        
        for bin_idx in range(self.n_bins):
            bin_weights = cell_weights[bin_idx]
            
            # Skip if all weights are zero
            if np.sum(bin_weights) == 0:
                interpolated_means.append(0)
                interpolated_stds.append(0)
                continue
                
            # Calculate weighted mean
            weighted_mean = np.sum(bin_weights * expression_vector) / np.sum(bin_weights)
            
            # Calculate weighted standard deviation
            weighted_var = np.sum(bin_weights * (expression_vector - np.mean(expression_vector))**2)
            weighted_std = np.sqrt(weighted_var / (np.sum(bin_weights) * (len(bin_weights)-1)/len(bin_weights)))
            
            # Weight std by cell density
            if hasattr(self, 'cell_density_estimates_original'):
                weighted_std = weighted_std * self.cell_density_estimates_original[bin_idx]
            
            interpolated_means.append(weighted_mean)
            interpolated_stds.append(weighted_std)
            
        return np.array(interpolated_means), np.array(interpolated_stds)
    
    def anndata_to_3d_matrix(self, adata, pseudo_col, batch_col, 
                           gene_thred=0.1, batch_thred=0.3, 
                           ensure_tail=True, tail_width=0.3, tail_num=0.02, 
                           verbose=True, n_jobs=-1, layer=None):
        """
        Convert AnnData object to 3D matrix using Gaussian kernel interpolation.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object
        pseudo_col : str
            Column in adata.obs containing pseudotime
        batch_col : str
            Column in adata.obs containing batch information
        gene_thred : float
            Threshold for gene filtering (fraction of bins where gene is expressed)
        batch_thred : float
            Threshold for batch filtering (fraction of bins covered)
        ensure_tail : bool
            Whether to ensure batches cover the tail region
        tail_width : float
            Width of the tail region (fraction of bins)
        tail_num : float
            Minimum fraction of tail bins that must be covered
        verbose : bool
            Whether to print progress information
        n_jobs : int, optional
            Number of parallel jobs to run. -1 means using all processors.
        layer : str, optional
            If provided, use this layer in AnnData object instead of .X
            
        Returns
        -------
        dict
            Dictionary containing:
            - reshaped_data: 3D array (batch x time x gene)
            - binned_means: Matrix of binned means
            - filtered_genes: List of genes that passed filtering
            - batch_names: List of batches that passed filtering
            - metadata: DataFrame with metadata
        """
        # Extract data
        if layer is None:
            # Use default expression matrix (X)
            if scipy.sparse.issparse(adata.X):
                expression_matrix = adata.X.T.tocsr()  # genes x cells
            else:
                expression_matrix = adata.X.T  # genes x cells
        else:
            # Use specified layer
            if scipy.sparse.issparse(adata.layers[layer]):
                expression_matrix = adata.layers[layer].T.tocsr()  # genes x cells
            else:
                expression_matrix = adata.layers[layer].T  # genes x cells
            
        pseudotime = np.array(adata.obs[pseudo_col])
        batch_labels = np.array(adata.obs[batch_col])
        
        # Create metadata with binned pseudotime
        binned_pseudotime = self.bin_pseudotime(pseudotime)
        metadata = pd.DataFrame({
            'batch': batch_labels,
            'pseudotime_binned': binned_pseudotime
        })
        metadata['bin'] = metadata['batch'] + "_" + metadata['pseudotime_binned'].astype(str)
        
        # Calculate bin means using traditional binning (for gene filtering)
        unique_bins = metadata['bin'].unique()
        bin_to_idx = {bin_name: idx for idx, bin_name in enumerate(unique_bins)}
        cell_bin_indices = np.array([bin_to_idx[bin_name] for bin_name in metadata['bin']])
        
        # Define the helper function for parallel bin mean calculation
        def _process_bin(bin_idx):
            bin_mask = cell_bin_indices == bin_idx
            if np.sum(bin_mask) > 0:
                if scipy.sparse.issparse(expression_matrix):
                    bin_expr = expression_matrix[:, bin_mask].toarray()
                else:
                    bin_expr = expression_matrix[:, bin_mask]
                return np.mean(bin_expr, axis=1)
            return np.zeros(expression_matrix.shape[0])
        
        # First just create a simple binned mean matrix for filtering - using parallel processing
        if verbose:
            print("Calculating bin means in parallel...")
            
        # Use joblib for parallel bin mean calculation
        n_jobs = n_jobs if n_jobs > 0 else joblib.cpu_count()
        actual_n_jobs = min(n_jobs, len(unique_bins))
        
        binned_means = np.zeros((expression_matrix.shape[0], len(unique_bins)))
        
        # Use progress_bar=True if verbose for joblib feedback
        bin_means_results = Parallel(n_jobs=actual_n_jobs, verbose=1 if verbose else 0)(
            delayed(_process_bin)(bin_idx) for bin_idx in range(len(unique_bins))
        )
        
        # Convert results to array
        for bin_idx, bin_mean in enumerate(bin_means_results):
            binned_means[:, bin_idx] = bin_mean
        
        # Filter genes
        gene_expressed = (binned_means > 0).sum(axis=1)
        gene_threshold = gene_thred * binned_means.shape[1]
        filtered_gene_indices = np.where(gene_expressed > gene_threshold)[0]
        filtered_genes = np.array(adata.var_names)[filtered_gene_indices]
        
        if verbose:
            print(f"Filtered to {len(filtered_genes)} genes that meet expression threshold")
        
        # Filter batches
        batch_names = self.filter_batches_by_coverage(
            batch_labels, binned_pseudotime, batch_thred, 
            ensure_tail, tail_width, tail_num, verbose
        )
        
        # Set up Gaussian kernel interpolation
        # Sort cells by pseudotime for better performance
        sort_idx = np.argsort(pseudotime)
        sorted_pseudotime = pseudotime[sort_idx]
        sorted_batch_labels = batch_labels[sort_idx]
        
        # Normalize pseudotime to [0, 1]
        min_time = np.min(sorted_pseudotime)
        max_time = np.max(sorted_pseudotime)
        normalized_pseudotime = (sorted_pseudotime - min_time) / (max_time - min_time)
        
        if verbose:
            print("Computing Gaussian kernel weights...")
            
        # Compute Gaussian kernel weights
        abs_timediff_mat = self.compute_abs_timediff_mat(normalized_pseudotime)
        
        if self.adaptive_kernel:
            reciprocal_cell_density = self.compute_cell_densities(normalized_pseudotime)
            adaptive_win_denoms = self.compute_adaptive_window_denominator(reciprocal_cell_density)
            weight_matrix = self.compute_weight_matrix(abs_timediff_mat, adaptive_win_denoms)
        else:
            weight_matrix = self.compute_weight_matrix(abs_timediff_mat)
        
        # Create 3D matrix with interpolated values
        filtered_expression = expression_matrix[filtered_gene_indices]
        
        if scipy.sparse.issparse(filtered_expression):
            filtered_expression = filtered_expression.toarray()
            
        # Initialize 3D array
        result_3d = np.zeros((len(batch_names), self.n_bins, len(filtered_genes)))
        
        if verbose:
            print(f"Interpolating gene expression for {len(batch_names)} batches and {len(filtered_genes)} genes using {actual_n_jobs} parallel jobs...")
        
        # Define the helper function for parallel gene interpolation
        def _process_gene_batch(batch_idx, gene_idx):
            batch = batch_names[batch_idx]
            batch_mask = sorted_batch_labels == batch
            
            # Skip if no cells in this batch
            if not np.any(batch_mask):
                return batch_idx, gene_idx, None
            
            batch_weights = weight_matrix[:, batch_mask]
            gene_expr = filtered_expression[gene_idx, sort_idx]
            batch_gene_expr = gene_expr[batch_mask]
            
            # Skip if gene not expressed in this batch
            if np.all(batch_gene_expr == 0):
                return batch_idx, gene_idx, None
            
            # Calculate interpolated values
            interpolated_means, _ = self.interpolate_gene_expression(
                batch_gene_expr, batch_weights
            )
            
            return batch_idx, gene_idx, interpolated_means
        
        # Create list of all (batch_idx, gene_idx) pairs
        tasks = []
        for batch_idx in range(len(batch_names)):
            batch = batch_names[batch_idx]
            batch_mask = sorted_batch_labels == batch
            if np.sum(batch_mask) > 0:  # Only include batch if it has cells
                for gene_idx in range(len(filtered_genes)):
                    tasks.append((batch_idx, gene_idx))
        
        # Process all gene-batch pairs in parallel
        results = Parallel(n_jobs=actual_n_jobs, verbose=1 if verbose else 0)(
            delayed(_process_gene_batch)(batch_idx, gene_idx) 
            for batch_idx, gene_idx in tasks
        )
        
        # Update the 3D array with results
        for batch_idx, gene_idx, interpolated_means in results:
            if interpolated_means is not None:
                result_3d[batch_idx, :, gene_idx] = interpolated_means
        
        if verbose:
            print(f"Interpolation complete. 3D matrix shape: {result_3d.shape}")
            
        # Return results
        return {
            'reshaped_data': result_3d,
            'binned_means': pd.DataFrame(binned_means, index=adata.var_names),
            'filtered_genes': filtered_genes,
            'batch_names': batch_names,
            'metadata': metadata
        }
    
    def plot_interpolation_example(self, adata, gene_name, pseudo_col, batch_col):
        """
        Plot an example of Gaussian kernel interpolation for a gene.
        
        This is now a wrapper around the visualization module function.

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

        Returns
        -------
        matplotlib.figure.Figure
            Figure with plot
        """
        from .visualization import plot_interpolation_example
        return plot_interpolation_example(
            adata=adata,
            gene_name=gene_name, 
            pseudo_col=pseudo_col, 
            batch_col=batch_col,
            interpolator=self
        )

def anndata_to_3d_matrix(adata, pseudo_col, batch_col, n_bins=100, 
                         adaptive_kernel=True, kernel_window_size=0.1, 
                         gene_thred=0.1, batch_thred=0.3, 
                         ensure_tail=True, tail_width=0.3, tail_num=0.02, 
                         verbose=True, n_jobs=-1, layer=None):
    """
    Convert AnnData object to 3D matrix using Gaussian kernel interpolation.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    pseudo_col : str
        Column in adata.obs containing pseudotime
    batch_col : str
        Column in adata.obs containing batch information
    n_bins : int
        Number of interpolation points along pseudotime
    adaptive_kernel : bool
        Whether to use adaptive kernel width based on cell density
    kernel_window_size : float
        Base window size for the Gaussian kernel
    gene_thred : float
        Threshold for gene filtering (fraction of bins where gene is expressed)
    batch_thred : float
        Threshold for batch filtering (fraction of bins covered)
    ensure_tail : bool
        Whether to ensure batches cover the tail region
    tail_width : float
        Width of the tail region (fraction of bins)
    tail_num : float
        Minimum fraction of tail bins that must be covered
    verbose : bool
        Whether to show progress bars and additional information during processing
    n_jobs : int, optional
        Number of parallel jobs to run. -1 means using all processors.
    layer : str, optional
        If provided, use this layer in AnnData object instead of .X
        
    Returns
    -------
    dict
        Dictionary containing:
        - reshaped_data: 3D array (batch x time x gene)
        - binned_means: Matrix of binned means
        - filtered_genes: List of genes that passed filtering
        - batch_names: List of batches that passed filtering
        - metadata: DataFrame with metadata
    """
    interpolator = GaussianTrajectoryInterpolator(
        n_bins=n_bins,
        adaptive_kernel=adaptive_kernel,
        kernel_window_size=kernel_window_size
    )
    
    return interpolator.anndata_to_3d_matrix(
        adata=adata,
        pseudo_col=pseudo_col,
        batch_col=batch_col,
        gene_thred=gene_thred,
        batch_thred=batch_thred,
        ensure_tail=ensure_tail,
        tail_width=tail_width,
        tail_num=tail_num,
        verbose=verbose,
        n_jobs=n_jobs,
        layer=layer
    )

def visualize_fitting_results(standard_results, optimized_results, top_genes_data, 
                       top_gene_names, time_points, output_dir, max_genes_to_plot=10):
    """
    Visualize the results of spline fitting, comparing standard and optimized approaches.
    
    This is now a wrapper around the visualization module function.

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
    from .visualization import visualize_fitting_results as viz_func
    return viz_func(
        standard_results=standard_results,
        optimized_results=optimized_results,
        top_genes_data=top_genes_data,
        top_gene_names=top_gene_names,
        time_points=time_points,
        output_dir=output_dir,
        max_genes_to_plot=max_genes_to_plot
    )

def create_fitting_summary(standard_results, optimized_results, top_gene_names, 
                         top_genes_data, output_file, adata_shape=None, 
                         reshaped_data_shape=None, batch_names=None):
    """
    Create a summary report of the fitting results.
    
    Parameters:
    -----------
    standard_results : dict
        Results from standard fitting
    optimized_results : dict
        Results from optimized fitting
    top_gene_names : list
        Names of fitted genes
    top_genes_data : list
        List of specialized datasets for each gene
    output_file : str or pathlib.Path
        File to save summary to
    adata_shape : tuple, optional
        Shape of original AnnData object
    reshaped_data_shape : tuple, optional
        Shape of the reshaped data
    batch_names : list, optional
        Names of batches
        
    Returns:
    --------
    str
        Path to the summary file
    """
    from pathlib import Path
    from datetime import datetime
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("=== Trajectory Fitting Analysis Summary ===\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset information if provided
        if adata_shape or reshaped_data_shape:
            f.write("Dataset Information:\n")
            if adata_shape:
                f.write(f"- Original AnnData shape: {adata_shape}\n")
            if reshaped_data_shape:
                f.write(f"- 3D Matrix shape: {reshaped_data_shape} (batches, timepoints, genes)\n")
                f.write(f"- Number of batches: {reshaped_data_shape[0]}\n")
                f.write(f"- Number of genes: {reshaped_data_shape[2]}\n")
            if batch_names:
                f.write(f"- Batch names: {', '.join(batch_names)}\n")
            f.write("\n")
        
        # Sample selection approach
        f.write("Sample Selection Approach:\n")
        f.write(f"- For each gene, only the most conserved half of the samples were used for fitting\n")
        f.write(f"- Samples were ranked by their mean pairwise distance to other samples\n")
        f.write(f"- This approach focuses the model on the most reliable, least variable samples\n\n")
        
        # Overall fitting results
        f.write("Spline Fitting Results:\n")
        improvement = ((-standard_results['model_score']) - (-optimized_results['model_score']))
        percent_improvement = 100 * improvement / (-standard_results['model_score'])
        f.write(f"- Standard spline approach - mean DTW distance: {-standard_results['model_score']:.4f}\n")
        f.write(f"- DTW-optimized approach - mean DTW distance: {-optimized_results['model_score']:.4f}\n")
        f.write(f"- Improvement: {improvement:.4f} ({percent_improvement:.2f}%)\n\n")
        
        # Individual gene results
        f.write(f"Results for Individual Genes:\n")
        for i, gene_name in enumerate(top_gene_names):
            smoothing = optimized_results['smoothing_values'][i]
            std_dtw = standard_results['dtw_distances'][i]
            opt_dtw = optimized_results['dtw_distances'][i]
            percent_imp = 100 * (std_dtw - opt_dtw) / std_dtw
            n_samples_used = top_genes_data[i].shape[0] if i < len(top_genes_data) else "N/A"
            total_samples = "N/A"
            if reshaped_data_shape:
                total_samples = reshaped_data_shape[0]
                
            f.write(f"  {i+1}. {gene_name}:\n")
            f.write(f"     - Optimized smoothing: {smoothing:.4f}\n")
            f.write(f"     - Standard DTW distance: {std_dtw:.4f}\n")
            f.write(f"     - Optimized DTW distance: {opt_dtw:.4f}\n")
            f.write(f"     - Improvement: {percent_imp:.2f}%\n")
            f.write(f"     - Samples used: {n_samples_used}/{total_samples}\n\n")
        
        f.write("\n=== Analysis Complete ===\n")
    
    return str(output_file)

def _process_single_gene(i, gene_name, gene_data, fitter, model_type, 
                         spline_degree, spline_smoothing,
                        use_dtw_optimization, n_jobs_per_gene=1):
    """
    Helper function to process a single gene with both standard and optional DTW-optimized fitting.
    
    Parameters:
    -----------
    i : int
        Index of the gene
    gene_name : str
        Name of the gene
    gene_data : numpy.ndarray
        Data for this gene with shape (n_samples, n_timepoints, 1)
    fitter : TrajectoryFitter
        Initialized TrajectoryFitter object
    model_type : str
        Type of model to fit
    spline_degree : int
        Degree of spline to fit
    spline_smoothing : float
        Smoothing factor for spline fitting
    use_dtw_optimization : bool
        Whether to perform DTW optimization
    n_jobs_per_gene : int
        Number of jobs to use for fitting this gene
        
    Returns:
    --------
    dict
        Results for this gene including standard and optimized fits
    """
    # Set fitter's n_jobs for this gene
    fitter.n_jobs = n_jobs_per_gene
    
    # Fit standard spline model
    gene_standard_results = fitter.fit(
        gene_data,
        model_type=model_type,
        spline_degree=spline_degree,
        spline_smoothing=spline_smoothing,
        optimize_spline_dtw=False,
        #verbose=False  # Disable verbose in worker
    )
    
    # Fit DTW-optimized model if requested
    if use_dtw_optimization:
        gene_optimized_results = fitter.fit(
            gene_data,
            model_type=model_type,
            spline_degree=spline_degree,
            spline_smoothing=spline_smoothing,  # Initial value, will be optimized
            optimize_spline_dtw=True,
            #verbose=False  # Disable verbose in worker
        )
    else:
        # Create placeholder results with same structure as standard results
        gene_optimized_results = {
            'fitted_params': gene_standard_results['fitted_params'],
            'fitted_trajectories': gene_standard_results['fitted_trajectories'],
            'dtw_distances': gene_standard_results['dtw_distances'],
            'smoothing_values': gene_standard_results['smoothing_values']
        }
    
    # Return all results for this gene
    return {
        'gene_idx': i,
        'gene_name': gene_name,
        'standard_results': {
            'fitted_params': gene_standard_results['fitted_params'][0],
            'fitted_trajectory': gene_standard_results['fitted_trajectories'][:, 0],
            'dtw_distance': gene_standard_results['dtw_distances'][0],
            'smoothing_value': gene_standard_results['smoothing_values'][0]
        },
        'optimized_results': {
            'fitted_params': gene_optimized_results['fitted_params'][0],
            'fitted_trajectory': gene_optimized_results['fitted_trajectories'][:, 0],
            'dtw_distance': gene_optimized_results['dtw_distances'][0],
            'smoothing_value': gene_optimized_results['smoothing_values'][0]
        }
    }

def fit_with_conserved_samples(reshaped_data, gene_names, conserved_samples, time_points=None, 
                              top_n_genes=None, n_jobs=-1, verbose=False,
                              interpolation_factor=1, gene_positions=None,
                              model_type='spline', spline_degree=3, spline_smoothing=2,
                              use_dtw_optimization=True):
    """
    Fit trajectory models using only the most conserved samples for each gene.
    
    This function processes each gene individually, using only its most conserved samples for fitting.
    This approach leads to better fits as it focuses on the most reliable data for each gene.
    
    Parameters:
    -----------
    reshaped_data : numpy.ndarray
        3D array with shape (sample, time, gene)
    gene_names : list or numpy.ndarray
        List of gene names corresponding to the gene axis
    conserved_samples : dict
        Dictionary mapping gene names to indices of most conserved samples
        (output from get_most_conserved_samples)
    time_points : numpy.ndarray, optional (default=None)
        Time points for fitting. If None, evenly spaced points from 0 to 1 will be used.
    top_n_genes : int or None, optional (default=None)
        Number of top genes to fit. If None, all genes will be fitted.
    n_jobs : int, optional (default=1)
        Number of parallel jobs for processing genes. If -1, all available cores are used.
    verbose : bool, optional (default=False)
        Whether to show progress bar and additional information
    interpolation_factor : int, optional (default=1)
        Interpolation factor for TrajectoryFitter
    gene_positions : dict or list, optional (default=None)
        Mapping of gene names to their position indices in reshaped_data's third dimension.
        If a dict, keys are gene names and values are their positions.
        If a list, should be the same length as gene_names, containing corresponding position indices.
        If None, positions will be determined by searching gene_names in all_gene_names.
    model_type : str, optional (default='spline')
        Type of model to fit
    spline_degree : int, optional (default=3)
        Degree of spline to fit
    spline_smoothing : float, optional (default=2)
        Smoothing factor for spline fitting
    use_dtw_optimization : bool, optional (default=True)
        Whether to perform DTW optimization. If False, only standard fitting is performed.
        
    Returns:
    --------
    dict
        Dictionary containing:
        - standard_results: Results from standard fitting
        - optimized_results: Results from DTW-optimized fitting (same as standard if use_dtw_optimization=False)
        - top_gene_names: Names of fitted genes
        - top_genes_data: List of specialized datasets for each gene
    """
    # If time_points is not provided, create evenly spaced points from 0 to 1
    if time_points is None:
        time_points = np.linspace(0, 1, reshaped_data.shape[1])
        
    try:
        from .trajectory_fitter import TrajectoryFitter
    except ImportError:
        try:
            from .trajectory_fitter import TrajectoryFitter
        except ImportError:
            raise ImportError("Cannot import TrajectoryFitter. Make sure trajectory_fitter.py is in the same directory or accessible in the Python path.")
    
    # Initialize total sample count
    n_samples = reshaped_data.shape[0]
    
    # Select top genes if gene_names is a pandas DataFrame with 'normalized_score' column
    # (e.g., from conservation_scores output)
    if hasattr(gene_names, 'head') and 'normalized_score' in gene_names.columns:
        # If top_n_genes is None, use all genes (that aren't filtered)
        if top_n_genes is None:
            if 'was_filtered' in gene_names.columns:
                top_gene_df = gene_names[~gene_names['was_filtered']]
            else:
                top_gene_df = gene_names
        else:
            # Otherwise, use the top N genes
            if 'was_filtered' in gene_names.columns:
                top_gene_df = gene_names[~gene_names['was_filtered']].head(top_n_genes)
            else:
                top_gene_df = gene_names.head(top_n_genes)
        top_gene_names = top_gene_df['gene'].tolist()
    else:
        # If gene_names is a list or array, use all genes if top_n_genes is None
        if top_n_genes is None:
            top_gene_names = gene_names
        else:
            # Otherwise, take the first top_n_genes
            top_gene_names = gene_names[:top_n_genes]
    
    # Determine gene positions based on input parameters
    if gene_positions is not None:
        # If gene_positions is a dict, get positions directly from it
        if isinstance(gene_positions, dict):
            top_gene_positions = []
            for gene in top_gene_names:
                if gene in gene_positions:
                    pos = gene_positions[gene]
                    # Validate position is within range
                    if pos < 0 or pos >= reshaped_data.shape[2]:
                        raise ValueError(f"Position {pos} for gene {gene} is out of range (0-{reshaped_data.shape[2]-1})")
                    top_gene_positions.append(pos)
                else:
                    raise ValueError(f"Gene {gene} not found in gene_positions dictionary")
        # If gene_positions is a list or array, use it directly
        elif isinstance(gene_positions, (list, np.ndarray)):
            # Verify it has enough elements
            if len(gene_positions) < len(top_gene_names):
                raise ValueError(f"gene_positions has {len(gene_positions)} elements but {len(top_gene_names)} genes were selected")
            top_gene_positions = gene_positions[:len(top_gene_names)]
            # Validate positions are within range
            for i, pos in enumerate(top_gene_positions):
                if pos < 0 or pos >= reshaped_data.shape[2]:
                    raise ValueError(f"Position {pos} for gene {top_gene_names[i]} is out of range (0-{reshaped_data.shape[2]-1})")
        else:
            raise ValueError("gene_positions must be a dict or list-like")
    else:
        # Use traditional lookup method if gene_positions not provided
        if verbose:
            print("No gene_positions provided. Finding positions by searching in gene_names...")
        all_gene_names = gene_names if isinstance(gene_names, (list, np.ndarray)) else gene_names['gene'].values
        top_gene_positions = []
        for gene in top_gene_names:
            matches = np.where(np.array(all_gene_names) == gene)[0]
            if len(matches) > 0:
                top_gene_positions.append(matches[0])
            else:
                raise ValueError(f"Gene {gene} not found in all_gene_names")
    
    # Create a specialized dataset for each gene, using only its most conserved samples
    top_genes_data = []
    
    if verbose:
        print("Creating specialized datasets for each gene...")
    
    for i, gene_name in enumerate(top_gene_names):
        gene_pos = top_gene_positions[i]
        
        # Get the most conserved samples for this gene
        if gene_name in conserved_samples:
            cons_sample_indices = conserved_samples[gene_name]
            n_cons_samples = len(cons_sample_indices)
            
            # Extract data only for the most conserved samples for this gene
            gene_data = reshaped_data[cons_sample_indices, :, gene_pos]
            
            # Reshape to match expected input format (samples, timepoints, 1 feature)
            gene_data = gene_data.reshape(n_cons_samples, reshaped_data.shape[1], 1)
            
            if verbose:
                if i == 0 or (i+1) % max(1, len(top_gene_names)//5) == 0:  # Print only for a subset of genes
                    print(f"  Gene {gene_name}: Using {n_cons_samples} most conserved samples out of {n_samples} total (position: {gene_pos})")
        else:
            # Fallback if gene not in conserved_samples
            if verbose:
                print(f"  Gene {gene_name}: Using all samples (gene not found in conserved samples dict) (position: {gene_pos})")
            gene_data = reshaped_data[:, :, gene_pos:gene_pos+1]
        
        top_genes_data.append(gene_data)
    
    # Initialize TrajectoryFitter
    if verbose:
        print("Initializing TrajectoryFitter...")
    
    # Initialize with n_jobs=1 since we'll parallelize at the gene level
    fitter = TrajectoryFitter(
        time_points=time_points,
        n_jobs=1,  # Will be set in the worker for each gene
        verbose=False,  # Disable verbose in the fitter
        interpolation_factor=interpolation_factor
    )
    
    # Initialize result structures
    standard_results = {
        'fitted_params': [],
        'fitted_trajectories': [],
        'dtw_distances': [],
        'smoothing_values': []
    }
    
    optimized_results = {
        'fitted_params': [],
        'fitted_trajectories': [],
        'dtw_distances': [],
        'smoothing_values': []
    }
    
    # Determine number of jobs to use
    if n_jobs <= 0:
        n_jobs = joblib.cpu_count()
    actual_n_jobs = min(n_jobs, len(top_gene_names))
    
    # Calculate jobs per gene if processing one gene at a time
    n_jobs_per_gene = max(1, n_jobs // len(top_gene_names)) if len(top_gene_names) < n_jobs else 1
    
    if verbose:
        print(f"\nProcessing {len(top_gene_names)} genes with {actual_n_jobs} parallel jobs...")
        if not use_dtw_optimization:
            print("DTW optimization is disabled - only standard fitting will be performed")
    
    # Process genes in parallel with progress bar
    with tqdm(total=len(top_gene_names), disable=not verbose, desc="Processing genes") as pbar:
        # Run parallel processing
        results = Parallel(n_jobs=actual_n_jobs, verbose=0)(
            delayed(_process_single_gene)(
                i, gene_name, top_genes_data[i], fitter, 
                model_type, spline_degree, spline_smoothing, 
                use_dtw_optimization, n_jobs_per_gene
            ) for i, gene_name in enumerate(top_gene_names)
        )
        
    # Process results
    for result in results:
        # Extract standard results
        standard_results['fitted_params'].append(result['standard_results']['fitted_params'])
        standard_results['fitted_trajectories'].append(result['standard_results']['fitted_trajectory'])
        standard_results['dtw_distances'].append(result['standard_results']['dtw_distance'])
        standard_results['smoothing_values'].append(result['standard_results']['smoothing_value'])
        
        # Extract optimized results
        optimized_results['fitted_params'].append(result['optimized_results']['fitted_params'])
        optimized_results['fitted_trajectories'].append(result['optimized_results']['fitted_trajectory'])
        optimized_results['dtw_distances'].append(result['optimized_results']['dtw_distance'])
        optimized_results['smoothing_values'].append(result['optimized_results']['smoothing_value'])
    
    # Convert lists to arrays for consistency
    standard_results['fitted_trajectories'] = np.array(standard_results['fitted_trajectories']).T
    optimized_results['fitted_trajectories'] = np.array(optimized_results['fitted_trajectories']).T
    standard_results['dtw_distances'] = np.array(standard_results['dtw_distances'])
    optimized_results['dtw_distances'] = np.array(optimized_results['dtw_distances'])
    standard_results['smoothing_values'] = np.array(standard_results['smoothing_values'])
    optimized_results['smoothing_values'] = np.array(optimized_results['smoothing_values'])
    
    # Add time points to results
    standard_results['time_points'] = fitter.fine_time_points
    optimized_results['time_points'] = fitter.fine_time_points
    
    # Calculate overall scores
    standard_results['model_score'] = -np.mean(standard_results['dtw_distances'])
    optimized_results['model_score'] = -np.mean(optimized_results['dtw_distances'])
    
    # Add mean_dtw_distance for compatibility with example_pipeline.py
    standard_results['mean_dtw_distance'] = np.mean(standard_results['dtw_distances'])
    optimized_results['mean_dtw_distance'] = np.mean(optimized_results['dtw_distances'])
    
    # Print overall comparison if verbose and DTW optimization was performed
    if verbose and use_dtw_optimization:
        improvement = ((-standard_results['model_score']) - (-optimized_results['model_score']))
        percent_improvement = 100 * improvement / (-standard_results['model_score']) if -standard_results['model_score'] != 0 else 0
        
        print("\nSpline Fitting Results Comparison:")
        print(f"Standard approach - mean DTW distance: {-standard_results['model_score']:.4f}")
        print(f"DTW-optimized approach - mean DTW distance: {-optimized_results['model_score']:.4f}")
        print(f"Improvement: {improvement:.4f}")
        print(f"Percentage improvement: {percent_improvement:.2f}%")
    
    # Return comprehensive results
    return {
        'standard_results': standard_results,
        'optimized_results': optimized_results,
        'top_gene_names': top_gene_names,
        'top_genes_data': top_genes_data
    }

def run_trajectory_conservation_analysis(
    adata_path,
    output_dir,
    pseudo_col='pseudo',
    batch_col='Sample',
    n_bins=100,
    adaptive_kernel=True,
    gene_thred=0.1,
    batch_thred=0.4,
    tail_num=0.05,
    ensure_tail=True,
    dtw_radius=3,
    use_fastdtw=True,
    normalize='zscore',
    variation_filter_level='basic',
    top_n_genes=4000,
    spline_smoothing=2,
    interpolation_factor=1,
    n_jobs=-1,
    save_figures=True,
    gene_subset=None,
    layer="logcounts"
):
    """
    Run trajectory conservation analysis on scRNA-seq data.
    
    Parameters:
    -----------
    adata_path : str
        Path to the AnnData h5ad file
    output_dir : str or Path
        Directory to save output files
    pseudo_col : str, default='pseudo'
        Column in adata.obs containing pseudotime values
    batch_col : str, default='Sample'
        Column in adata.obs containing batch/sample information
    n_bins : int, default=100
        Number of interpolation points along pseudotime
    adaptive_kernel : bool, default=True
        Whether to use adaptive kernel width for interpolation
    gene_thred : float, default=0.1
        Filter genes expressed in at least this fraction of bins
    batch_thred : float, default=0.3
        Filter batches covering at least this fraction of timeline
    ensure_tail : bool, default=True
        Ensure batches cover the tail region
    dtw_radius : int, default=3
        Radius parameter for fastdtw algorithm
    use_fastdtw : bool, default=True
        Whether to use fastdtw algorithm
    normalize : str, default='zscore'
        Method to normalize trajectories before DTW calculation
    variation_filter_level : str, default='basic'
        Level of filtering for sample variation ('off', 'basic', 'stringent')
    top_n_genes : int, default=4000
        Number of top conserved genes to select for fitting
    spline_smoothing : float, default=2
        Smoothing parameter for spline fitting
    interpolation_factor : int, default=1
        Factor for interpolation when fitting
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all available cores)
    save_figures : bool, default=True
        Whether to save visualization figures
    gene_subset : list, default=None
        Optional list of genes to use (if None, all genes are used)
    layer : str, default="logcounts"
        Layer in anndata to use for expression values
        
    Returns:
    --------
    dict
        Dictionary containing:
        - conservation_results: Results from conservation analysis
        - fit_results: Results from trajectory fitting
        - filtered_genes: List of filtered genes
        - selected_genes: List of selected genes for fitting
        - reshaped_data: 3D matrix of expression data
    """
    import scanpy as sc
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import os
    from datetime import datetime
    
    # Import from the trajDTW package
    from trajDTW import (
        anndata_to_3d_matrix, 
        calculate_trajectory_conservation,
        TrajectoryFitter,
        get_most_conserved_samples,
        fit_with_conserved_samples,
        extract_pairwise_distances,
        create_gene_position_mapping
    )
    
    # Convert output_dir to Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define variation filtering parameters
    VARIATION_FILTERING = {
        'off': {
            'filter_samples_by_variation': False
        },
        'basic': {
            'filter_samples_by_variation': True,
            'variation_threshold': 0.1,  # Minimum coefficient of variation
            'variation_metric': 'max',
            'min_valid_samples': 2       # At least 2 samples needed
        },
        'stringent': {
            'filter_samples_by_variation': True,
            'variation_threshold': 0.2, 
            'variation_metric': 'max',
            'min_valid_samples': 2
        }
    }
    
    # Setup logging
    log_file = output_dir / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log(message):
        """Log message to both console and file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
    
    # Start the analysis
    log(f"\n=== Trajectory Conservation Analysis Pipeline ===")
    log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Input data: {adata_path}")
    log(f"Output directory: {output_dir}")
    
    # ================ 1. BUILD 3D MATRIX ================
    log("\n1. Building 3D Matrix from AnnData")
    log("-" * 50)
    
    # Load AnnData
    log("Loading AnnData...")
    adata = sc.read_h5ad(adata_path)
    
    # Apply gene subset if provided
    if gene_subset is not None:
        adata = adata[:, gene_subset].copy()
        log(f"Using subset of {len(gene_subset)} genes")
    
    log(f"AnnData shape: {adata.shape}")
    
    # Convert to 3D matrix
    log("\nConverting to 3D matrix using Gaussian kernel interpolation...")
    result = anndata_to_3d_matrix(
        adata=adata,
        pseudo_col=pseudo_col,
        batch_col=batch_col,
        n_bins=n_bins,
        adaptive_kernel=adaptive_kernel,
        gene_thred=gene_thred,
        batch_thred=batch_thred,
        tail_num= tail_num,
        ensure_tail=ensure_tail,
        layer=layer
    )
    reshaped_data = result["reshaped_data"]
    filtered_genes = result['filtered_genes']
    batch_names = result['batch_names']
    
    # Save reshaped_data as npy file
    log("Saving reshaped data as npy file...")
    np.save(output_dir / "reshaped_data.npy", reshaped_data)
    pd.DataFrame(filtered_genes).to_csv(output_dir / "filtered_genes.csv")
    pd.DataFrame(batch_names).to_csv(output_dir / "batch_names.csv")
    log(f"Reshaped data dimensions: {reshaped_data.shape}")
    log(f"Number of filtered genes: {len(filtered_genes)}")
    log(f"Batches included: {batch_names}")
    
    # ================ 2. CALCULATE CONSERVATION ================
    log("\n2. Calculating Trajectory Conservation")
    log("-" * 50)
    
    filter_params = VARIATION_FILTERING[variation_filter_level]
    log(f"Using variation filter level: {variation_filter_level}")
    
    log("Calculating trajectory conservation...")
    conservation_results = calculate_trajectory_conservation(
        trajectory_data=reshaped_data,
        gene_names=filtered_genes, 
        save_dir=output_dir,
        prefix="traj_conservation",
        dtw_radius=dtw_radius,
        use_fastdtw=use_fastdtw,
        normalize=normalize,
        **filter_params
    )
    
    # Extract and save pairwise distances
    log("Extracting pairwise distances...")
    pairwise_distances_df = extract_pairwise_distances(
        conservation_results, 
        output_csv=output_dir / "pairwise_distances.csv"
    )
    
    # Save conservation scores
    conservation_scores_df = conservation_results["conservation_scores"]
    conservation_scores_df.to_csv(output_dir / "conservation_scores.csv")
    log(f"Conservation scores saved to: {output_dir / 'conservation_scores.csv'}")
    
    # Select top conserved genes
    log(f"Selecting top {top_n_genes} conserved genes for fitting...")
    selected_genes = np.array(conservation_scores_df["gene"].head(n=top_n_genes))
    
    # Create gene position mapping
    log("Creating gene position mapping...")
    gene_mapping = create_gene_position_mapping(selected_genes, filtered_genes)
    
    # ================ 3. FIT TRAJECTORIES ================
    log("\n3. Fitting Gene Expression Trajectories")
    log("-" * 50)
    
    log("Fitting trajectories with conserved samples...")
    fit_res = fit_with_conserved_samples(
        reshaped_data=reshaped_data, 
        gene_names=selected_genes,
        gene_positions=gene_mapping,
        conserved_samples=conservation_results["conserved_samples"], 
        interpolation_factor=interpolation_factor,
        top_n_genes=None,  # Use all selected genes
        verbose=True, 
        spline_smoothing=spline_smoothing,
        n_jobs=n_jobs
    )
    
    # Save fitted trajectories
    log("Saving fitted trajectories...")
    fitdf = pd.DataFrame(fit_res["standard_results"]["fitted_trajectories"])
    fitdf.columns = fit_res["top_gene_names"]
    fitdf.to_csv(output_dir / "fitted_trajectories.csv")
    
    fitdfOptimized = pd.DataFrame(fit_res["optimized_results"]["fitted_trajectories"])
    fitdfOptimized.columns = fit_res["top_gene_names"]
    fitdfOptimized.to_csv(output_dir / "fitted_trajectories_optimized.csv")
    
    # ================ 4. VISUALIZATIONS ================
    if save_figures:
        log("\n4. Creating Visualizations")
        log("-" * 50)
        
        # 1. Heatmap of fitted trajectories
        log("Creating heatmap of fitted trajectories...")
        plt.figure(figsize=(16, 12))
        sns.heatmap(fitdf, cmap='viridis', cbar=True)
        plt.title("Fitted Gene Expression Trajectories")
        plt.xlabel("Genes")
        plt.ylabel("Pseudotime Points")
        plt.savefig(output_dir / "fitted_trajectories_heatmap.png", bbox_inches='tight', dpi=300)
        
        # 2. K-means clustering of genes
        log("Performing K-means clustering of genes...")
        from sklearn.cluster import KMeans
        
        # Choose number of clusters (this can be parameterized)
        n_clusters = 8
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(fitdf.T)  # Transpose to cluster genes
        
        # Add cluster labels to gene data
        gene_clusters = pd.DataFrame({
            'gene': fit_res["top_gene_names"],
            'cluster': cluster_labels
        })
        gene_clusters.to_csv(output_dir / "gene_clusters.csv", index=False)
        
        # 3. Clustered heatmap
        log("Creating clustered heatmap...")
        # Sort genes by cluster
        order = gene_clusters.sort_values('cluster').index
        fitdf_clustered = fitdf[fit_res["top_gene_names"][order]]
        
        plt.figure(figsize=(16, 12))
        sns.heatmap(fitdf_clustered, cmap='viridis', cbar=True)
        plt.title("K-means Clustered Gene Expression Trajectories")
        plt.xlabel("Genes (Clustered)")
        plt.ylabel("Pseudotime Points")
        plt.savefig(output_dir / "clustered_fitted_trajectories_heatmap.png", bbox_inches='tight', dpi=300)
        
        # 4. Cluster profile plot
        log("Creating cluster profile plot...")
        plt.figure(figsize=(14, 10))
        
        for cluster in range(n_clusters):
            cluster_genes = gene_clusters[gene_clusters['cluster'] == cluster]['gene']
            if len(cluster_genes) > 0:
                cluster_data = fitdf[cluster_genes].mean(axis=1)
                plt.plot(cluster_data, label=f'Cluster {cluster} (n={len(cluster_genes)})')
        
        plt.title("Average Expression Profiles by Cluster")
        plt.xlabel("Pseudotime Points")
        plt.ylabel("Average Expression")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "cluster_profiles.png", bbox_inches='tight', dpi=300)
    
    log("\n=== Analysis Complete ===")
    log(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return results
    return {
        'conservation_results': conservation_results,
        'fit_results': fit_res,
        'filtered_genes': filtered_genes,
        'selected_genes': selected_genes,
        'reshaped_data': reshaped_data
    }

def create_gene_position_mapping(gene_names, filtered_genes=None, original_genes=None):
    """
    Create a mapping from gene names to their positions in a reshaped data array.
    
    This utility function helps create the gene_positions dictionary needed for 
    fit_with_conserved_samples when gene names don't directly map to positions
    in the reshaped_data array.
    
    Parameters:
    -----------
    gene_names : list or numpy.ndarray
        List of gene names to include in the mapping
    filtered_genes : list or numpy.ndarray, optional
        List of gene names that correspond to the third dimension of reshaped_data.
        If provided, positions will be determined by finding each gene_name in filtered_genes.
    original_genes : list or numpy.ndarray, optional
        List of all original gene names before any filtering.
        If provided (and filtered_genes also provided), will help trace gene positions
        through the filtering process.
        
    Returns:
    --------
    dict
        Dictionary mapping gene names to their positions
    
    Examples:
    ---------
    >>> # Simple case - create mapping for all genes
    >>> gene_names = ['gene1', 'gene2', 'gene3']
    >>> mapping = create_gene_position_mapping(gene_names)
    >>> print(mapping)
    {'gene1': 0, 'gene2': 1, 'gene3': 2}
    
    >>> # With filtering - map from original gene list to filtered positions
    >>> original_genes = ['gene1', 'gene2', 'gene3', 'gene4', 'gene5']
    >>> filtered_genes = ['gene1', 'gene3', 'gene5']  # gene2 and gene4 were filtered out
    >>> mapping = create_gene_position_mapping(['gene1', 'gene5'], filtered_genes, original_genes)
    >>> print(mapping)
    {'gene1': 0, 'gene5': 2}
    """
    # Convert inputs to numpy arrays for easier searching
    gene_names = np.array(gene_names)
    
    # Case 1: No filtering information provided
    if filtered_genes is None:
        # Simply map each gene to its position in the gene_names array
        return {gene: i for i, gene in enumerate(gene_names)}
    
    # Case 2: Filtered genes provided
    filtered_genes = np.array(filtered_genes)
    mapping = {}
    
    for gene in gene_names:
        # Find the gene in filtered_genes
        matches = np.where(filtered_genes == gene)[0]
        if len(matches) > 0:
            # Gene found in filtered_genes, use its position
            mapping[gene] = matches[0]
        else:
            # Gene not found in filtered_genes
            if original_genes is not None:
                # Try to find it in original_genes to give a helpful error
                orig_matches = np.where(np.array(original_genes) == gene)[0]
                if len(orig_matches) > 0:
                    raise ValueError(f"Gene '{gene}' was found in original_genes but not in filtered_genes, suggesting it was filtered out")
            
            # Generic error if we can't find it anywhere
            raise ValueError(f"Gene '{gene}' not found in filtered_genes")
    
    return mapping



# Example usage:
"""
# Create AnnData object
import scanpy as sc
import anndata

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Add pseudotime and batch information if not already present
# adata.obs['pseudotime'] = ...
# adata.obs['batch'] = ...

# Convert to 3D matrix using the default X matrix
result = anndata_to_3d_matrix(
    adata=adata,
    pseudo_col='pseudotime',
    batch_col='batch',
    n_bins=100,
    adaptive_kernel=True
)

# Or convert using a specific layer (e.g., 'normalized' or 'counts')
result_from_layer = anndata_to_3d_matrix(
    adata=adata,
    pseudo_col='pseudotime',
    batch_col='batch',
    n_bins=100,
    adaptive_kernel=True,
    layer='counts'  # Specify the layer to use
)

# Access results
reshaped_data = result['reshaped_data']  # 3D array (batch x time x gene)
filtered_genes = result['filtered_genes']
batch_names = result['batch_names']

# Plot example for a specific gene
interpolator = GaussianTrajectoryInterpolator(n_bins=100, adaptive_kernel=True)
fig = interpolator.plot_interpolation_example(
    adata=adata,
    gene_name='YourGene',
    pseudo_col='pseudotime',
    batch_col='batch'
)
fig.savefig('interpolation_example.png')

# Calculate conservation scores with normalization
conservation_results = calculate_trajectory_conservation(
    trajectory_data=reshaped_data,
    gene_names=filtered_genes,
    save_dir='./conservation_results',
    prefix='my_dataset',
    normalize='zscore'  # Normalize trajectories to address variation differences
)

# Access results
conservation_scores = conservation_results['conservation_scores']
most_conserved_genes = conservation_scores.head(10)
print("Most conserved genes:")
print(most_conserved_genes)

# Access pairwise distances for a specific gene
gene_of_interest = filtered_genes[0]
pairwise_dtw = conservation_results['pairwise_distances'][gene_of_interest]

# Create gene position mapping for a subset of genes
selected_genes = filtered_genes[:5]  # Select first 5 genes
gene_mapping = create_gene_position_mapping(selected_genes, filtered_genes)

# Fit models using only these specific genes with their positions
fit_results = fit_with_conserved_samples(
    reshaped_data=reshaped_data,
    gene_names=selected_genes,
    conserved_samples=conservation_results["conserved_samples"],
    time_points=time_points,
    gene_positions=gene_mapping
)

# Run the full pipeline with a specific layer
pipeline_results = run_trajectory_conservation_analysis(
    adata_path="your_data.h5ad",
    output_dir="./analysis_results",
    pseudo_col="pseudotime",
    batch_col="batch",
    layer="logcounts",
    top_n_genes=500
)
""" 