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
                           verbose=True, n_jobs=-1):
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
        if scipy.sparse.issparse(adata.X):
            expression_matrix = adata.X.T.tocsr()  # genes x cells
        else:
            expression_matrix = adata.X.T  # genes x cells
            
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
                         verbose=True, n_jobs=-1):
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
        n_jobs=n_jobs
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
                              interpolation_factor=1,
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
    
    # Find indices in the data array for the selected genes
    all_gene_names = gene_names if isinstance(gene_names, (list, np.ndarray)) else gene_names['gene'].values
    top_gene_positions = [np.where(np.array(all_gene_names) == gene)[0][0] for gene in top_gene_names]
    
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
                    print(f"  Gene {gene_name}: Using {n_cons_samples} most conserved samples out of {n_samples} total")
        else:
            # Fallback if gene not in conserved_samples
            if verbose:
                print(f"  Gene {gene_name}: Using all samples (gene not found in conserved samples dict)")
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

def run_conserved_sample_fitting_pipeline(adata, batch_key, time_key, n_jobs=4, 
                                        output_dir=None, top_n_genes=20, 
                                        conserved_fraction=0.5, interpolation_factor=1,
                                        spline_degree=3, spline_smoothing=0.5, 
                                        model_type='spline', verbose=True,
                                        max_genes_to_plot=10):
    """
    Run the entire conserved sample fitting pipeline in one function call.
    
    This pipeline:
    1. Processes the AnnData object into a 3D array
    2. Calculates pairwise distances for all genes
    3. Identifies the most conserved samples for each gene
    4. Fits both standard and DTW-optimized models using only conserved samples
    5. Creates visualizations of the fitting results
    6. Generates a comprehensive summary report
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing gene expression data
    batch_key : str
        Key in adata.obs for batch/sample information
    time_key : str
        Key in adata.obs for pseudotime information
    n_jobs : int, optional (default=4)
        Number of parallel jobs for computations
    output_dir : str or pathlib.Path, optional
        Directory to save outputs (if None, a directory named 'conserved_fitting_results' 
        will be created in the current working directory)
    top_n_genes : int, optional (default=20)
        Number of top most conserved genes to analyze
    conserved_fraction : float, optional (default=0.5)
        Fraction of most conserved samples to use for each gene (0.0-1.0)
    interpolation_factor : int, optional (default=2)
        Factor for interpolating time points
    spline_degree : int, optional (default=3)
        Degree of the spline for fitting
    spline_smoothing : float, optional (default=0.5)
        Smoothing parameter for standard spline fitting
    model_type : str, optional (default='spline')
        Type of model to fit ('spline' or other supported types)
    verbose : bool, optional (default=True)
        Whether to print progress information
    max_genes_to_plot : int, optional (default=10)
        Maximum number of genes to create visualizations for
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'standard_results': Results from standard fitting
        - 'optimized_results': Results from DTW-optimized fitting
        - 'top_gene_names': Names of fitted genes
        - 'visualizations': Paths to visualization files
        - 'summary_file': Path to summary report
        - 'pairwise_distances': Calculated pairwise distances
        - 'conserved_samples': Dictionary of conserved samples for each gene
        - 'top_genes_data': List of gene-specific datasets
    """
    import numpy as np
    import scanpy as sc
    import pandas as pd
    from pathlib import Path
    import time
    import matplotlib.pyplot as plt
    import sys
    import os
    
    # Add current directory to path to ensure imports work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Try different import strategies for trajectory_fitter
    try:
        from .trajectory_fitter import TrajectoryFitter
    except ImportError:
        try:
            from .trajectory_fitter import TrajectoryFitter
        except ImportError:
            # Try relative import from parent directory
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            try:
                from utils.trajectory_fitter import TrajectoryFitter
            except ImportError:
                raise ImportError("Could not import TrajectoryFitter. Make sure the module is installed or in the Python path.")
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path.cwd() / "conserved_fitting_results"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    if verbose:
        print(f"Starting conserved sample fitting pipeline...")
        print(f"Output will be saved to: {output_dir}")
    
    # Step 1: Process AnnData into 3D array
    if verbose:
        print("\n1. Processing AnnData object into 3D array...")
    
    gene_names = adata.var_names.tolist()
    batch_names = adata.obs[batch_key].cat.categories.tolist() if hasattr(adata.obs[batch_key], 'cat') else sorted(adata.obs[batch_key].unique())
    
    # Get unique time points from adata for metadata only
    orig_time_points = np.sort(adata.obs[time_key].unique())
    
    # Reshape data into 3D array: (batches, time points, genes)
    try:
        result = anndata_to_3d_matrix(
            adata=adata,
            pseudo_col=time_key,     # Column containing pseudotime
            batch_col=batch_key,     # Column containing batch information
            n_bins=100,              # Number of interpolation points
            adaptive_kernel=True,    # Use adaptive kernel width
            gene_thred=0.1,          # Filter genes expressed in at least 10% of bins
            batch_thred=0.3,         # Set to 0 to keep all batches (was 0.3)
            ensure_tail=True         # Ensure batches cover the tail region
        )
        # Extract components from the result dictionary
        reshaped_data = result['reshaped_data']  # 3D array (batch x time x gene)
        filtered_genes = result['filtered_genes']
        batch_names = result['batch_names']
        
        # Check if any batches were returned
        if reshaped_data.shape[0] == 0:
            if verbose:
                print("   - Warning: No batches passed filtering. Trying again with different parameters...")
            # Try again with even more lenient parameters
            result = anndata_to_3d_matrix(
                adata=adata,
                pseudo_col=time_key,
                batch_col=batch_key,
                n_bins=100,
                adaptive_kernel=False,  # Turn off adaptive kernel
                gene_thred=0.05,        # Lower gene threshold
                batch_thred=0.3,        # No batch threshold
                ensure_tail=False       # Don't ensure tail coverage
            )
            reshaped_data = result['reshaped_data']
            filtered_genes = result['filtered_genes']
            batch_names = result['batch_names']
            
        # If still no batches, we'll need to create synthetic data or raise an error
        if reshaped_data.shape[0] == 0:
            raise ValueError("No batches passed filtering even with lenient parameters. Cannot continue with fitting.")
            
    except Exception as e:
        raise RuntimeError(f"Error reshaping data: {str(e)}")
    
    # Calculate time points for modeling (consistent with build_matrix.py approach)
    time_points = np.linspace(0, 1, reshaped_data.shape[1])
    
    if verbose:
        print(f"   - Data reshaped to 3D array with shape: {reshaped_data.shape}")
        print(f"   - Number of batches: {len(batch_names)}")
        print(f"   - Number of time points: {len(time_points)}")
        print(f"   - Number of genes: {len(filtered_genes)}")
    
    # Step 2: Calculate pairwise distances for conservation analysis
    if verbose:
        print("\n2. Calculating pairwise distances for conservation analysis...")
    
    # Create visualization directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    try:
        conservation_results = calculate_trajectory_conservation(
            trajectory_data=reshaped_data,
            gene_names=filtered_genes, 
            save_dir=output_dir,
            prefix="traj_conservation",
            dtw_radius=3,            # Radius parameter for fastdtw
            use_fastdtw=True,
            normalize='zscore',      # Normalize trajectories before DTW calculation
            filter_samples_by_variation=True,  # Filter out samples with too little variation
            variation_threshold=0.1,          # Minimum coefficient of variation
            variation_metric='max',           # Metric for variation
            min_valid_samples=2               # At least 2 samples needed
        )
        
        # Extract key results
        pairwise_distances = conservation_results['pairwise_distances']
        conservation_scores = conservation_results['conservation_scores']
        similarity_matrix = conservation_results['similarity_matrix']
        print(conservation_scores.head())
        # Print info about pairwise_distances
        if verbose:
            print(f"   - Pairwise distances dictionary contains {len(pairwise_distances)} genes")

    except Exception as e:
        raise RuntimeError(f"Error calculating conservation scores: {str(e)}")
    
    # Get mean conservation scores for each gene - using the correct data structure
    if isinstance(conservation_scores, pd.DataFrame):
        # Sort by normalized_score if it exists
        if 'normalized_score' in conservation_scores.columns:
            # Filter out genes that were filtered by variation if that column exists
            if 'was_filtered' in conservation_scores.columns:
                top_conserved = conservation_scores[~conservation_scores['was_filtered']].head(top_n_genes)
            else:
                top_conserved = conservation_scores.head(top_n_genes)
            
            top_gene_names = top_conserved['gene'].tolist()
        else:
            # Fall back to just taking the first top_n_genes
            top_gene_names = conservation_scores['gene'].tolist()[:top_n_genes]
    else:
        # If conservation_scores is not a DataFrame, assume it's a simple list/array of scores
        sorted_indices = np.argsort(conservation_scores)[:top_n_genes]
        top_gene_names = [filtered_genes[i] for i in sorted_indices]
    print("top_gene_names", top_gene_names)
    if verbose:
        print(f"   - Pairwise distances calculated for {len(filtered_genes)} genes")
        print(f"   - Selected top {top_n_genes} most conserved genes for detailed analysis")
    
    # Step 3: Identify most conserved samples for each gene
    if verbose:
        print("\n3. Identifying most conserved samples for each gene...")
    
    try:
        # Get the number of samples
        n_samples = reshaped_data.shape[0]
        
        # Get most conserved samples for each gene
        conserved_samples = get_most_conserved_samples(
            pairwise_distances, 
            n_samples=n_samples,  # Explicitly passing n_samples
            fraction=conserved_fraction
        )
        print(conserved_samples)
        # Check if conserved_samples is empty
        if not conserved_samples:
            if verbose:
                print("   - Warning: No conserved samples found. Using all samples for each gene.")
            # Create a fallback conserved_samples dictionary
            # For each gene, use all samples
            conserved_samples = {gene_name: list(range(n_samples)) for gene_name in top_gene_names}
        
        avg_samples = sum(len(samples) for samples in conserved_samples.values()) / len(conserved_samples) if conserved_samples else 0
        if verbose:
            print(f"   - Selected {conserved_fraction*100:.0f}% most conserved samples for each gene")
            print(f"   - Average number of samples selected per gene: {avg_samples:.1f}")
    except Exception as e:
        raise RuntimeError(f"Error identifying conserved samples: {str(e)}")
    
    # Step 4: Fit models using only conserved samples
    if verbose:
        print("\n4. Fitting models using only the most conserved samples...")
    
    try:
        # Find indices in filtered_genes for top_gene_names
        top_gene_positions = []
        print("filtering: top_gene_names", top_gene_names)
        for gene_name in top_gene_names:
            gene_pos = np.where(filtered_genes == gene_name)[0]
            if len(gene_pos) > 0:
                top_gene_positions.append(gene_pos[0])
            else:
                raise ValueError(f"Gene {gene_name} not found in filtered_genes")
        
        # Create specialized datasets for each gene
        top_genes_data = []
        print("conserved_samples", conserved_samples)   
        for i, gene_name in enumerate(top_gene_names):
            print("gene_name", gene_name)
            gene_pos = top_gene_positions[i]

            # Get the most conserved samples for this gene
            if gene_name in conserved_samples:
                cons_sample_indices = conserved_samples[gene_name]
                n_cons_samples = len(cons_sample_indices)
                
                # Extract data only for the most conserved samples for this gene
                gene_data = reshaped_data[cons_sample_indices, :, gene_pos]
                
                # Reshape to match expected input format (samples, timepoints, 1 feature)
                gene_data = gene_data.reshape(n_cons_samples, reshaped_data.shape[1], 1)
                
                if verbose and i == 0:  # Print just for the first gene as example
                    print(f"   - For gene {gene_name}: Using {n_cons_samples} out of {n_samples} samples")
            else:
                # Fallback if gene not in conserved_samples
                if verbose:
                    print(f"   - Gene {gene_name}: Using all samples (gene not found in conserved samples dict)")
                gene_data = reshaped_data[:, :, gene_pos:gene_pos+1]
            
            top_genes_data.append(gene_data)
        print('filtered_genes', filtered_genes)
        print('top_gene_names', top_gene_names)
        # Perform fitting using fit_with_conserved_samples
        fitting_results = fit_with_conserved_samples(
            reshaped_data=reshaped_data,  # Pass full reshaped data
            gene_names=top_gene_names,    # Pass all filtered genes
            conserved_samples=conserved_samples,  # Pass conserved samples dict
            top_n_genes=len(top_gene_names),  # Pass actual number of top genes
            n_jobs=n_jobs,
            verbose=verbose,
            interpolation_factor=interpolation_factor,
            model_type=model_type,
            spline_degree=spline_degree,
            spline_smoothing=spline_smoothing,
            use_dtw_optimization=True
        )
        
        standard_results = fitting_results['standard_results']
        optimized_results = fitting_results['optimized_results']
        
        # Add model_score as negative mean of dtw_distances (matching build_matrix.py)
        if 'model_score' not in standard_results:
            standard_results['model_score'] = -np.mean(standard_results['dtw_distances'])
        if 'model_score' not in optimized_results:
            optimized_results['model_score'] = -np.mean(optimized_results['dtw_distances'])
            
        # Add mean_dtw_distance for compatibility with example_pipeline.py
        standard_results['mean_dtw_distance'] = np.mean(standard_results['dtw_distances'])
        optimized_results['mean_dtw_distance'] = np.mean(optimized_results['dtw_distances'])
    
    except Exception as e:
        raise RuntimeError(f"Error fitting models: {str(e)}")
    
    # Step 5: Create visualizations
    if verbose:
        print("\n5. Creating visualizations of fitting results...")
    
    try:
        visualization_paths = visualize_fitting_results(
            standard_results=standard_results,
            optimized_results=optimized_results,
            top_genes_data=top_genes_data,
            top_gene_names=top_gene_names,
            time_points=standard_results['time_points'],  # Use time_points from the fitting results
            output_dir=output_dir,
            max_genes_to_plot=max_genes_to_plot
        )
    except Exception as e:
        if verbose:
            print(f"Warning: Error creating visualizations: {str(e)}")
        visualization_paths = {"error": str(e)}
    
    # Step 6: Generate summary report
    if verbose:
        print("\n6. Generating comprehensive summary report...")
    
    try:
        summary_file = create_fitting_summary(
            standard_results=standard_results,
            optimized_results=optimized_results,
            top_gene_names=top_gene_names,
            top_genes_data=top_genes_data,
            output_file=output_dir / "fitting_summary.txt",
            adata_shape=adata.shape,
            reshaped_data_shape=reshaped_data.shape,
            batch_names=batch_names
        )
    except Exception as e:
        if verbose:
            print(f"Warning: Error creating summary report: {str(e)}")
        summary_file = str(output_dir / "fitting_summary_error.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Error creating summary: {str(e)}\n")
    
    # Calculate overall time
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\nPipeline completed in {elapsed_time:.2f} seconds")
        print(f"Results saved to: {output_dir}")
        print(f"Summary report: {summary_file}")
    
    # Return comprehensive results dictionary
    return {
        'standard_results': standard_results,
        'optimized_results': optimized_results,
        'top_gene_names': top_gene_names,
        'visualizations': visualization_paths if 'visualization_paths' in locals() else None,
        'summary_file': summary_file,
        'pairwise_distances': pairwise_distances,
        'conserved_samples': conserved_samples,
        'top_genes_data': top_genes_data
    }

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

# Convert to 3D matrix
result = anndata_to_3d_matrix(
    adata=adata,
    pseudo_col='pseudotime',
    batch_col='batch',
    n_bins=100,
    adaptive_kernel=True
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
""" 