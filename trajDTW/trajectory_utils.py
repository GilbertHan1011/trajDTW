#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Utility Functions

This module provides utility functions for working with trajectory data,
including normalization, variation calculation, and Dynamic Time Warping (DTW)
distance metrics.
"""

import numpy as np
import warnings
from typing import Union, Optional, Tuple, List, Dict, Any

# Try to import fastdtw, but continue if not available
try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None
    warnings.warn("fastdtw not installed. Will use alternative DTW implementations.")

# Try to import scipy's DTW if available (newer versions)
try:
    from scipy.spatial.distance import dtw as scipy_dtw
except ImportError:
    scipy_dtw = None

def normalize_trajectory(x: np.ndarray, method: str = 'zscore', eps: float = 1e-10) -> np.ndarray:
    """
    Normalize a trajectory for DTW comparison to make distances scale-invariant.
    
    Parameters
    ----------
    x : array-like
        Trajectory array to normalize
    method : str, optional
        Normalization method: 'zscore', 'minmax', 'cv', or 'none'
    eps : float, optional
        Small constant to avoid division by zero
        
    Returns
    -------
    np.ndarray
        Normalized trajectory
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Handle flat trajectories specially
    if np.allclose(x, x[0], rtol=1e-5, atol=1e-8):
        return np.zeros_like(x)
        
    if method == 'none':
        return x
    elif method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        std = np.std(x)
        if std < eps:  # Handle near-constant trajectories
            return np.zeros_like(x)
        return (x - np.mean(x)) / (std + eps)
    elif method == 'minmax':
        # Min-max normalization (range [0, 1])
        min_val = np.min(x)
        range_val = np.max(x) - min_val
        if range_val < eps:  # Handle near-constant trajectories
            return np.zeros_like(x)
        return (x - min_val) / (range_val + eps)
    elif method == 'cv':
        # Coefficient of variation normalization
        mean_val = np.mean(x)
        if abs(mean_val) < eps:  # Handle near-zero mean
            return x / (np.std(x) + eps)
        return x / (mean_val + eps)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def calculate_trajectory_variation(trajectory: np.ndarray, metric: str = 'max', eps: float = 1e-10) -> float:
    """
    Calculate variation in a trajectory using different metrics.
    
    Parameters
    ----------
    trajectory : array-like
        Trajectory array
    metric : str, optional
        Variation metric: 'cv', 'std', 'range', 'mad', or 'max'
    eps : float, optional
        Small constant to avoid division by zero
        
    Returns
    -------
    float
        Variation value
    """
    trajectory = np.asarray(trajectory)
    
    if metric == 'cv':  # Coefficient of variation
        mean = np.mean(trajectory)
        if abs(mean) < eps:
            return np.std(trajectory) / eps  # Avoid division by zero
        return np.std(trajectory) / abs(mean)
    
    elif metric == 'std':  # Standard deviation
        return np.std(trajectory)
    
    elif metric == 'range':  # Range (max - min)
        return np.max(trajectory) - np.min(trajectory)
    
    elif metric == 'mad':  # Mean absolute deviation
        return np.mean(np.abs(trajectory - np.mean(trajectory)))
    
    elif metric == 'max':  # Maximum value
        return np.max(trajectory)
    
    else:
        raise ValueError(f"Unknown variation metric: {metric}")

def optimized_dtw(x: np.ndarray, y: np.ndarray, window: int = 3) -> float:
    """
    Optimized DTW implementation with window constraint for faster computation.
    
    Parameters
    ----------
    x, y : np.ndarray
        Time series to compare
    window : int, optional
        Window constraint for DTW calculation
        
    Returns
    -------
    float
        DTW distance between x and y
    """
    n, m = len(x), len(y)
    w = max(window, abs(n-m))  # Ensure window is at least as large as length difference
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        # Define the range of columns to consider for this row
        j_start = max(1, i-w)
        j_end = min(m+1, i+w+1)
        
        for j in range(j_start, j_end):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # insertion
                dtw_matrix[i, j-1],     # deletion
                dtw_matrix[i-1, j-1]    # match
            )
    
    return dtw_matrix[n, m]

def compute_dtw(x: np.ndarray, y: np.ndarray, radius: int = 3, 
               norm_method: str = 'zscore', 
               use_fastdtw: bool = True) -> float:
    """
    Compute DTW distance between two trajectories with normalization.
    
    This function tries to use the fastest available DTW implementation,
    falling back to simpler methods if needed:
    1. fastdtw (if available and use_fastdtw=True)
    2. scipy's DTW (if available)
    3. Custom optimized DTW implementation
    4. Euclidean distance (last resort)
    
    Parameters
    ----------
    x, y : np.ndarray
        Trajectories to compare
    radius : int, optional
        Constraint window size for fastdtw
    norm_method : str, optional
        Normalization method to use ('zscore', 'minmax', 'cv', 'none')
    use_fastdtw : bool, optional
        Whether to try using fastdtw library
        
    Returns
    -------
    float
        DTW distance between normalized trajectories
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Flatten arrays if multi-dimensional
    if x.ndim > 1:
        x = x.flatten()
    if y.ndim > 1:
        y = y.flatten()
        
    # Ensure both arrays have the same length
    if len(x) != len(y):
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
    
    # Normalize trajectories to make comparison fair
    x_norm = normalize_trajectory(x, method=norm_method)
    y_norm = normalize_trajectory(y, method=norm_method)
    
    # Define a custom distance function that works with 1-D arrays
    def custom_dist(a, b):
        return np.linalg.norm(a - b)
    
    # Calculate DTW on normalized data
    if use_fastdtw and fastdtw is not None:
        try:
            distance, _ = fastdtw(x_norm, y_norm, dist=custom_dist, radius=radius)
            return distance
        except Exception as e:
            warnings.warn(f"Error in fastdtw: {str(e)}. Falling back to alternative method.")
            # Fall back to simple Euclidean distance if fastdtw fails
            return np.sqrt(np.sum((x_norm - y_norm) ** 2))
    
    # Try to use scipy's DTW if available
    if scipy_dtw is not None:
        try:
            distance, _, _, _ = scipy_dtw(x_norm, y_norm, dist=custom_dist)
            return distance
        except Exception as e:
            warnings.warn(f"Error in scipy_dtw: {str(e)}. Falling back to optimized_dtw.")
    
    # If neither fastdtw nor scipy_dtw worked, use our optimized implementation
    return optimized_dtw(x_norm, y_norm, window=radius) 