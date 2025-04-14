# trajDTW

A Python package for trajectory analysis and dynamic time warping for cell trajectories.

## Installation

```bash
pip install trajDTW
```

For development installation:

```bash
git clone https://github.com/gilberthan/trajDTW.git
cd trajDTW
pip install -e .
```

## Quick Start

```python
import scanpy as sc
import numpy as np
from trajDTW import anndata_to_3d_matrix, calculate_trajectory_conservation, TrajectoryFitter

# Load AnnData
adata = sc.read_h5ad("your_data.h5ad")

# Convert to 3D matrix using Gaussian kernel interpolation
result = anndata_to_3d_matrix(
    adata=adata,
    pseudo_col='pseudotime',
    batch_col='batch',
    n_bins=100,
    adaptive_kernel=True
)

# Calculate conservation scores
conservation_results = calculate_trajectory_conservation(
    trajectory_data=result['reshaped_data'],
    gene_names=result['filtered_genes'],
    normalize='zscore'
)

# Fit trajectory models
time_points = np.linspace(0, 1, result['reshaped_data'].shape[1])
fitter = TrajectoryFitter(time_points=time_points)
fitting_results = fitter.fit(result['reshaped_data'], model_type='spline')
```

## Usage Guide

### 1. Loading and Preparing Data

The package expects data in AnnData format with pseudotime information and batch/sample annotations:

```python
import scanpy as sc
adata = sc.read_h5ad("your_data.h5ad")
```

### 2. Creating 3D Matrices

Convert your AnnData object to a 3D matrix format (batch × time × gene):

```python
from trajDTW import anndata_to_3d_matrix

result = anndata_to_3d_matrix(
    adata=adata,
    pseudo_col='pseudotime',   # Column in adata.obs with pseudotime values
    batch_col='sample',        # Column in adata.obs with batch/sample annotations
    n_bins=100,                # Number of interpolation bins along pseudotime
    adaptive_kernel=True,      # Use adaptive kernel width based on cell density
    gene_thred=0.1,            # Filter genes expressed in at least 10% of bins
    batch_thred=0.3            # Filter batches covering at least 30% of timeline
)

# Access the results
reshaped_data = result['reshaped_data']  # 3D array (batch × time × gene)
filtered_genes = result['filtered_genes'] # Gene names
batch_names = result['batch_names']      # Batch/sample names
```

### 3. Computing Trajectory Conservation

Calculate how conserved each gene's trajectory is across batches/samples:

```python
from trajDTW import calculate_trajectory_conservation

conservation_results = calculate_trajectory_conservation(
    trajectory_data=reshaped_data,
    gene_names=filtered_genes,
    normalize='zscore',                   # Normalize trajectories before comparison
    filter_samples_by_variation=True,     # Filter out samples with low variation
    variation_threshold=0.1               # Minimum variation threshold
)

# Access conservation scores
conservation_scores = conservation_results['conservation_scores']
print("Top 10 most conserved genes:")
print(conservation_scores.head(10))
```

### 4. Fitting Trajectory Models

Fit mathematical models to gene trajectories:

```python
from trajDTW import TrajectoryFitter
import numpy as np

# Create time points
time_points = np.linspace(0, 1, reshaped_data.shape[1])

# Initialize fitter
fitter = TrajectoryFitter(
    time_points=time_points,
    n_jobs=4,                # Use 4 parallel jobs for fitting
    verbose=True
)

# Fit different model types
model_types = ['spline', 'polynomial', 'sine', 'double_sine']
results = {}

for model_type in model_types:
    print(f"Fitting {model_type} model...")
    results[model_type] = fitter.fit(
        reshaped_data,        # 3D data array 
        model_type=model_type,
        # Model-specific parameters:
        spline_degree=3,
        polynomial_degree=4,
        optimize_spline_dtw=True  # For spline models, optimize smoothing parameter
    )
    
# Compare model performance
for model_type, result in results.items():
    mean_dtw = np.mean(result['dtw_distances'])
    print(f"{model_type} model: mean DTW distance = {mean_dtw:.4f}")
```

## Example Scripts

Check the `examples/` directory for complete usage examples:

- `simple_example.py`: Basic usage demonstration
- `run_pipeline.py`: Complete analysis pipeline

## Features

- Gaussian trajectory interpolation for single-cell data
- Calculation of trajectory conservation scores using DTW
- Trajectory fitting with various models (sine, polynomial, spline)
- Visualization utilities for trajectory data

## License

MIT License 