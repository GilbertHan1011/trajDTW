# trajDTW

A Python package for trajectory conservation analysis and dynamic time warping for cell trajectories.

## Installation



For development installation:

```bash
git clone https://github.com/GilbertHan1011/trajDTW.git
cd trajDTW
pip install -e .
```

## Quick Start

### Method 1: Step-by-step approach

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

### Method 2: All-in-one pipeline approach

```python
from pathlib import Path
from trajDTW import run_trajectory_conservation_analysis

# Define input path and output directory
adata_path = "your_data.h5ad"
output_dir = Path("./output_results")

# Run the complete trajectory conservation analysis pipeline
results = run_trajectory_conservation_analysis(
    adata_path=adata_path,
    output_dir=output_dir,
    pseudo_col='pseudotime',
    batch_col='batch',
    n_bins=100,
    adaptive_kernel=True,
    gene_thred=0.1,
    batch_thred=0.3,
    variation_filter_level='basic',
    top_n_genes=100,
    spline_smoothing=0.5,
    n_jobs=4,
    save_figures=True
)

# Access the results
conservation_results = results['conservation_results']
fit_results = results['fit_results']
filtered_genes = results['filtered_genes']
selected_genes = results['selected_genes']

# Print top conserved genes
print(conservation_results['conservation_scores'].head(10))
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

### 5. Using the All-in-One Pipeline

For a streamlined workflow that handles all the steps automatically:

```python
from pathlib import Path
from trajDTW import run_trajectory_conservation_analysis

# Define path to AnnData file and output directory
adata_path = "your_data.h5ad"
output_dir = Path("./output_dir")
output_dir.mkdir(exist_ok=True, parents=True)

# Run the complete analysis pipeline
results = run_trajectory_conservation_analysis(
    adata_path=adata_path,           # Path to the AnnData h5ad file
    output_dir=output_dir,           # Directory to save output files
    pseudo_col='pseudotime',         # Column in adata.obs containing pseudotime values
    batch_col='batch',               # Column in adata.obs containing batch information
    n_bins=100,                      # Number of interpolation points along pseudotime
    adaptive_kernel=True,            # Whether to use adaptive kernel width for interpolation
    gene_thred=0.1,                  # Filter genes expressed in at least this fraction of bins
    batch_thred=0.3,                 # Filter batches covering at least this fraction of timeline
    tail_num=0.05,                   # Size of tail region
    ensure_tail=True,                # Ensure batches cover the tail region
    dtw_radius=3,                    # Radius parameter for fastdtw algorithm
    use_fastdtw=True,                # Whether to use fastdtw algorithm
    normalize='zscore',              # Method to normalize trajectories before DTW calculation
    variation_filter_level='basic',  # Level of filtering ('off', 'basic', 'stringent')
    top_n_genes=100,                 # Number of top conserved genes to select for fitting
    spline_smoothing=0.5,            # Smoothing parameter for spline fitting
    interpolation_factor=2,          # Factor for increasing time point density
    n_jobs=4,                        # Number of parallel jobs (-1 for all available cores)
    save_figures=True,               # Whether to save visualization figures
    layer=None                       # Layer in anndata to use (None = default X matrix)
)

# This pipeline will:
# 1. Load the AnnData file
# 2. Process it into a 3D matrix
# 3. Calculate conservation scores for all genes
# 4. Identify the most conserved samples for each gene
# 5. Fit standard and DTW-optimized models using conserved samples
# 6. Create visualizations (heatmaps, clustering of genes, trajectory plots)
# 7. Save all results and figures to the output directory

# Access the results
conservation_results = results['conservation_results']
fit_results = results['fit_results']
filtered_genes = results['filtered_genes']
selected_genes = results['selected_genes']
reshaped_data = results['reshaped_data']

# The output directory will contain:
# - 3D array data saved as numpy files
# - CSV files with conserved genes and clusters
# - Heatmaps of fitted trajectories
# - K-means clustering results
# - Log file with detailed analysis steps
```

## Example Scripts

Check the `examples/` directory for complete usage examples:

- `simple_example.py`: Demonstrates both step-by-step and all-in-one approaches
- `run_pipeline.py`: Complete analysis pipeline using the all-in-one function

## Features

- Gaussian trajectory interpolation for single-cell data
- Calculation of trajectory conservation scores using DTW
- Trajectory fitting with various models (sine, polynomial, spline)
- All-in-one pipeline for streamlined analysis
- Automated clustering of conserved genes
- Comprehensive visualization utilities for trajectory data
- Detailed logging and results export

## License

MIT License 
