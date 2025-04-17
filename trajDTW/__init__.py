"""
trajDTW: A package for trajectory analysis and dynamic time warping for cell trajectories.
"""

__version__ = "0.1.0"

from .cell_interpolation import (
    GaussianTrajectoryInterpolator,
    anndata_to_3d_matrix,
    visualize_fitting_results,
    create_fitting_summary,
    fit_with_conserved_samples,
    create_gene_position_mapping,
    run_trajectory_conservation_analysis
)

from .trajectory_conservation import (
    calculate_trajectory_conservation,
    get_most_conserved_samples,
    extract_pairwise_distances
)

from .trajectory_fitter import TrajectoryFitter

# Import trajectory utilities
from .trajectory_utils import (
    normalize_trajectory,
    calculate_trajectory_variation,
    compute_dtw,
    optimized_dtw
)

# Import visualization functions
from .visualization import (
    plot_interpolation_example,
    plot_conservation_scores,
    plot_fitted_trajectories,
    plot_clusters,
    plot_model_comparison,
    plot_feature_variation
)

