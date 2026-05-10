"""SciPy bundle adjustment public module.

The implementation lives in `slam.optimization.bundle_adjustment`; this module
keeps the target repository shape stable for examples and users.
"""

from slam.optimization.bundle_adjustment import (
    BALObservation,
    BALProblem,
    BundleAdjustmentResult,
    bal_jacobian_sparsity,
    pack_bal_parameters,
    project_bal_point,
    read_bal_problem,
    reprojection_residuals,
    reprojection_rmse,
    residuals_from_parameters,
    solve_bundle_adjustment,
    unpack_bal_parameters,
)

__all__ = [
    "BALObservation",
    "BALProblem",
    "BundleAdjustmentResult",
    "bal_jacobian_sparsity",
    "pack_bal_parameters",
    "project_bal_point",
    "read_bal_problem",
    "reprojection_residuals",
    "reprojection_rmse",
    "residuals_from_parameters",
    "solve_bundle_adjustment",
    "unpack_bal_parameters",
]
