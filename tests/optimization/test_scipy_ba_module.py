from slam.optimization.scipy_ba import solve_bundle_adjustment


def test_scipy_ba_module_exports_solver():
    assert callable(solve_bundle_adjustment)
