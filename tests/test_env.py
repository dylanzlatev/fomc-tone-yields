"""Environment smoke tests: catch BLAS/DLL breakage early (see environment.yml)."""
import numpy as np


def test_numpy_linalg_runs():
    rng = np.random.default_rng(0)
    x = rng.random((50, 3))
    y = x @ np.array([1.0, 2.0, 3.0])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    assert np.allclose(beta, [1.0, 2.0, 3.0])


def test_statsmodels_hac_ols_runs():
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    x = rng.random(100)
    y = 2.0 * x + rng.normal(0, 0.01, 100)
    res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    assert abs(res.params[1] - 2.0) < 0.05
