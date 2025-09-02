from derivx import bs_call_price, PiecewiseFlatCurve
import numpy as np, math

def test_bs_call_price_numeric():
    v = bs_call_price(100, 100, 0.05, 0.0, 0.2, 1.0)
    assert 9.0 < v < 11.5

def test_curve_df_basic():
    rc = PiecewiseFlatCurve(np.array([1e-8]), np.array([0.05]))
    assert abs(rc.df(0.0, 1.0) - math.exp(-0.05)) < 5e-4
