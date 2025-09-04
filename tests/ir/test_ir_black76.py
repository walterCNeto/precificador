# tests/ir/test_ir_black76.py
import numpy as np
from derivx import price_from_spec
from derivx.ir.black76 import fra_forward_rate, swap_par_rate, black_call_forward, black_put_forward
from derivx.curves import PiecewiseFlatCurve

def test_fra_at_par_has_zero_pv():
    # curva flat 5%
    r = 0.05
    curve = PiecewiseFlatCurve(np.array([1e-8]), np.array([r]))
    D = lambda t: curve.df(0.0, float(t))
    T1, T2 = 0.5, 1.0
    tau = T2 - T1
    F = fra_forward_rate(D(T1), D(T2), tau)
    spec = {
        "engine":"analytic",
        "model":{"r":r},
        "grid":{"T":T2},
        "S0":[1.0],
        "product":{"style":"european","type":"fra","T1":T1,"T2":T2,"tau":tau,"K":F}
    }
    pv,_ = price_from_spec(spec)
    assert abs(pv) < 1e-12

def test_swap_at_par_has_zero_pv():
    r = 0.03
    curve = PiecewiseFlatCurve(np.array([1e-8]), np.array([r]))
    D = lambda t: curve.df(0.0, float(t))
    T0 = 0.0
    pay_times = [1.0, 2.0, 3.0, 4.0, 5.0]
    taus = [1.0]*len(pay_times)
    k_par = swap_par_rate(D(T0), [D(t) for t in pay_times], taus)
    spec = {
        "engine":"analytic",
        "model":{"r":r},
        "grid":{"T":pay_times[-1]},
        "S0":[1.0],
        "product":{"style":"european","type":"swap","T0":T0,"payment_times":pay_times,"tau":taus,"fixed_rate":k_par}
    }
    pv,_ = price_from_spec(spec)
    assert abs(pv) < 1e-12

def test_cap_monotonic_in_strike():
    r = 0.02
    curve = PiecewiseFlatCurve(np.array([1e-8]), np.array([r]))
    D = lambda t: curve.df(0.0, float(t))
    pay_times = [1,2,3,4,5]
    taus = [1.0]*5
    reset_times = [t-1.0 for t in pay_times]
    base = {
        "engine":"analytic",
        "model":{"r":r},
        "grid":{"T":5.0},
        "S0":[1.0],
        "product":{"style":"european","type":"cap",
                   "payment_times":pay_times,"tau":taus,"reset_times":reset_times,"sigma":0.2}
    }
    low = {**base, "product":{**base["product"], "K":0.01}}
    hi  = {**base, "product":{**base["product"], "K":0.05}}
    pL,_ = price_from_spec(low)
    pH,_ = price_from_spec(hi)
    assert pL >= pH - 1e-12

def test_black76_caplet_put_call_parity():
    # paridade: D(0,T2)*tau*(Call - Put) = D(0,T2)*tau*(F - K)
    r = 0.04
    curve = PiecewiseFlatCurve(np.array([1e-8]), np.array([r]))
    D = lambda t: curve.df(0.0, float(t))
    T1, T2 = 1.0, 1.5
    tau = T2 - T1
    F = fra_forward_rate(D(T1), D(T2), tau)
    K = F * 1.05
    sigma = 0.25
    # Black-76 no forward (sem DF, sem tau)
    c = black_call_forward(F, K, sigma, T1)
    p = black_put_forward(F, K, sigma, T1)
    lhs = D(T2) * tau * (c - p)
    rhs = D(T2) * tau * (F - K)
    assert abs(lhs - rhs) < 1e-12

def test_payer_swaption_positive_when_strike_below_par():
    r = 0.03
    curve = PiecewiseFlatCurve(np.array([1e-8]), np.array([r]))
    D = lambda t: curve.df(0.0, float(t))
    T0 = 2.0                      # expiração da swaption
    pay_times = [3.0, 4.0, 5.0]   # swap 3 anos anual após T0
    taus = [1.0, 1.0, 1.0]
    k_par = swap_par_rate(D(T0), [D(t) for t in pay_times], taus)
    K = 0.9 * k_par
    spec = {
        "engine":"analytic",
        "model":{"r":r},
        "grid":{"T":5.0},
        "S0":[1.0],
        "product":{"style":"european","type":"payer_swaption","expiry":T0,
                   "payment_times":pay_times,"tau":taus,"K":K,"sigma":0.25}
    }
    pv,_ = price_from_spec(spec)
    assert pv > 0.0
