from derivx import price_from_spec
from tests.ref_formulas.digitals import (
    cash_or_nothing_call, asset_or_nothing_call
)

def test_cash_or_nothing_call_mc_vs_closed():
    S0=K=100.0; r=0.05; q=0.0; sig=0.2; T=1.0
    spec = {
        "engine":"mc",
        "model":{"name":"gbm","r":r,"q":q,"sigma":sig},
        "grid":{"T":T,"steps":256},
        "S0":[S0],
        "product":{"style":"european","type":"cash_or_nothing_call","asset":0,"K":K,"cash":1.0},
        "n_paths":200_000, "seed":11
    }
    pmc, se = price_from_spec(spec)
    pref = cash_or_nothing_call(S0,K,r,q,sig,T, cash=1.0)
    tol = max(6*se, 0.01 * max(1.0, pref))
    assert abs(pmc - pref) <= tol

def test_asset_or_nothing_call_mc_vs_closed():
    S0=K=100.0; r=0.05; q=0.0; sig=0.2; T=1.0
    spec = {
        "engine":"mc",
        "model":{"name":"gbm","r":r,"q":q,"sigma":sig},
        "grid":{"T":T,"steps":256},
        "S0":[S0],
        "product":{"style":"european","type":"asset_or_nothing_call","asset":0,"K":K},
        "n_paths":200_000, "seed":13
    }
    pmc, se = price_from_spec(spec)
    pref = asset_or_nothing_call(S0,K,r,q,sig,T)
    tol = max(6*se, 0.01 * max(1.0, pref))
    assert abs(pmc - pref) <= tol
