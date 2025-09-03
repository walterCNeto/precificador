from derivx import price_from_spec
from tests.ref_formulas.margrabe import margrabe_exchange_call

def test_exchange_call_mc_vs_margrabe():
    S1=100.0; S2=120.0
    r=0.05; q1=0.01; q2=0.03
    sig1=0.20; sig2=0.30; rho=0.5; T=1.0

    spec = {
        "engine":"mc",
        "model":{"name":"gbm","r":r,"q":[q1,q2],"sigma":[sig1,sig2],
                 "corr":[[1.0, rho],[rho,1.0]]},
        "grid":{"T":T,"steps":256},
        "S0":[S1, S2],
        "product":{"style":"european","type":"exchange_call","asset_long":0,"asset_short":1},
        "n_paths":180_000, "seed":17
    }
    pmc, se = price_from_spec(spec)
    pref = margrabe_exchange_call(S1,S2,r,q1,q2,sig1,sig2,rho,T)

    # tolerância conservadora (6·SE ou 1.5%)
    tol = max(6*se, 0.015*max(1.0, pref))
    assert abs(pmc - pref) <= tol
