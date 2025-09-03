import math
from derivx import price_from_spec
from tests.ref_formulas.closed_form import bs_call_price, bs_put_price

def test_vanilla_call_bs_matches_reference():
    S0=K=100.0; r=0.05; q=0.0; sig=0.2; T=1.0
    spec = {
        "engine":"mc",
        "model":{"name":"gbm","r":r,"q":q,"sigma":sig},
        "grid":{"T":T,"steps":64},
        "S0":[S0],
        "product":{"style":"european","type":"european_call","asset":0,"K":K},
        "n_paths":80_000, "seed":42
    }
    pmc,se = price_from_spec(spec)
    pref   = bs_call_price(S0,K,r,q,sig,T)
    tol = max(4*se, 2.5e-3)
    assert abs(pmc - pref) <= tol

def test_put_call_parity_mc_within_tolerance():
    S0=K=100.0; r=0.05; q=0.0; sig=0.2; T=1.0
    # Call via MC
    spec_c = {
        "engine":"mc",
        "model":{"name":"gbm","r":r,"q":q,"sigma":sig},
        "grid":{"T":T,"steps":64},
        "S0":[S0],
        "product":{"style":"european","type":"european_call","asset":0,"K":K},
        "n_paths":80_000, "seed":123
    }
    c_mc, se_c = price_from_spec(spec_c)

    # Put via BS exato
    p_bs = bs_put_price(S0,K,r,q,sig,T)

    lhs = c_mc - p_bs
    rhs = S0*math.exp(-q*T) - K*math.exp(-r*T)
    tol = max(4*se_c, 3e-3)
    assert abs(lhs - rhs) <= tol
