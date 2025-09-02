from derivx import price_from_spec, bs_call_price

def test_mc_gbm_call_matches_bs_within_error():
    S0=100.0; K=100.0; r=0.05; q=0.0; sigma=0.2; T=1.0
    spec = {"engine":"mc",
            "model":{"name":"gbm","r":r,"q":q,"sigma":sigma},
            "grid":{"T":T,"steps":128},
            "S0":[S0],
            "product":{"style":"european","type":"european_call","asset":0,"K":K},
            "n_paths":80_000,"seed":42}
    pmc, se = price_from_spec(spec)
    bs = bs_call_price(S0,K,r,q,sigma,T)
    # dentro de 4*SE (estatístico) ou 0.25 abs (fallback)
    tol = max(4.0*se, 0.25)
    assert abs(pmc - bs) < tol
    assert se > 0.0
