from derivx import price_from_spec

def price(spec):
    p, se = price_from_spec(spec)
    return p

def test_up_and_out_call_is_below_vanilla_and_monotone_in_barrier():
    S0=100.0; K=100.0; r=0.05; q=0.0; sigma=0.2; T=1.0
    base = {"engine":"mc","model":{"name":"gbm","r":r,"q":q,"sigma":sigma},
            "grid":{"T":T,"steps":128},"S0":[S0],"n_paths":80_000,"seed":7}
    vanilla = price({**base,"product":{"style":"european","type":"european_call","asset":0,"K":K}})
    uo_130 = price({**base,"product":{"style":"european","type":"up_and_out_call","asset":0,"K":K,"barrier":130.0}})
    uo_140 = price({**base,"product":{"style":"european","type":"up_and_out_call","asset":0,"K":K,"barrier":140.0}})
    assert 0.0 <= uo_130 <= vanilla + 1e-8
    assert uo_130 <= uo_140 <= vanilla + 1e-8

def test_asian_arith_call_below_vanilla_call():
    S0=100.0; K=100.0; r=0.05; q=0.0; sigma=0.2; T=1.0
    base = {"engine":"mc","model":{"name":"gbm","r":r,"q":q,"sigma":sigma},
            "grid":{"T":T,"steps":128},"S0":[S0],"n_paths":80_000,"seed":17}
    vanilla = price({**base,"product":{"style":"european","type":"european_call","asset":0,"K":K}})
    asian  = price({**base,"product":{"style":"european","type":"asian_arith_call","asset":0,"K":K}})
    assert asian <= vanilla + 1e-8
