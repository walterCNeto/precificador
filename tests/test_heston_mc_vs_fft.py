import pytest
from derivx import price_from_spec

@pytest.mark.slow
def test_heston_mc_close_to_fft():
    common = {"name":"heston","r":0.05,"q":0.0,"kappa":1.5,"theta":0.04,"xi":0.5,"rho":-0.7,"v0":0.04}
    S0=100.0; K=100.0; T=1.0
    spec_fft = {"engine":"fft","model":common,"grid":{"T":T},"S0":[S0],
                "product":{"style":"european","type":"european_call","asset":0,"K":K},
                "alpha":1.5,"N":4096,"eta":0.25}
    p_fft,_ = price_from_spec(spec_fft)

    spec_mc = {"engine":"mc","model":common,"grid":{"T":T,"steps":512},"S0":[S0],
               "product":{"style":"european","type":"european_call","asset":0,"K":K},
               "n_paths":120_000,"seed":7}
    p_mc,se = price_from_spec(spec_mc)

    # tolerância estatística generosa (MC tem viés em Δt): 6*SE ou 0.7 abs
    tol = max(6.0*se, 0.7)
    assert abs(p_mc - p_fft) < tol
