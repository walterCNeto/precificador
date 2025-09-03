from derivx import price_from_spec
from tests.ref_formulas.closed_form import asian_arith_call_levy

def test_asian_arith_call_mc_vs_levy():
    S0=K=100.0; r=0.05; q=0.0; sig=0.2; T=1.0
    steps = 256
    n_paths = 200_000

    # MC (seu produto já existente): asian_arith_call
    spec = {
        "engine":"mc",
        "model":{"name":"gbm","r":r,"q":q,"sigma":sig},
        "grid":{"T":T,"steps":steps},  # média no tempo dos passos
        "S0":[S0],
        "product":{"style":"european","type":"asian_arith_call","asset":0,"K":K},
        "n_paths":n_paths, "seed":19
    }
    pmc, se = price_from_spec(spec)

    # Aproximação de Lévy (Haug)
    pref = asian_arith_call_levy(S0,K,r,q,sig,T, n=steps)

    # Tolerância: max(6·SE, 1.5% do preço)
    tol = max(6*se, 0.015 * max(1.0, pref))
    assert abs(pmc - pref) <= tol
