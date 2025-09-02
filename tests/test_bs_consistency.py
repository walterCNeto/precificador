import numpy as np
from derivx import price_from_spec, bs_call_price
# Se vocÃª renomeou o pacote para 'precificador', troque a linha acima por:
# from precificador import price_from_spec, bs_call_price

def test_european_call_bs_consistency():
    spec = {
        "model": {
            "r": 0.05,
            "q": [0.00],
            "sigma": [0.2],
            "corr": [[1.0]],
        },
        "grid": {"T": 1.0, "steps": 64},
        "S0": [100.0],
        "product": {
            "style": "european",
            "type": "european_call",
            "asset": 0,
            "K": 100.0,
        },
        "n_paths": 30_000,
        "seed": 123,
    }
    price, se = price_from_spec(spec)
    bs = bs_call_price(100.0, 100.0, r=0.05, q=0.0, sigma=0.2, T=1.0)


