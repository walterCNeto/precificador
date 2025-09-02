from derivx import price_from_spec

def run(spec):
    price, se = price_from_spec(spec)
    print(f"{spec['product']}: {price:.4f}  ± {1.96*se:.4f}")

base = {
    "model": {"r": 0.05, "q": [0.00], "sigma": [0.20], "corr": [[1.0]]},
    "grid": {"T": 1.0, "steps": 64},
    "S0": [100.0],
    "seed": 123,
}

for prod in [
    {"style": "european", "type": "european_call", "asset": 0, "K": 100.0},
    {"style": "european", "type": "european_put", "asset": 0, "K": 100.0},
    {"style": "european", "type": "asian_arith_call", "asset": 0, "K": 100.0},
    {"style": "european", "type": "up_and_out_call", "asset": 0, "K": 100.0, "barrier": 130.0},
    {"style": "bermudan", "type": "european_put", "asset": 0, "K": 100.0, "exercise_every": 16},
]:
    spec = base | {"product": prod, "n_paths": 40_000}
    run(spec)
