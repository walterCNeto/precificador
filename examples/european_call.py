from derivx import price_from_spec, bs_call_price


spec = {
"model": {
"r": 0.12, # 12% a.a. contÃ­nuo
"q": [0.02], # dividend yield
"sigma": [0.25],
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
"n_paths": 100_000,
"seed": 7,
}


price, se = price_from_spec(spec)
print(f"MC: {price:.4f} Â± {1.96*se:.4f}")


bs = bs_call_price(100.0, 100.0, r=0.12, q=0.02, sigma=0.25, T=1.0)
print(f"BS: {bs:.4f}")

