from __future__ import annotations




def _build_exercise(product: Dict[str, Any], times: np.ndarray):
style = product.get("style", "european").lower()
if style == "european":
return None


# Bermudan/American
if style == "american":
# todas as datas da grade (exceto t=0)
ex_idx = list(range(1, len(times)))
else:
# bermudan: a cada 'freq' passos ou lista explícita de índices/tempos
if "exercise_idx" in product:
ex_idx = list(map(int, product["exercise_idx"]))
elif "exercise_times" in product:
# converter tempos para índices
tset = set(product["exercise_times"])
ex_idx = [i for i, t in enumerate(times) if t in tset]
else:
freq = int(product.get("exercise_every", 8))
ex_idx = list(range(freq, len(times), freq))
if (len(times) - 1) not in ex_idx:
ex_idx.append(len(times) - 1)


# immediate payoff para put/call genérico
ptype = product.get("type", "european_call")
K = float(product.get("K", 100.0))
asset = int(product.get("asset", 0))


from ..payoffs.core import PF, relu


if "put" in ptype:
def imm(paths, k):
St = PF.at_time(paths, asset, k)
return relu(K - St)
else:
def imm(paths, k):
St = PF.at_time(paths, asset, k)
return relu(St - K)


return ExerciseSpec(exercise_idx=ex_idx, immediate_payoff=imm)




# --- Orquestração ---


def price_from_spec(spec: Dict[str, Any]):
eng, times, S0 = build_engine_from_spec(spec)
product = spec["product"]
style = product.get("style", "european").lower()


if style == "european":
payoff = _build_payoff(product)
return eng.price(payoff, S0, times, n_paths=int(spec.get("n_paths", 100_000)), seed=spec.get("seed"))
else:
ex = _build_exercise(product, times)
return eng.price_exercisable(ex, S0, times, n_paths=int(spec.get("n_paths", 120_000)), seed=spec.get("seed"))