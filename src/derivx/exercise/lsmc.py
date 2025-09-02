from __future__ import annotations
cash = np.zeros(n, dtype=float)


# Pré-computar discounts entre quaisquer dois exercise times
# Usaremos df(0, t_tau) no final; para continuidade usamos df(t_k, t_tau)
def df(t0, t1):
return model.df(float(t0), float(t1))


# Comece na última data: se ITM, exercer; senão, nada (tau=-1)
k_last = ex_idx[-1]
imm_last = np.asarray(spec.immediate_payoff(paths, k_last), float)
take_last = imm_last > 1e-12
tau_idx[take_last] = k_last
cash[take_last] = imm_last[take_last]


# Retroceda
for k in reversed(ex_idx[:-1]):
imm = np.asarray(spec.immediate_payoff(paths, k), float)
not_done = tau_idx < 0 # ainda não exerceu no futuro
# Y = valor de continuar: cash futuro descontado de t_tau para t_k (0 caso não haja cash futuro)
Y = np.zeros(n, float)
has_future = tau_idx >= 0
if np.any(has_future):
t_k = times[k]
t_tau = times[tau_idx[has_future]]
Y[has_future] = cash[has_future] * np.array([df(t_k, tti) for tti in t_tau])


# Regressão apenas nos caminhos candidatos (not_done) e, usualmente, ITM melhora estabilidade
candidates = not_done & (imm > 1e-12)
if np.any(candidates):
phi = basis_fn(feature_fn(paths, k))[candidates, :]
y = Y[candidates]
if phi.shape[0] >= phi.shape[1]: # overspecified
beta = _regress(phi, y)
cont_all = basis_fn(feature_fn(paths, k)) @ beta
else:
# poucos pontos: use média como fallback
cont_all = np.full(n, y.mean() if y.size else 0.0)
else:
cont_all = np.zeros(n)


# Decisão ótima: exerça onde imm >= cont; só para quem não exerceu ainda
take = candidates & (imm >= cont_all)
tau_idx[take] = k
cash[take] = imm[take]
# quem não tomou segue com a decisão futura já gravada


# Preço = E[df(0, t_tau) * cash]; df(0, t_tau) = 0 quando tau=-1 (cash=0)
disc = np.ones(n, float)
valid = tau_idx >= 0
if np.any(valid):
disc[valid] = np.array([df(0.0, times[i]) for i in tau_idx[valid]])
price_paths = disc * cash
price = float(price_paths.mean())
se = float(price_paths.std(ddof=1) / np.sqrt(n))
return price, se