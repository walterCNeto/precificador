from __future__ import annotations


if not isinstance(self.q_funcs, (list, tuple)):
self.q_funcs = [to_func(self.q_funcs)]
else:
self.q_funcs = [to_func(f) for f in self.q_funcs]


if not isinstance(self.sigma_funcs, (list, tuple)):
self.sigma_funcs = [to_func(self.sigma_funcs)]
else:
self.sigma_funcs = [to_func(f) for f in self.sigma_funcs]


self.dim = max(len(self.q_funcs), len(self.sigma_funcs))
if len(self.q_funcs) == 1 and self.dim > 1:
self.q_funcs = [self.q_funcs[0] for _ in range(self.dim)]
if len(self.sigma_funcs) == 1 and self.dim > 1:
self.sigma_funcs = [self.sigma_funcs[0] for _ in range(self.dim)]
assert len(self.q_funcs) == len(self.sigma_funcs)


self.corr = np.eye(self.dim) if self.corr is None else np.asarray(self.corr, float)
assert self.corr.shape == (self.dim, self.dim)
self._chol = np.linalg.cholesky(self.corr + 1e-12 * np.eye(self.dim))


def simulate_paths(
self,
S0: Sequence[float],
times: np.ndarray,
n_paths: int = 100_000,
antithetic: bool = True,
seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
S0 = np.asarray(S0, float); assert S0.shape == (self.dim,)
times = np.asarray(times, float); assert times[0] == 0.0
Tn = len(times) - 1
n_eff = n_paths if not antithetic else (n_paths + (n_paths % 2)) // 2


rng = Generator(PCG64(seed))
Z = rng.standard_normal(size=(n_eff, Tn, self.dim))
if antithetic:
Z = np.concatenate([Z, -Z], axis=0)
Z = Z[:n_paths]


S = np.empty((n_paths, Tn + 1, self.dim), float)
S[:, 0, :] = S0[None, :]


for k in range(Tn):
t0, t1 = times[k], times[k + 1]
dt = float(t1 - t0)
t_mid = 0.5 * (t0 + t1)
r = self.r_curve.r(t_mid)
q = np.array([f(t_mid) for f in self.q_funcs])
sig = np.array([f(t_mid) for f in self.sigma_funcs])
drift = (r - q - 0.5 * sig ** 2) * dt
vol = sig * math.sqrt(dt)
dW = Z[:, k, :] @ self._chol.T
S[:, k + 1, :] = S[:, k, :] * np.exp(drift[None, :] + vol[None, :] * dW)


return {"times": times, "S": S}


def df(self, t0: float, t1: float) -> float:
return self.r_curve.df(t0, t1)