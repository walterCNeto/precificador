from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Dict
from ..models.gbm import RiskNeutralGBM
from ..exercise.lsmc import ExerciseSpec, lsmc_price


@dataclass
class MonteCarloEngine:
model: RiskNeutralGBM


def price(
self,
payoff: Callable[[Dict[str, np.ndarray]], np.ndarray],
S0: Sequence[float],
times: np.ndarray,
n_paths: int = 100_000,
antithetic: bool = True,
seed: Optional[int] = None,
control_variate: Optional[Tuple[Callable, float]] = None,
) -> Tuple[float, float]:
paths = self.model.simulate_paths(S0, times, n_paths, antithetic, seed)
X = np.asarray(payoff(paths), float)
if X.ndim == 0:
X = np.full((paths["S"].shape[0],), float(X))
df0T = self.model.df(0.0, float(times[-1]))
disc_X = df0T * X


if control_variate is not None:
cv_func, cv_true = control_variate
Y = df0T * np.asarray(cv_func(paths), float)
cov = np.cov(disc_X, Y, bias=False)
varY = cov[1, 1]
if varY > 0:
beta = cov[0, 1] / varY
disc_X = disc_X - beta * (Y - cv_true)


price = float(disc_X.mean())
se = float(disc_X.std(ddof=1) / math.sqrt(disc_X.size))
return price, se


def price_exercisable(
self,
spec: ExerciseSpec,
S0: Sequence[float],
times: np.ndarray,
n_paths: int = 120_000,
antithetic: bool = True,
seed: Optional[int] = None,
) -> Tuple[float, float]:
paths = self.model.simulate_paths(S0, times, n_paths, antithetic, seed)
return lsmc_price(self.model, paths, spec, times)