from __future__ import annotations

import math
from typing import Callable, Sequence, Dict, Optional

import numpy as np


class PF:
    @staticmethod
    def terminal(paths: Dict[str, np.ndarray], asset: int) -> np.ndarray:
        return paths["S"][:, -1, asset]

    @staticmethod
    def at_time(paths: Dict[str, np.ndarray], asset: int, t_idx: int) -> np.ndarray:
        return paths["S"][:, t_idx, asset]

    @staticmethod
    def running_max(paths: Dict[str, np.ndarray], asset: int) -> np.ndarray:
        return paths["S"][:, :, asset].max(axis=1)

    @staticmethod
    def running_min(paths: Dict[str, np.ndarray], asset: int) -> np.ndarray:
        return paths["S"][:, :, asset].min(axis=1)

    @staticmethod
    def average(paths: Dict[str, np.ndarray], asset: int, start: int = 1, end: Optional[int] = None) -> np.ndarray:
        S = paths["S"][:, start:end, asset]
        return S.mean(axis=1)

    @staticmethod
    def basket(paths: Dict[str, np.ndarray], weights: Sequence[float], t_idx: Optional[int] = None) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        if t_idx is None:
            ST = paths["S"][:, -1, :]
        else:
            ST = paths["S"][:, t_idx, :]
        return (ST * w[None, :]).sum(axis=1)

    @staticmethod
    def barrier_touched(paths: Dict[str, np.ndarray], asset: int, level: float, direction: str = "up") -> np.ndarray:
        S = paths["S"][:, :, asset]
        if direction == "up":
            return (S.max(axis=1) >= level)
        elif direction == "down":
            return (S.min(axis=1) <= level)
        else:
            raise ValueError("direction must be 'up' or 'down'")


relu = lambda x: np.maximum(x, 0.0)
max_ = np.maximum
min_ = np.minimum
where = np.where


def european_call(asset: int, K: float) -> Callable:
    return lambda paths: relu(PF.terminal(paths, asset) - K)

def european_put(asset: int, K: float) -> Callable:
    return lambda paths: relu(K - PF.terminal(paths, asset))

def asian_arith_call(asset: int, K: float, start: int = 1, end: Optional[int] = None) -> Callable:
    return lambda paths: relu(PF.average(paths, asset, start, end) - K)

def up_and_out_call(asset: int, K: float, barrier: float) -> Callable:
    def _f(paths):
        knocked = PF.barrier_touched(paths, asset, barrier, "up")
        base = relu(PF.terminal(paths, asset) - K)
        base[knocked] = 0.0
        return base
    return _f

def basket_call(weights: Sequence[float], K: float) -> Callable:
    return lambda paths: relu(PF.basket(paths, weights) - K)


# --- Normal CDF sem SciPy ---
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_price(S0, K, r, q, sigma, T):
    if T <= 0:
        return max(S0 - K, 0.0)
    sT = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / sT
    d2 = d1 - sT
    return S0 * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
