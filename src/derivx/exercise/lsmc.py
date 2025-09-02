from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
from ..models.gbm import RiskNeutralGBM


@dataclass
class ExerciseSpec:
    exercise_idx: List[int]
    immediate_payoff: Callable[[Dict[str, np.ndarray], int], np.ndarray]


def default_basis(features: np.ndarray) -> np.ndarray:
    x = features
    cols = [np.ones((x.shape[0], 1))]
    cols.append(x)
    cols.append(x ** 2)
    if x.shape[1] > 1:
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                cols.append(x[:, [i]] * x[:, [j]])
    return np.concatenate(cols, axis=1)


def make_features(paths: Dict[str, np.ndarray], t_idx: int) -> np.ndarray:
    St = paths["S"][:, t_idx, :]
    return np.log(np.clip(St, 1e-12, None))


def _regress(phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(phi) @ y


def lsmc_price(
    model: RiskNeutralGBM,
    paths: Dict[str, np.ndarray],
    spec: ExerciseSpec,
    times: np.ndarray,
    feature_fn: Callable[[Dict[str, np.ndarray], int], np.ndarray] = make_features,
    basis_fn: Callable[[np.ndarray], np.ndarray] = default_basis,
) -> Tuple[float, float]:
    ex_idx = sorted(list(spec.exercise_idx))
    n = paths["S"].shape[0]

    # For each path: chosen exercise index and cashflow at that time
    tau_idx = np.full(n, -1, dtype=int)
    cash = np.zeros(n, dtype=float)

    def df(t0, t1):
        return model.df(float(t0), float(t1))

    # Initialize at the last exercise date
    k_last = ex_idx[-1]
    imm_last = np.asarray(spec.immediate_payoff(paths, k_last), dtype=float)
    take_last = imm_last > 1e-12
    tau_idx[take_last] = k_last
    cash[take_last] = imm_last[take_last]

    # Backward induction for earlier exercise dates
    for k in reversed(ex_idx[:-1]):
        imm = np.asarray(spec.immediate_payoff(paths, k), dtype=float)
        not_done = tau_idx < 0

        # Continuation value target: discount future cash from tau to current k
        Y = np.zeros(n, dtype=float)
        has_future = tau_idx >= 0
        if np.any(has_future):
            t_k = times[k]
            t_tau = times[tau_idx[has_future]]
            Y[has_future] = cash[has_future] * np.array([df(t_k, tti) for tti in t_tau])

        candidates = not_done & (imm > 1e-12)

        if np.any(candidates):
            phi_cand = basis_fn(feature_fn(paths, k))[candidates, :]
            y = Y[candidates]
            if phi_cand.shape[0] >= phi_cand.shape[1] and phi_cand.size > 0:
                beta = _regress(phi_cand, y)
                phi_all = basis_fn(feature_fn(paths, k))
                cont_all = phi_all @ beta
            else:
                cont_all = np.full(n, y.mean() if y.size else 0.0)
        else:
            cont_all = np.zeros(n, dtype=float)

        # Optimal decision at k
        take = candidates & (imm >= cont_all)
        tau_idx[take] = k
        cash[take] = imm[take]
        # others keep their future decision

    # Discount chosen cashflows to t=0
    disc = np.zeros(n, dtype=float)
    valid = tau_idx >= 0
    if np.any(valid):
        disc[valid] = np.array([df(0.0, times[i]) for i in tau_idx[valid]])
    price_paths = disc * cash
    price = float(price_paths.mean())
    se = float(price_paths.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return price, se

