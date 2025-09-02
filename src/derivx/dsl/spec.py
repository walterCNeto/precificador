from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple

from ..curves import PiecewiseFlatCurve
from ..models.gbm import RiskNeutralGBM
from ..engine.montecarlo import MonteCarloEngine
from ..exercise.lsmc import ExerciseSpec
from ..payoffs.core import (
    european_call,
    european_put,
    asian_arith_call,
    up_and_out_call,
    basket_call,
    PF,
    relu,
)


def _build_curve(spec: Dict[str, Any]) -> PiecewiseFlatCurve:
    if "r_curve" in spec:
        rc = spec["r_curve"]
        return PiecewiseFlatCurve(np.array(rc["times"], dtype=float),
                                  np.array(rc["rates"], dtype=float))
    r = float(spec.get("r", 0.0))
    return PiecewiseFlatCurve(np.array([1e-8], dtype=float), np.array([r], dtype=float))


def build_engine_from_spec(spec: Dict[str, Any]) -> Tuple[MonteCarloEngine, np.ndarray, list]:
    model_spec = spec["model"]
    r_curve = _build_curve(model_spec)

    q = model_spec.get("q", 0.0)
    sigma = model_spec.get("sigma", 0.2)
    corr = np.array(model_spec.get("corr", [[1.0]]), dtype=float)

    model = RiskNeutralGBM(r_curve, q_funcs=q, sigma_funcs=sigma, corr=corr)
    eng = MonteCarloEngine(model)

    grid = spec.get("grid", {"T": 1.0, "steps": 64})
    T = float(grid.get("T", 1.0))
    steps = int(grid.get("steps", 64))
    times = np.linspace(0.0, T, steps + 1)

    S0 = spec.get("S0", [100.0])
    return eng, times, S0


def _build_payoff(product: Dict[str, Any]):
    ptype = product.get("type", "european_call")
    if ptype == "european_call":
        return european_call(int(product.get("asset", 0)), float(product["K"]))
    if ptype == "european_put":
        return european_put(int(product.get("asset", 0)), float(product["K"]))
    if ptype == "asian_arith_call":
        return asian_arith_call(int(product.get("asset", 0)), float(product["K"]))
    if ptype == "up_and_out_call":
        return up_and_out_call(int(product.get("asset", 0)), float(product["K"]), float(product["barrier"]))
    if ptype == "basket_call":
        return basket_call(list(product["weights"]), float(product["K"]))
    raise ValueError(f"payoff type nao suportado: {ptype}")


def _build_exercise(product: Dict[str, Any], times: np.ndarray):
    style = product.get("style", "european").lower()
    if style == "european":
        return None

    # american: todas as datas (exceto t=0)
    if style == "american":
        ex_idx = list(range(1, len(times)))
    else:
        # bermudan
        if "exercise_idx" in product:
            ex_idx = list(map(int, product["exercise_idx"]))
        elif "exercise_times" in product:
            # converte tempos para indices com tolerancia
            want = list(map(float, product["exercise_times"]))
            ex_idx = []
            for wt in want:
                idx = int(np.argmin(np.abs(times - wt)))
                if idx not in ex_idx and idx > 0:
                    ex_idx.append(idx)
            ex_idx.sort()
        else:
            freq = int(product.get("exercise_every", 8))
            ex_idx = list(range(freq, len(times), freq))
            if (len(times) - 1) not in ex_idx:
                ex_idx.append(len(times) - 1)

    K = float(product.get("K", 100.0))
    asset = int(product.get("asset", 0))

    def imm_call(paths, k):
        St = PF.at_time(paths, asset, k)
        return relu(St - K)

    def imm_put(paths, k):
        St = PF.at_time(paths, asset, k)
        return relu(K - St)

    ptype = product.get("type", "european_call")
    imm = imm_put if "put" in ptype else imm_call

    return ExerciseSpec(exercise_idx=ex_idx, immediate_payoff=imm)


def price_from_spec(spec: Dict[str, Any]):
    eng, times, S0 = build_engine_from_spec(spec)
    product = spec["product"]
    style = product.get("style", "european").lower()

    if style == "european":
        payoff = _build_payoff(product)
        return eng.price(
            payoff, S0, times,
            n_paths=int(spec.get("n_paths", 100_000)),
            seed=spec.get("seed"),
        )
    else:
        ex = _build_exercise(product, times)
        return eng.price_exercisable(
            ex, S0, times,
            n_paths=int(spec.get("n_paths", 120_000)),
            seed=spec.get("seed"),
        )
