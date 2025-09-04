# src/derivx/dsl/spec.py
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple

from ..payoffs.extra import build_extra_payoff
from ..curves import PiecewiseFlatCurve
from ..models.gbm import RiskNeutralGBM
from ..engine.montecarlo import MonteCarloEngine
from ..exercise.lsmc import ExerciseSpec
from ..ir.black76 import (
    fra_pv, swap_par_rate, swap_pv,
    cap_price, floor_price, payer_swaption_price, receiver_swaption_price,
)
from ..payoffs.core import (
    european_call,
    european_put,
    asian_arith_call,
    up_and_out_call,
    basket_call,
    PF,
    relu,
)

# === Importes analíticos (equity — fórmulas fechadas) ===
from ..analytic import (
    bs_call, bs_put,
    cash_or_nothing_call, cash_or_nothing_put,
    asset_or_nothing_call, asset_or_nothing_put,
    gap_call as an_gap_call, gap_put as an_gap_put,
    margrabe_exchange_call,
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
    """
    Constrói o payoff para estilos europeus.
    Primeiro tenta os 'extras' (digitais/gap/exchange, etc).
    Se não reconhecer, cai no conjunto 'core' padrão.
    """
    # 1) tenta construir via payoffs extras (digitais/gap/exchange em MC)
    extra = build_extra_payoff(product)
    if extra is not None:
        return extra

    # 2) fallback: payoffs core
    ptype = product.get("type", "european_call")
    if ptype == "european_call":
        return european_call(int(product.get("asset", 0)), float(product["K"]))
    if ptype == "european_put":
        return european_put(int(product.get("asset", 0)), float(product["K"]))
    if ptype == "asian_arith_call":
        return asian_arith_call(int(product.get("asset", 0)), float(product["K"]))
    if ptype == "up_and_out_call":
        return up_and_out_call(
            int(product.get("asset", 0)),
            float(product["K"]),
            float(product["barrier"]),
        )
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

    ptype = product.get("type", "european_call").lower()
    imm = imm_put if "put" in ptype else imm_call

    return ExerciseSpec(exercise_idx=ex_idx, immediate_payoff=imm)


# =========================
#   Analítico — JUROS (IR)
# =========================
def _analytic_ir_price(model_spec: Dict[str, Any], grid: Dict[str, Any], product: Dict[str, Any]) -> Tuple[float, float]:
    """
    Fechadas de IR (single-curve, Black-76): zcb, fra, swap, cap, floor, payer/receiver swaption.
    """
    r_curve = _build_curve(model_spec)

    def D(t: float) -> float:
        return float(r_curve.df(0.0, float(t)))

    ptype = product.get("type", "").lower()
    notional = float(product.get("notional", 1.0))

    if ptype == "zcb":
        T = float(product["T"])
        return notional * D(T), 0.0

    if ptype == "fra":
        T1 = float(product["T1"]); T2 = float(product["T2"])
        tau = float(product["tau"]); K = float(product["K"])
        pv = fra_pv(D(T1), D(T2), tau, K, notional)
        return pv, 0.0

    if ptype == "swap":
        T0 = float(product.get("T0", 0.0))
        pay_times = [float(x) for x in product["payment_times"]]
        tau = product.get("tau", None)
        if isinstance(tau, (list, tuple, np.ndarray)):
            taus = [float(x) for x in tau]
        else:
            prevs = [T0] + pay_times[:-1]
            taus = [pt - pr for pt, pr in zip(pay_times, prevs)]
        D0_Ts = [D(t) for t in pay_times]
        if product.get("par", False) and ("fixed_rate" not in product):
            # swap par → PV ~ 0
            return 0.0, 0.0
        K = float(product["fixed_rate"])
        pv = swap_pv(K, D(T0), D0_Ts, taus, notional)
        return pv, 0.0

    if ptype in ("cap", "floor"):
        pay_times = [float(x) for x in product["payment_times"]]
        tau = product.get("tau", None)
        if isinstance(tau, (list, tuple, np.ndarray)):
            taus = [float(x) for x in tau]
        else:
            prevs = [product.get("start", 0.0)] + pay_times[:-1]
            taus = [pt - pr for pt, pr in zip(pay_times, prevs)]
        if "reset_times" in product:
            reset_times = [float(x) for x in product["reset_times"]]
        else:
            reset_times = [pt - ta for pt, ta in zip(pay_times, taus)]
        D0_Tip1 = [D(t) for t in pay_times]
        D0_Ti   = [D(t) for t in reset_times]
        K = float(product["K"])
        sigma = float(product["sigma"])
        if ptype == "cap":
            pv = cap_price(D0_Ti, D0_Tip1, taus, K, sigma, reset_times, notional)
        else:
            pv = floor_price(D0_Ti, D0_Tip1, taus, K, sigma, reset_times, notional)
        return pv, 0.0

    if ptype in ("payer_swaption", "receiver_swaption"):
        T0 = float(product["expiry"])
        pay_times = [float(x) for x in product["payment_times"]]
        tau = product.get("tau", None)
        if isinstance(tau, (list, tuple, np.ndarray)):
            taus = [float(x) for x in tau]
        else:
            prevs = [T0] + pay_times[:-1]
            taus = [pt - pr for pt, pr in zip(pay_times, prevs)]
        D0_Ts = [D(t) for t in pay_times]
        K = float(product["K"]); sigma = float(product["sigma"])
        if ptype == "payer_swaption":
            pv = payer_swaption_price(D(T0), D0_Ts, taus, K, sigma, T0, notional)
        else:
            pv = receiver_swaption_price(D(T0), D0_Ts, taus, K, sigma, T0, notional)
        return pv, 0.0

    raise ValueError(f"produto analitico IR nao suportado: {ptype}")


# ==========================================
#   Analítico — EQUITY (BS/Haug/Margrabe)
# ==========================================
def _to_scalar_or_list(x, i: int) -> float:
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(x[i])
    return float(x)

def _price_analytic(spec: Dict[str, Any]) -> Tuple[float, float] | None:
    product = spec["product"]
    model = spec["model"]
    grid = spec.get("grid", {"T": 1.0})
    T = float(grid.get("T", 1.0))
    S0 = spec.get("S0", [100.0])

    r = float(model.get("r", 0.0))
    q = model.get("q", 0.0)
    sigma = model.get("sigma", 0.2)
    corr = np.array(model.get("corr", [[1.0]]), dtype=float)

    ptype = str(product.get("type", "")).lower()
    style = str(product.get("style", "european")).lower()
    if style != "european":
        return None  # analítico apenas para europeias aqui

    # BS vanilla
    if ptype == "european_call":
        a = int(product.get("asset", 0))
        K = float(product["K"])
        return bs_call(S0[a], K, r, _to_scalar_or_list(q, a), _to_scalar_or_list(sigma, a), T), 0.0
    if ptype == "european_put":
        a = int(product.get("asset", 0))
        K = float(product["K"])
        return bs_put(S0[a], K, r, _to_scalar_or_list(q, a), _to_scalar_or_list(sigma, a), T), 0.0

    # Digitais (Haug)
    if ptype in ("cash_or_nothing_call", "digital_cash_call", "binary_cash_call"):
        a = int(product.get("asset", 0)); K = float(product["K"]); cash = float(product.get("cash", 1.0))
        return cash_or_nothing_call(S0[a], K, r, _to_scalar_or_list(q, a), _to_scalar_or_list(sigma, a), T, cash), 0.0
    if ptype in ("cash_or_nothing_put", "digital_cash_put", "binary_cash_put"):
        a = int(product.get("asset", 0)); K = float(product["K"]); cash = float(product.get("cash", 1.0))
        return cash_or_nothing_put (S0[a], K, r, _to_scalar_or_list(q, a), _to_scalar_or_list(sigma, a), T, cash), 0.0
    if ptype in ("asset_or_nothing_call", "digital_asset_call"):
        a = int(product.get("asset", 0)); K = float(product["K"])
        return asset_or_nothing_call(S0[a], K, r, _to_scalar_or_list(q, a), _to_scalar_or_list(sigma, a), T), 0.0
    if ptype in ("asset_or_nothing_put", "digital_asset_put"):
        a = int(product.get("asset", 0)); K = float(product["K"])
        return asset_or_nothing_put (S0[a], K, r, _to_scalar_or_list(q, a), _to_scalar_or_list(sigma, a), T), 0.0

    # Gap
    if ptype == "gap_call":
        a = int(product.get("asset", 0)); K1 = float(product.get("K1", product.get("payoff_strike")))
        K2 = float(product.get("K2", product.get("trigger")))
        return an_gap_call(S0[a], K1, K2, r, _to_scalar_or_list(q, a), _to_scalar_or_list(sigma, a), T), 0.0
    if ptype == "gap_put":
        a = int(product.get("asset", 0)); K1 = float(product.get("K1", product.get("payoff_strike")))
        K2 = float(product.get("K2", product.get("trigger")))
        return an_gap_put (S0[a], K1, K2, r, _to_scalar_or_list(q, a), _to_scalar_or_list(sigma, a), T), 0.0

    # Margrabe (exchange)
    if ptype in ("margrabe_exchange_call", "exchange_call"):
        a_long  = int(product.get("asset1", product.get("asset_long", 0)))
        a_short = int(product.get("asset2", product.get("asset_short", 1)))
        q1 = _to_scalar_or_list(q, a_long);  q2 = _to_scalar_or_list(q, a_short)
        s1 = _to_scalar_or_list(sigma, a_long); s2 = _to_scalar_or_list(sigma, a_short)
        rho = float(corr[a_long, a_short])
        return margrabe_exchange_call(S0[a_long], S0[a_short], r, q1, q2, s1, s2, rho, T), 0.0

    return None


def price_from_spec(spec: Dict[str, Any]):
    """
    Roteia a DSL para:
      - engine 'analytic'/'auto': tenta primeiro JUROS (IR) e depois Equity (BS/Haug/Margrabe),
      - caso não aplicável, cai para MC/LSMC (GBM).
    """
    engine = str(spec.get("engine", "mc")).lower()

    if engine in ("analytic", "auto"):
        # 1) tenta analítico de IR (FRA/swap/cap/floor/swaption) primeiro
        try:
            return _analytic_ir_price(spec["model"], spec.get("grid", {}), spec["product"])
        except Exception:
            # não parece ser IR ou faltou algum parâmetro — seguimos
            pass

        # 2) tenta analítico de equity (BS/Haug/Margrabe)
        out = _price_analytic(spec)
        if out is not None:
            return out

        # 3) se era estritamente 'analytic' e nada serviu, avisa
        if engine == "analytic":
            raise ValueError("engine='analytic' não suporta este payoff/modelo. Tente 'mc' (ou 'pde'/'fft' se disponível).")

    # === MC/LSMC (default) ===
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
