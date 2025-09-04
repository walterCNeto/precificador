# src/derivx/ir/black76.py
from __future__ import annotations
import math
from typing import Sequence, Tuple

def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _d1d2(F: float, K: float, sigma: float, T: float) -> Tuple[float, float]:
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        inf = float("inf")
        return inf, inf
    sT = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sT * sT) / sT
    d2 = d1 - sT
    return d1, d2

# ---------------------------------------------------------------------
# Black-76 em termos do forward F (sem desconto). O DF entra fora.
# ---------------------------------------------------------------------
def black_call_forward(F: float, K: float, sigma: float, T: float) -> float:
    d1, d2 = _d1d2(F, K, sigma, T)
    if math.isinf(d1):
        return max(F - K, 0.0)
    return F * Phi(d1) - K * Phi(d2)

def black_put_forward(F: float, K: float, sigma: float, T: float) -> float:
    d1, d2 = _d1d2(F, K, sigma, T)
    if math.isinf(d1):
        return max(K - F, 0.0)
    return K * Phi(-d2) - F * Phi(-d1)

# ---------------------------------------------------------------------
# FRA e Swap (single-curve, accrual simples τ)
# ---------------------------------------------------------------------
def fra_forward_rate(D0_T1: float, D0_T2: float, tau: float) -> float:
    """F = (1/τ) * (D(T1)/D(T2) - 1)."""
    return (D0_T1 / D0_T2 - 1.0) / tau

def fra_pv(D0_T1: float, D0_T2: float, tau: float, K: float, notional: float = 1.0) -> float:
    """
    PV de um FRA que liquida em T2: PV = N * D(0,T2) * τ * (F - K),
    sinal 'comprado F' (recebe F, paga K).
    """
    F = fra_forward_rate(D0_T1, D0_T2, tau)
    return notional * D0_T2 * tau * (F - K)

def swap_par_rate(D0_T0: float, D0_Ts: Sequence[float], taus: Sequence[float]) -> float:
    """
    Par rate: S* = (D(0,T0) - D(0,Tn)) / Σ τ_j D(0,T_j), j=1..n.
    """
    if len(D0_Ts) != len(taus):
        raise ValueError("D0_Ts e taus devem ter mesmo tamanho.")
    A = sum(tau * D for tau, D in zip(taus, D0_Ts))
    if A <= 0:
        raise ValueError("Anuidade inválida (A <= 0).")
    return (D0_T0 - D0_Ts[-1]) / A

def swap_pv(fixed_rate: float, D0_T0: float, D0_Ts: Sequence[float], taus: Sequence[float], notional: float = 1.0) -> float:
    """
    PV (payer fixed) = N * [ (D(0,T0) - D(0,Tn)) - K * Σ τ_j D(0,T_j) ].
    """
    if len(D0_Ts) != len(taus):
        raise ValueError("D0_Ts e taus devem ter mesmo tamanho.")
    A = sum(tau * D for tau, D in zip(taus, D0_Ts))
    float_leg = D0_T0 - D0_Ts[-1]
    fixed_leg = fixed_rate * A
    return notional * (float_leg - fixed_leg)

# ---------------------------------------------------------------------
# Caplets/Floorlets e Cap/Floor (soma de caplets/floorlets)
# caplet: opção sobre L(T_i,T_{i+1}), pago em T_{i+1}
# preço = D(0,T_{i+1}) * τ * Black(F,K,σ, T_i), com F forward simples.
# ---------------------------------------------------------------------
def caplet_price(D0_Ti: float, D0_Tip1: float, tau: float, K: float, sigma: float, T_exp: float, notional: float = 1.0) -> float:
    F = fra_forward_rate(D0_Ti, D0_Tip1, tau)
    core = black_call_forward(F, K, sigma, T_exp)
    return notional * D0_Tip1 * tau * core

def floorlet_price(D0_Ti: float, D0_Tip1: float, tau: float, K: float, sigma: float, T_exp: float, notional: float = 1.0) -> float:
    F = fra_forward_rate(D0_Ti, D0_Tip1, tau)
    core = black_put_forward(F, K, sigma, T_exp)
    return notional * D0_Tip1 * tau * core

def cap_price(D0_Ti: Sequence[float], D0_Tip1: Sequence[float], taus: Sequence[float], K: float, sigma: float, reset_times: Sequence[float], notional: float = 1.0) -> float:
    if not (len(D0_Ti) == len(D0_Tip1) == len(taus) == len(reset_times)):
        raise ValueError("Tamanhos inconsistentes (cap).")
    pv = 0.0
    for i in range(len(taus)):
        pv += caplet_price(D0_Ti[i], D0_Tip1[i], taus[i], K, sigma, reset_times[i], notional)
    return pv

def floor_price(D0_Ti: Sequence[float], D0_Tip1: Sequence[float], taus: Sequence[float], K: float, sigma: float, reset_times: Sequence[float], notional: float = 1.0) -> float:
    if not (len(D0_Ti) == len(D0_Tip1) == len(taus) == len(reset_times)):
        raise ValueError("Tamanhos inconsistentes (floor).")
    pv = 0.0
    for i in range(len(taus)):
        pv += floorlet_price(D0_Ti[i], D0_Tip1[i], taus[i], K, sigma, reset_times[i], notional)
    return pv

# ---------------------------------------------------------------------
# Swaption (Black-76) sobre a taxa do swap-forward S
# Payer:  A * Black_call(S,K,σ,T0)
# Receiver: A * Black_put(S,K,σ,T0)
# onde A = Σ τ_j D(0,T_j), S = (D(0,T0)-D(0,Tn))/A
# ---------------------------------------------------------------------
def _annuity(D0_Ts: Sequence[float], taus: Sequence[float]) -> float:
    if len(D0_Ts) != len(taus):
        raise ValueError("D0_Ts e taus devem ter mesmo tamanho.")
    return sum(tau * D for tau, D in zip(taus, D0_Ts))

def payer_swaption_price(D0_T0: float, D0_Ts: Sequence[float], taus: Sequence[float], K: float, sigma: float, T0: float, notional: float = 1.0) -> float:
    A = _annuity(D0_Ts, taus)
    S = swap_par_rate(D0_T0, D0_Ts, taus)
    core = black_call_forward(S, K, sigma, T0)
    return notional * A * core

def receiver_swaption_price(D0_T0: float, D0_Ts: Sequence[float], taus: Sequence[float], K: float, sigma: float, T0: float, notional: float = 1.0) -> float:
    A = _annuity(D0_Ts, taus)
    S = swap_par_rate(D0_T0, D0_Ts, taus)
    core = black_put_forward(S, K, sigma, T0)
    return notional * A * core
