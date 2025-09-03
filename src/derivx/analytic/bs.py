from __future__ import annotations
import math
from typing import Tuple


def Phi(x: float) -> float:
    """CDF da Normal padrão."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _safe_pos(x: float, eps: float = 1e-300) -> float:
    """Evita log(0) em cenários patológicos."""
    return x if x > eps else eps


def d1d2(S0: float, K: float, r: float, q: float, sigma: float, T: float) -> Tuple[float, float]:
    """
    Retorna (d1, d2) do Black–Scholes.
    Obs.: se T<=0 ou sigma<=0, devolve (inf, inf) — as funções de preço
    tratam esses casos explicitamente antes de chamar d1d2.
    """
    if T <= 0.0 or sigma <= 0.0:
        return float("inf"), float("inf")
    S = _safe_pos(S0)
    Kp = _safe_pos(K)
    sT = sigma * math.sqrt(T)
    d1 = (math.log(S / Kp) + (r - q + 0.5 * sigma * sigma) * T) / sT
    d2 = d1 - sT
    return d1, d2


def bs_call(S0: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Preço do call europeu (Black–Scholes).
    Casos-limite:
      - T <= 0  -> payoff intrínseco: max(S0 - K, 0)
      - sigma=0 -> subjacente determinístico: max(S0*e^{-qT} - K*e^{-rT}, 0)
    """
    if T <= 0.0:
        return max(S0 - K, 0.0)
    if sigma <= 0.0:
        return max(S0 * math.exp(-q * T) - K * math.exp(-r * T), 0.0)

    d1, d2 = d1d2(S0, K, r, q, sigma, T)
    return S0 * math.exp(-q * T) * Phi(d1) - K * math.exp(-r * T) * Phi(d2)


def bs_put(S0: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Preço do put europeu (Black–Scholes).
    Casos-limite:
      - T <= 0  -> payoff intrínseco: max(K - S0, 0)
      - sigma=0 -> subjacente determinístico: max(K*e^{-rT} - S0*e^{-qT}, 0)
    """
    if T <= 0.0:
        return max(K - S0, 0.0)
    if sigma <= 0.0:
        return max(K * math.exp(-r * T) - S0 * math.exp(-q * T), 0.0)

    d1, d2 = d1d2(S0, K, r, q, sigma, T)
    return K * math.exp(-r * T) * Phi(-d2) - S0 * math.exp(-q * T) * Phi(-d1)
