from __future__ import annotations
import math

# -------- Helpers de Black–Scholes --------
def Phi(x: float) -> float:
    """CDF normal padrão."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def d1d2(S0: float, K: float, r: float, q: float, sigma: float, T: float):
    """
    Retorna (d1, d2). Para T<=0 ou sigma<=0 devolve (inf, inf)
    para evitar divisão por zero; os preços abaixo tratam esse caso.
    """
    if T <= 0 or sigma <= 0:
        return float("inf"), float("inf")
    sT = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / sT
    d2 = d1 - sT
    return d1, d2

# -------- Digitais de Haug (preços fechados) --------
def cash_or_nothing_call(S0, K, r, q, sigma, T, cash: float = 1.0):
    """
    Paga 'cash' se S_T > K (cash-or-nothing call).
    Fórmula: cash * e^{-rT} * Phi(d2)
    """
    if T <= 0 or sigma <= 0:
        # liquidação imediata por limite
        return cash * math.exp(-r * T) * (1.0 if S0 * math.exp(-q * T) > K * math.exp(-r * T) else 0.0)
    _, d2 = d1d2(S0, K, r, q, sigma, T)
    return cash * math.exp(-r * T) * Phi(d2)

def cash_or_nothing_put(S0, K, r, q, sigma, T, cash: float = 1.0):
    """
    Paga 'cash' se S_T < K (cash-or-nothing put).
    Fórmula: cash * e^{-rT} * Phi(-d2)
    """
    if T <= 0 or sigma <= 0:
        return cash * math.exp(-r * T) * (1.0 if S0 * math.exp(-q * T) < K * math.exp(-r * T) else 0.0)
    _, d2 = d1d2(S0, K, r, q, sigma, T)
    return cash * math.exp(-r * T) * Phi(-d2)

def asset_or_nothing_call(S0, K, r, q, sigma, T):
    """
    Paga S_T se S_T > K (asset-or-nothing call).
    Fórmula: S0 * e^{-qT} * Phi(d1)
    """
    if T <= 0 or sigma <= 0:
        return S0 * math.exp(-q * T) * (1.0 if S0 * math.exp(-q * T) > K * math.exp(-r * T) else 0.0)
    d1, _ = d1d2(S0, K, r, q, sigma, T)
    return S0 * math.exp(-q * T) * Phi(d1)

def asset_or_nothing_put(S0, K, r, q, sigma, T):
    """
    Paga S_T se S_T < K (asset-or-nothing put).
    Fórmula: S0 * e^{-qT} * Phi(-d1)
    """
    if T <= 0 or sigma <= 0:
        return S0 * math.exp(-q * T) * (1.0 if S0 * math.exp(-q * T) < K * math.exp(-r * T) else 0.0)
    d1, _ = d1d2(S0, K, r, q, sigma, T)
    return S0 * math.exp(-q * T) * Phi(-d1)
