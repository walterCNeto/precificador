from __future__ import annotations
import math
from typing import Optional
from .bs import Phi, d1d2


def _deterministic_ST(S0: float, r: float, q: float, T: float) -> float:
    """S_T determinístico quando sigma=0: S0 * exp((r-q)T)."""
    return S0 * math.exp((r - q) * T)


# --- Digitais (cash- / asset-or-nothing) -------------------------------------

def cash_or_nothing_call(
    S0: float, K: float, r: float, q: float, sigma: float, T: float, cash: float = 1.0
) -> float:
    """
    Paga 'cash' se S_T > K. Preço: cash * e^{-rT} * Phi(d2).
    Casos-limite:
      - T<=0:  cash * 1_{S0 > K}
      - sigma=0: cash * e^{-rT} * 1_{S_T > K} com S_T determinístico
    """
    if T <= 0.0:
        return cash * (1.0 if S0 > K else 0.0)
    if sigma <= 0.0:
        ST = _deterministic_ST(S0, r, q, T)
        return cash * math.exp(-r * T) * (1.0 if ST > K else 0.0)
    _, d2 = d1d2(S0, K, r, q, sigma, T)
    return cash * math.exp(-r * T) * Phi(d2)


def cash_or_nothing_put(
    S0: float, K: float, r: float, q: float, sigma: float, T: float, cash: float = 1.0
) -> float:
    """
    Paga 'cash' se S_T < K. Preço: cash * e^{-rT} * Phi(-d2).
    Casos-limite análogos ao call.
    """
    if T <= 0.0:
        return cash * (1.0 if S0 < K else 0.0)
    if sigma <= 0.0:
        ST = _deterministic_ST(S0, r, q, T)
        return cash * math.exp(-r * T) * (1.0 if ST < K else 0.0)
    _, d2 = d1d2(S0, K, r, q, sigma, T)
    return cash * math.exp(-r * T) * Phi(-d2)


def asset_or_nothing_call(S0: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Paga S_T se S_T > K. Preço fechado: S0 * e^{-qT} * Phi(d1).
    Casos-limite:
      - T<=0:  S0 * 1_{S0 > K}
      - sigma=0: e^{-rT} * S_T * 1_{S_T > K}
    """
    if T <= 0.0:
        return S0 * (1.0 if S0 > K else 0.0)
    if sigma <= 0.0:
        ST = _deterministic_ST(S0, r, q, T)
        return math.exp(-r * T) * ST * (1.0 if ST > K else 0.0)
    d1, _ = d1d2(S0, K, r, q, sigma, T)
    return S0 * math.exp(-q * T) * Phi(d1)


def asset_or_nothing_put(S0: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Paga S_T se S_T < K. Preço fechado: S0 * e^{-qT} * Phi(-d1).
    Casos-limite análogos ao call.
    """
    if T <= 0.0:
        return S0 * (1.0 if S0 < K else 0.0)
    if sigma <= 0.0:
        ST = _deterministic_ST(S0, r, q, T)
        return math.exp(-r * T) * ST * (1.0 if ST < K else 0.0)
    d1, _ = d1d2(S0, K, r, q, sigma, T)
    return S0 * math.exp(-q * T) * Phi(-d1)


# --- Gap options (gatilho K2, pagamento com strike K1) -----------------------

def gap_call(S0: float, K1: float, K2: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Payoff: 1_{S_T > K2} * (S_T - K1).
    Fórmula fechada (Haug): S0 e^{-qT} Phi(d1(K2)) - K1 e^{-rT} Phi(d2(K2)).
    Casos-limite:
      - T<=0:  max(S0 - K1, 0) se S0 > K2, senão 0
      - sigma=0: e^{-rT} * max(S_T - K1, 0) * 1_{S_T > K2}
    """
    if T <= 0.0:
        return max(S0 - K1, 0.0) if S0 > K2 else 0.0
    if sigma <= 0.0:
        ST = _deterministic_ST(S0, r, q, T)
        return math.exp(-r * T) * (max(ST - K1, 0.0) if ST > K2 else 0.0)
    d1, d2 = d1d2(S0, K2, r, q, sigma, T)
    return S0 * math.exp(-q * T) * Phi(d1) - K1 * math.exp(-r * T) * Phi(d2)


def gap_put(S0: float, K1: float, K2: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Payoff: 1_{S_T < K2} * (K1 - S_T).
    Fórmula fechada (Haug): K1 e^{-rT} Phi(-d2(K2)) - S0 e^{-qT} Phi(-d1(K2)).
    Casos-limite análogos.
    """
    if T <= 0.0:
        return (K1 - S0) if S0 < K2 else 0.0
    if sigma <= 0.0:
        ST = _deterministic_ST(S0, r, q, T)
        return math.exp(-r * T) * ((K1 - ST) if ST < K2 else 0.0)
    d1, d2 = d1d2(S0, K2, r, q, sigma, T)
    return K1 * math.exp(-r * T) * Phi(-d2) - S0 * math.exp(-q * T) * Phi(-d1)


# --- Margrabe (exchange option) ---------------------------------------------

def margrabe_exchange_call(
    S1: float, S2: float, r: float, q1: float, q2: float,
    sigma1: float, sigma2: float, rho: float, T: float
) -> float:
    """
    Call para trocar S2 por S1 (strike = 1 * S2). Preço (Margrabe, 1978):
      C = S1 e^{-q1 T} Phi(d1) - S2 e^{-q2 T} Phi(d2),
      com sigma_ex^2 = sigma1^2 + sigma2^2 - 2 rho sigma1 sigma2,
      d1 = [ln(S1/S2) + (q2 - q1 + 0.5 sigma_ex^2) T]/(sigma_ex sqrt(T)),
      d2 = d1 - sigma_ex sqrt(T).
    Casos-limite:
      - T<=0 ou sigma_ex<=0: max(S1 e^{-q1 T} - S2 e^{-q2 T}, 0).
    """
    s_ex2 = sigma1 * sigma1 + sigma2 * sigma2 - 2.0 * rho * sigma1 * sigma2
    # devido a arredondamento numérico, pode ficar levemente negativo:
    s_ex = math.sqrt(max(0.0, s_ex2))
    if T <= 0.0 or s_ex <= 0.0:
        return max(S1 * math.exp(-q1 * T) - S2 * math.exp(-q2 * T), 0.0)

    sT = s_ex * math.sqrt(T)
    mu = (math.log(S1 / S2) + (q2 - q1 + 0.5 * s_ex * s_ex) * T) / sT
    d1 = mu
    d2 = d1 - sT
    return S1 * math.exp(-q1 * T) * Phi(d1) - S2 * math.exp(-q2 * T) * Phi(d2)
