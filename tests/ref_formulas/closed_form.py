import math
import numpy as np

# -----------------------------
# Helpers numéricos
# -----------------------------
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

def Phi(x: float) -> float:
    # CDF normal padrão
    return 0.5 * (1.0 + math.erf(x / SQRT2))

# -----------------------------
# Black–Scholes vanilla (Haug cap. 1)
# -----------------------------
def bs_call_price(S0, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(0.0, S0 * math.exp(-q*T) - K * math.exp(-r*T))
    sT = sigma * math.sqrt(T)
    d1 = (math.log(S0/K) + (r - q + 0.5 * sigma * sigma) * T) / sT
    d2 = d1 - sT
    return S0 * math.exp(-q*T) * Phi(d1) - K * math.exp(-r*T) * Phi(d2)

def bs_put_price(S0, K, r, q, sigma, T):
    # via paridade
    c = bs_call_price(S0, K, r, q, sigma, T)
    return c - (S0*math.exp(-q*T) - K*math.exp(-r*T))

# -----------------------------
# Asiática aritmética (Lévy, discrete sampling)
# Haug (seção de asiáticas): aproximação lognormal por matching de momentos.
# -----------------------------
def asian_arith_call_levy(S0, K, r, q, sigma, T, n):
    """
    Aproximação de Lévy para a média aritmética de n observações igualmente espaçadas.
    ln(A) ~ N(mu, v), obtidos por momentos exatos de A.
    Preço ≈ e^{-rT} [ E[A 1_{A>K}] - K P(A>K) ] com a suposição lognormal.

    Referência: Haug, Asian Options – Levy Approx.
    """
    # tempos de amostragem: t_i = i*T/n, i=1..n
    t = np.arange(1, n+1, dtype=float) * (T / n)

    # E[S_t] e Cov(S_ti, S_tj) sob Q
    m1 = (S0 / n) * np.sum(np.exp((r - q) * t))

    # Cov(S_i, S_j) = E[S_i S_j] - E[S_i]E[S_j]
    # E[S_i S_j] = S0^2 * exp((r-q)(t_i + t_j)) * exp(sigma^2 * min(t_i, t_j))
    ti = t.reshape(-1, 1)
    tj = t.reshape(1, -1)
    Min = np.minimum(ti, tj)
    ES = S0**2 * np.exp((r - q) * (ti + tj)) * np.exp(sigma*sigma * Min)
    ES_i = S0 * np.exp((r - q) * ti)
    ES_j = S0 * np.exp((r - q) * tj)
    Cov = ES - ES_i @ ES_j

    varA = Cov.sum() / (n*n)
    m2 = varA + m1*m1

    # parâmetros lognormais (mu, v) p/ A
    v = math.log(m2 / (m1*m1))
    if v <= 0:
        # em casos degenerados, volta para BS com pequena vol efetiva
        v = 1e-12
    mu = math.log(m1) - 0.5 * v
    sv = math.sqrt(v)

    d1 = (mu - math.log(K) + v) / sv
    d2 = d1 - sv

    EA_1 = math.exp(mu + 0.5 * v) * Phi(d1)   # E[A 1_{A>K}]
    PK   = Phi(d2)                             # P(A > K)

    price = math.exp(-r * T) * (EA_1 - K * PK)
    return float(price)
