import math

SQRT2 = math.sqrt(2.0)

def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def d1_d2(S0,K,r,q,sigma,T):
    sT = sigma * math.sqrt(T)
    if T<=0 or sigma<=0:
        return float('inf'), float('inf')
    d1 = (math.log(S0/K) + (r - q + 0.5*sigma*sigma)*T)/sT
    d2 = d1 - sT
    return d1, d2

# Cash-or-nothing
def cash_or_nothing_call(S0,K,r,q,sigma,T, cash=1.0):
    _, d2 = d1_d2(S0,K,r,q,sigma,T)
    return cash * math.exp(-r*T) * Phi(d2)

def cash_or_nothing_put(S0,K,r,q,sigma,T, cash=1.0):
    _, d2 = d1_d2(S0,K,r,q,sigma,T)
    return cash * math.exp(-r*T) * Phi(-d2)

# Asset-or-nothing
def asset_or_nothing_call(S0,K,r,q,sigma,T):
    d1, _ = d1_d2(S0,K,r,q,sigma,T)
    return S0 * math.exp(-q*T) * Phi(d1)

def asset_or_nothing_put(S0,K,r,q,sigma,T):
    d1, _ = d1_d2(S0,K,r,q,sigma,T)
    return S0 * math.exp(-q*T) * Phi(-d1)
