from math import exp
from derivx import bs_call_price

def test_put_call_parity_atm():
    S0=K=100.0; r=0.05; q=0.0; sig=0.2; T=1.0
    c = bs_call_price(S0,K,r,q,sig,T)
    p = c - (S0*exp(-q*T) - K*exp(-r*T))
    assert 5.50 < p < 5.65  # ~5.5735
