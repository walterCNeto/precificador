import math
from .digitals import Phi, d1d2  # usa helpers que você já tem aí

def gap_call_ref(S0,K1,K2,r,q,sigma,T):
    d1,d2 = d1d2(S0,K2,r,q,sigma,T)
    return S0*math.exp(-q*T)*Phi(d1) - K1*math.exp(-r*T)*Phi(d2)

def gap_put_ref(S0,K1,K2,r,q,sigma,T):
    d1,d2 = d1d2(S0,K2,r,q,sigma,T)
    return K1*math.exp(-r*T)*Phi(-d2) - S0*math.exp(-q*T)*Phi(-d1)
