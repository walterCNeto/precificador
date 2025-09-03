import math
from .digitals import Phi

def margrabe_exchange_call(S1, S2, r, q1, q2, sigma1, sigma2, rho, T):
    # sigma_X = sqrt(sig1^2 + sig2^2 - 2 rho sig1 sig2)
    var = sigma1*sigma1 + sigma2*sigma2 - 2.0*rho*sigma1*sigma2
    sigmaX = math.sqrt(max(var, 0.0))
    if T <= 0 or sigmaX <= 0:
        return max(0.0, S1*math.exp(-q1*T) - S2*math.exp(-q2*T))
    sT = sigmaX * math.sqrt(T)
    d1 = (math.log(S1 / S2) + (q2 - q1 + 0.5*sigmaX*sigmaX)*T) / sT
    d2 = d1 - sT
    return S1*math.exp(-q1*T)*Phi(d1) - S2*math.exp(-q2*T)*Phi(d2)
