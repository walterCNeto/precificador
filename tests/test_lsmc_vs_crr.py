# tests/test_lsmc_vs_crr.py
import math
import numpy as np
from derivx import price_from_spec

def crr_bermudan_put(S0,K,r,q,sigma,T,N,exercise_every):
    dt = T/N; u = math.exp(sigma*math.sqrt(dt)); d = 1.0/u
    a  = math.exp((r-q)*dt); p = (a-d)/(u-d); disc = math.exp(-r*dt)
    V  = [max(K - S0*(u**j)*(d**(N-j)), 0.0) for j in range(N+1)]
    ex_idx = set(range(0, N, exercise_every))
    for i in range(N-1, -1, -1):
        Vn = []
        for j in range(i+1):
            S    = S0*(u**j)*(d**(i-j))
            cont = disc*(p*V[j+1] + (1-p)*V[j])
            Vn.append(max(K - S, cont) if i in ex_idx else cont)
        V = Vn
    return V[0]

def test_lsmc_bermudan_close_to_crr():
    S0=K=100.0; r=0.05; q=0.0; sig=0.2; T=1.0
    # CRR referências:
    ref = {ex: crr_bermudan_put(S0,K,r,q,sig,T,N=256,exercise_every=ex) for ex in (8,16,32)}
    # LSMC:
    for ex in (8,16,32):
        spec={"engine":"mc","model":{"name":"gbm","r":r,"q":q,"sigma":sig},
              "grid":{"T":T,"steps":256},"S0":[S0],
              "product":{"style":"bermudan","type":"european_put","asset":0,"K":K,"exercise_every":ex},
              "n_paths":120_000,"seed":7}
        p,se = price_from_spec(spec)
        # tolerância: 3*SE (ruído) + 0.03 (folga de regressão)
        tol = 3.0*se + 0.03
        assert abs(p - ref[ex]) < tol
        # e sempre abaixo do Americano CRR (upper bound prático)
    amer = crr_bermudan_put(S0,K,r,q,sig,T,N=1024,exercise_every=1)
    assert max(ref.values()) <= amer + 1e-6
