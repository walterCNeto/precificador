import math, numpy as np
from derivx import price_from_spec, bs_call_price, PiecewiseFlatCurve

def Phi(x):  # CDF normal padrão via erf (independente de SciPy)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_ref(S0, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(0.0, S0*math.exp(-q*T) - K*math.exp(-r*T))
    d1 = (math.log(S0/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S0*math.exp(-q*T)*Phi(d1) - K*math.exp(-r*T)*Phi(d2)

def approx(a, b, tol_abs=None, tol_rel=None):
    if tol_abs is None and tol_rel is None:
        tol_abs = 1e-8
    if tol_abs is not None and abs(a - b) <= tol_abs:
        return True
    if tol_rel is not None and abs(a - b) <= tol_rel * max(1.0, abs(b)):
        return True
    return False

def banner(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def run():
    total = 0; passed = 0
    def check(cond, msg, extra=""):
        nonlocal total, passed
        total += 1
        if cond:
            passed += 1
            print(f"[PASS] {msg} {extra}")
        else:
            print(f"[FAIL] {msg} {extra}")

    # ---------- 1) Black–Scholes analítico ----------
    banner("1) Black–Scholes analítico")
    S0=100.0; K=100.0; r=0.05; q=0.0; sigma=0.2; T=1.0

    c_ref = bs_call_ref(S0,K,r,q,sigma,T)
    c_lib = bs_call_price(S0,K,r,q,sigma,T)
    check(approx(c_lib, c_ref, tol_abs=1e-10), f"BS call derivx ~= ref", f" | derivx={c_lib:.6f} ref={c_ref:.6f}")

    p_par = c_ref - (S0*math.exp(-q*T) - K*math.exp(-r*T))
    check(5.50 < p_par < 5.65, "Put via paridade dentro de faixa ~5.57", f" | put_paridade={p_par:.6f}")

    # ---------- 2) Curva de desconto ----------
    banner("2) Curva PiecewiseFlat")
    rc = PiecewiseFlatCurve(np.array([1e-8]), np.array([0.05]))
    df01 = rc.df(0.0, 1.0)
    check(abs(df01 - math.exp(-0.05)) < 5e-4, "df(0,1) ~= e^{-0.05}", f" | df={df01:.9f} ref={math.exp(-0.05):.9f}")

    # ---------- 3) MC GBM vs BS ----------
    banner("3) MC (GBM) vs BS — tolerância por erro padrão")
    spec_mc = {"engine":"mc",
               "model":{"name":"gbm","r":r,"q":q,"sigma":sigma},
               "grid":{"T":T,"steps":128},
               "S0":[S0],
               "product":{"style":"european","type":"european_call","asset":0,"K":K},
               "n_paths":80_000,"seed":42}
    pmc, se = price_from_spec(spec_mc)
    tol = max(4.0*se, 0.25)
    check(abs(pmc - c_ref) < tol,
          "MC GBM ~= BS", f"| MC={pmc:.4f} ± {1.96*se:.4f}  BS={c_ref:.4f}  tol={tol:.4f}")

    # ---------- 4) PDE Euro vs BS; Amer >= Euro ----------
    banner("4) PDE — Euro ≈ BS e Amer >= Euro")
    spec_pde_e = {"engine":"pde","model":{"name":"gbm","r":r,"q":q,"sigma":sigma},
                  "grid":{"T":T},"S0":[S0],
                  "product":{"style":"european","type":"european_put","asset":0,"K":K},
                  "NS":800,"NT":800,"Smax_mult":5.0}
    p_pde_e, _ = price_from_spec(spec_pde_e)
    p_bs_put = p_par
    check(abs(p_pde_e - p_bs_put) < 0.08, "PDE Euro put ~= BS put", f"| PDE={p_pde_e:.4f} BS={p_bs_put:.4f}")

    spec_pde_a = {"engine":"pde","model":{"name":"gbm","r":r,"q":q,"sigma":sigma},
                  "grid":{"T":T},"S0":[S0],
                  "product":{"style":"american","type":"european_put","asset":0,"K":K},
                  "NS":800,"NT":800,"Smax_mult":5.0}
    p_pde_a, _ = price_from_spec(spec_pde_a)
    check(p_pde_a + 1e-10 >= p_pde_e, "PDE American >= Euro", f"| Amer={p_pde_a:.4f} Euro={p_pde_e:.4f}")

    # ---------- 5) Barrier e Asian (relações lógicas) ----------
    banner("5) Barrier e Asian — relações")
    base = {"engine":"mc","model":{"name":"gbm","r":r,"q":q,"sigma":sigma},
            "grid":{"T":T,"steps":128},"S0":[S0],"n_paths":80_000,"seed":7}
    vanilla = price_from_spec({**base,"product":{"style":"european","type":"european_call","asset":0,"K":K}})[0]
    uo_130  = price_from_spec({**base,"product":{"style":"european","type":"up_and_out_call","asset":0,"K":K,"barrier":130.0}})[0]
    uo_140  = price_from_spec({**base,"product":{"style":"european","type":"up_and_out_call","asset":0,"K":K,"barrier":140.0}})[0]
    check(0.0 <= uo_130 <= vanilla + 1e-8, "Up&Out <= Vanilla", f"| UO130={uo_130:.4f} Vanilla={vanilla:.4f}")
    check(uo_130 <= uo_140 <= vanilla + 1e-8, "Monotonia na barreira (130<=140)", f"| UO130={uo_130:.4f} UO140={uo_140:.4f}")

    asian = price_from_spec({**base,"product":{"style":"european","type":"asian_arith_call","asset":0,"K":K}})[0]
    check(asian <= vanilla + 1e-8, "Asian aritmética <= Vanilla", f"| Asian={asian:.4f} Vanilla={vanilla:.4f}")

    # ---------- 6) Heston MC vs FFT (tolerância generosa) ----------
    banner("6) Heston — MC vs FFT")
    common = {"name":"heston","r":0.05,"q":0.0,"kappa":1.5,"theta":0.04,"xi":0.5,"rho":-0.7,"v0":0.04}
    spec_fft = {"engine":"fft","model":common,"grid":{"T":1.0},"S0":[100.0],
                "product":{"style":"european","type":"european_call","asset":0,"K":100.0},
                "alpha":1.5,"N":4096,"eta":0.25}
    p_fft,_ = price_from_spec(spec_fft)
    spec_mc_h = {"engine":"mc","model":common,"grid":{"T":1.0,"steps":512},"S0":[100.0],
                 "product":{"style":"european","type":"european_call","asset":0,"K":100.0},
                 "n_paths":120_000,"seed":7}
    p_mc_h, se_h = price_from_spec(spec_mc_h)
    tol_h = max(6.0*se_h, 0.7)
    check(abs(p_mc_h - p_fft) < tol_h, "Heston MC ~= FFT", f"| MC={p_mc_h:.4f} ± {1.96*se_h:.4f} FFT={p_fft:.4f} tol={tol_h:.4f}")

    print(f"\nResumo: {passed}/{total} PASS")

if __name__ == "__main__":
    run()
