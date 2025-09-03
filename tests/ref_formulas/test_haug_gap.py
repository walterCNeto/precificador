from math import isfinite
from derivx import price_from_spec
from tests.ref_formulas.gap import gap_call_ref, gap_put_ref

def _spec_gap(ptype, K1, K2):
    return {
      "engine":"analytic",
      "model":{"name":"gbm","r":0.05,"q":[0.0],"sigma":[0.2],"corr":[[1.0]]},
      "grid":{"T":1.0,"steps":16},
      "S0":[100.0],
      "product":{"style":"european","type":ptype,"asset":0,"K1":K1,"K2":K2}
    }

def test_gap_call_put_match_refs():
    S0=100.0; r=0.05; q=0.0; sig=0.2; T=1.0
    K1=95.0; K2=100.0
    ref_c = gap_call_ref(S0,K1,K2,r,q,sig,T)
    ref_p = gap_put_ref (S0,K1,K2,r,q,sig,T)
    p_c,_ = price_from_spec(_spec_gap("gap_call",K1,K2))
    p_p,_ = price_from_spec(_spec_gap("gap_put" ,K1,K2))
    assert isfinite(p_c) and isfinite(p_p)
    assert abs(p_c-ref_c) < 1e-10
    assert abs(p_p-ref_p) < 1e-10
