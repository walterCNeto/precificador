import os
import math
import tempfile
import pytest

from derivx.report import plot_report

@pytest.mark.parametrize("ptype", ["european_call", "cash_or_nothing_call"])
def test_plot_report_basic(ptype):
    spec = {
        "engine": "analytic",
        "model": {"r": 0.05, "q": [0.0], "sigma": [0.2], "corr": [[1.0]]},
        "grid": {"T": 1.0},
        "S0": [100.0],
        "product": {"style": "european", "type": ptype, "asset": 0, "K": 100.0, "cash": 1.0}
    }
    with tempfile.TemporaryDirectory() as td:
        out_png = os.path.join(td, f"{ptype}.png")
        rep = plot_report(spec, filename=out_png)
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 0
        assert math.isfinite(rep.price)
        # presença de equação de payoff
        assert "g(" in rep.payoff_equation_tex or "Payoff" in rep.payoff_equation_tex
