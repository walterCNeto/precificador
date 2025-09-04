# tests/report/test_plot_report.py
import os
import math
import tempfile
import json
import pytest

# pula os testes se matplotlib não estiver instalado
pytest.importorskip("matplotlib")

from derivx.report import plot_report
from derivx import price_from_spec


# -- helper: salva figura (em tmp ou em _artifacts) e faz checagens básicas
def _save_and_assert_png(spec, name, dpi=120):
    keep = os.environ.get("KEEP_ARTIFACTS") == "1"
    if keep:
        out_dir = os.path.join(os.path.dirname(__file__), "_artifacts")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{name}.png")
    else:
        td = tempfile.mkdtemp()
        out = os.path.join(td, f"{name}.png")

    rep = plot_report(spec, filename=out, dpi=dpi)

    assert os.path.exists(out), "PNG não foi gerado"
    # checar header PNG (8 bytes)
    with open(out, "rb") as fh:
        head = fh.read(8)
    assert head.startswith(b"\x89PNG\r\n\x1a\n")

    assert math.isfinite(rep.price), "Preço não-finito"
    # dump rápido dos inputs (só quando keep=1) para inspeção manual
    if keep:
        meta = {
            "price": rep.price,
            "se": rep.se,
            "inputs": rep.inputs,
            "payoff_equation_tex": rep.payoff_equation_tex,
            "pricing_equation_tex": rep.pricing_equation_tex,
            "figure_path": rep.figure_path,
        }
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    return rep, out


def test_report_vanilla_call_analytic_matches_bs():
    # europeu (BS analítico)
    spec = {
        "engine": "analytic",
        "model": {"r": 0.05, "q": [0.0], "sigma": [0.2], "corr": [[1.0]]},
        "grid": {"T": 1.0},
        "S0": [100.0],
        "product": {"style": "european", "type": "european_call", "asset": 0, "K": 100.0},
    }
    rep, out_png = _save_and_assert_png(spec, "vanilla_call_analytic")

    # compara com o próprio price_from_spec (mesmo caminho analítico)
    p_ref, se_ref = price_from_spec(spec)
    assert abs(rep.price - p_ref) < 1e-12
    assert se_ref == 0.0
    assert "BS" in (rep.pricing_equation_tex or "")


def test_report_digital_cash_call_analytic():
    spec = {
        "engine": "analytic",
        "model": {"r": 0.05, "q": [0.0], "sigma": [0.2], "corr": [[1.0]]},
        "grid": {"T": 1.0},
        "S0": [100.0],
        "product": {
            "style": "european",
            "type": "cash_or_nothing_call",
            "asset": 0,
            "K": 100.0,
            "cash": 1.0,
        },
    }
    rep, out_png = _save_and_assert_png(spec, "digital_cash_call")
    assert "Haug" in (rep.pricing_equation_tex or "")
    # payoff é em degrau => traçado steps-post (não assertamos visualmente; só smoke test)


def test_report_exchange_margrabe_analytic():
    spec = {
        "engine": "analytic",
        "model": {
            "r": 0.05, "q": [0.01, 0.03], "sigma": [0.20, 0.30],
            "corr": [[1.0, 0.5], [0.5, 1.0]],
        },
        "grid": {"T": 1.0},
        "S0": [100.0, 120.0],
        "product": {"style": "european", "type": "exchange_call", "asset_long": 0, "asset_short": 1},
    }
    rep, out_png = _save_and_assert_png(spec, "exchange_margrabe")
    assert "Margrabe" in (rep.pricing_equation_tex or "")
    # variar S1 mantendo S2 => vary_asset deve ser 0 ou None (o report infere 0)
    assert rep.vary_asset in (0, None)


@pytest.mark.slow
def test_report_asian_mc_vs_vanilla_bound():
    # MC: asiática <= vanilla (em média). Usamos tolerância baseada no SE.
    base_model = {"name": "gbm", "r": 0.05, "q": 0.0, "sigma": 0.2}
    base_grid  = {"T": 1.0, "steps": 128}
    n_paths = 20_000  # mantenha razoável para o CI; aumenta localmente se quiser ver mais liso

    asian = {
        "engine": "mc",
        "model": base_model,
        "grid": base_grid,
        "S0": [100.0],
        "product": {"style": "european", "type": "asian_arith_call", "asset": 0, "K": 100.0},
        "n_paths": n_paths, "seed": 7,
    }
    vanilla = {
        "engine": "mc",
        "model": base_model,
        "grid": base_grid,
        "S0": [100.0],
        "product": {"style": "european", "type": "european_call", "asset": 0, "K": 100.0},
        "n_paths": n_paths, "seed": 7,
    }

    rep_asian, _ = _save_and_assert_png(asian, "asian_mc")
    p_van, se_van = price_from_spec(vanilla)

    # bound frouxo com tolerância por erro-padrão (amigável a flutuações)
    tol = 3.0 * max(rep_asian.se, se_van)
    assert rep_asian.price <= p_van + tol


def test_report_ir_cap_smoke():
    # Report para IR (cap) – não plota 1D, mas gera PNG com equações/inputs
    spec = {
        "engine": "analytic",
        "model": {"r": 0.05},  # curva flat por simplicidade
        "grid": {"T": 2.0},
        "S0": [],
        "product": {
            "style": "european",
            "type": "cap",
            "payment_times": [0.5, 1.0, 1.5, 2.0],
            "tau": 0.5,
            "K": 0.04,
            "sigma": 0.20,
            # reset_times será inferido como T_i - tau
            "notional": 1.0,
        },
    }
    rep, out_png = _save_and_assert_png(spec, "cap_ir")
    assert rep.vary_asset is None
    assert "Cap" in (rep.pricing_equation_tex or "")
