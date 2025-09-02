from derivx import price_from_spec, bs_call_price

def _base_spec():
    return {
        "model": {"r": 0.05, "q": [0.00], "sigma": [0.20], "corr": [[1.0]]},
        "grid": {"T": 1.0, "steps": 64},
        "n_paths": 25_000,
        "seed": 42,
    }

def test_monotonicidade_em_S0_call():
    spec = _base_spec()
    spec["product"] = {"style": "european", "type": "european_call", "asset": 0, "K": 100.0}
    # S0 menor
    spec["S0"] = [90.0]
    p_low, _ = price_from_spec(spec)
    # S0 maior
    spec["S0"] = [110.0]
    p_high, _ = price_from_spec(spec)
    assert p_high > p_low

def test_barreira_menor_que_vanilla():
    base = _base_spec()
    base["S0"] = [100.0]
    vanilla = base.copy()
    vanilla["product"] = {"style": "european", "type": "european_call", "asset": 0, "K": 100.0}
    p_van, _ = price_from_spec(vanilla)

    barrier = base.copy()
    barrier["product"] = {"style": "european", "type": "up_and_out_call", "asset": 0, "K": 100.0, "barrier": 130.0}
    p_bar, _ = price_from_spec(barrier)

    assert p_bar <= p_van + 1e-9  # nunca maior que a vanilla

def test_bermudana_put_maior_que_europeia_put():
    base = _base_spec()
    base["S0"] = [100.0]
    euro = base.copy()
    euro["product"] = {"style": "european", "type": "european_put", "asset": 0, "K": 100.0}
    p_euro, _ = price_from_spec(euro)

    berm = base.copy()
    berm["product"] = {
        "style": "bermudan",  # qualquer coisa != european vira LSMC
        "type": "european_put",
        "asset": 0,
        "K": 100.0,
        "exercise_every": 16,  # ~trimestral
    }
    p_berm, _ = price_from_spec(berm)
    assert p_berm >= p_euro - 1e-9

def test_bs_consistencia_basica():
    spec = _base_spec()
    spec["S0"] = [100.0]
    spec["product"] = {"style": "european", "type": "european_call", "asset": 0, "K": 100.0}
    mc, se = price_from_spec(spec)
    bs = bs_call_price(100.0, 100.0, r=0.05, q=0.0, sigma=0.2, T=1.0)
    # tolerância larga porque é MC com 25k paths
    assert abs(mc - bs) < 0.8

def test_se_diminui_com_mais_paths():
    spec_small = _base_spec()
    spec_small["n_paths"] = 15_000
    spec_small["product"] = {"style": "european", "type": "european_call", "asset": 0, "K": 100.0}
    p1, se1 = price_from_spec(spec_small)

    spec_big = _base_spec()
    spec_big["n_paths"] = 60_000
    spec_big["product"] = spec_small["product"]
    spec_big["seed"] = 7
    p2, se2 = price_from_spec(spec_big)

    assert se2 < se1  # mais paths -> menor erro
