# src/derivx/payoffs/extra.py
from __future__ import annotations
import numpy as np
from .core import PF, relu

def _term(paths, asset: int):
    return PF.terminal(paths, asset)

def _running_max(paths, asset: int):
    # usa PF.running_max se existir; senão, calcula via paths["S"]
    try:
        return PF.running_max(paths, asset)
    except AttributeError:
        S = paths["S"][:, :, asset]
        return np.maximum.accumulate(S, axis=1)[:, -1]

def build_extra_payoff(product: dict):
    """
    Retorna uma função payoff(paths)->np.ndarray para os tipos extras.
    Se não suportado aqui, retorna None (DSL original continua tratando).
    """
    style = product.get("style", "european")
    typ   = product.get("type", "")
    if style != "european":
        # todos os extras aqui são "european" (inclui path-dep. como up_and_in/out)
        return None

    # ----------------- Digitais -----------------
    if typ == "cash_or_nothing_call":
        asset = int(product["asset"]); K = float(product["K"])
        cash  = float(product.get("cash", 1.0))
        return lambda paths: ( _term(paths, asset) > K ).astype(float) * cash

    if typ == "cash_or_nothing_put":
        asset = int(product["asset"]); K = float(product["K"])
        cash  = float(product.get("cash", 1.0))
        return lambda paths: ( _term(paths, asset) < K ).astype(float) * cash

    if typ == "asset_or_nothing_call":
        asset = int(product["asset"]); K = float(product["K"])
        return lambda paths: _term(paths, asset) * ( _term(paths, asset) > K ).astype(float)

    if typ == "asset_or_nothing_put":
        asset = int(product["asset"]); K = float(product["K"])
        return lambda paths: _term(paths, asset) * ( _term(paths, asset) < K ).astype(float)

    # ----------------- Basket e Exchange -----------------
    if typ == "basket_call":
        # requer "assets" e "weights" do mesmo tamanho; K obrigatório
        assets  = list(product["assets"])
        weights = np.asarray(product["weights"], dtype=float)
        if len(assets) != len(weights):
            raise ValueError("basket_call: 'assets' e 'weights' devem ter mesmo tamanho.")
        K = float(product["K"])
        def payoff(paths):
            STw = 0.0
            for a, w in zip(assets, weights):
                STw = STw + w * _term(paths, int(a))
            return relu(STw - K)
        return payoff

    if typ == "exchange_call":
        # max(S_i - S_j, 0) (Margrabe K=0)
        i = int(product["asset_long"])
        j = int(product["asset_short"])
        return lambda paths: relu(_term(paths, i) - _term(paths, j))

    # ----------------- Barreira: Up-and-In vanilla call -----------------
    if typ == "up_and_in_call":
        asset = int(product["asset"]); K = float(product["K"]); B = float(product["barrier"])
        def payoff(paths):
            touched = (_running_max(paths, asset) >= B).astype(float)
            vanilla = relu(_term(paths, asset) - K)
            return vanilla * touched
        return payoff

    # não reconhecido aqui
    return None
