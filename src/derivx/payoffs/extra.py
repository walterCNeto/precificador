from __future__ import annotations
from typing import Any, Dict
import numpy as np
from .core import PF, relu

def _terminal(paths, asset: int):
    return PF.terminal(paths, asset)

def build_extra_payoff(product: Dict[str, Any]):
    """
    Payoffs europeus adicionais (para o motor MC):
      - Digitais: cash/asset-or-nothing (várias grafias)
      - Gap call/put (K1/K2)
      - Exchange (Margrabe), com aliases (exchange_call)
    Retorna callable(paths)->array ou None se não suportar aqui.
    """
    ptype = str(product.get("type","")).lower()

    # ---------- Digitais (sinônimos suportados) ----------
    if ptype in ("cash_or_nothing_call","digital_cash_call","binary_cash_call"):
        a=int(product.get("asset",0)); K=float(product["K"]); cash=float(product.get("cash",1.0))
        return lambda paths: ((_terminal(paths,a) > K).astype(float) * cash)

    if ptype in ("cash_or_nothing_put","digital_cash_put","binary_cash_put"):
        a=int(product.get("asset",0)); K=float(product["K"]); cash=float(product.get("cash",1.0))
        return lambda paths: ((_terminal(paths,a) < K).astype(float) * cash)

    if ptype in ("asset_or_nothing_call","digital_asset_call"):
        a=int(product.get("asset",0)); K=float(product["K"])
        return lambda paths: (_terminal(paths,a) * (_terminal(paths,a) > K).astype(float))

    if ptype in ("asset_or_nothing_put","digital_asset_put"):
        a=int(product.get("asset",0)); K=float(product["K"])
        return lambda paths: (_terminal(paths,a) * (_terminal(paths,a) < K).astype(float))

    # ---------- Gap options ----------
    if ptype == "gap_call":
        a=int(product.get("asset",0))
        K1=float(product.get("K1", product.get("payoff_strike")))
        K2=float(product.get("K2", product.get("trigger")))
        return lambda paths: ( relu(_terminal(paths,a) - K1)
                               * (_terminal(paths,a) > K2).astype(float) )

    if ptype == "gap_put":
        a=int(product.get("asset",0))
        K1=float(product.get("K1", product.get("payoff_strike")))
        K2=float(product.get("K2", product.get("trigger")))
        return lambda paths: ( relu(K1 - _terminal(paths,a))
                               * (_terminal(paths,a) < K2).astype(float) )

    # ---------- Margrabe (exchange) ----------
    if ptype in ("margrabe_exchange_call","exchange_call"):
        a_long  = int(product.get("asset1",  product.get("asset_long", 0)))
        a_short = int(product.get("asset2",  product.get("asset_short",1)))
        return lambda paths: relu(_terminal(paths,a_long) - _terminal(paths,a_short))

    return None
