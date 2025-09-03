from .bs import d1d2, Phi, bs_call, bs_put
from .haug import (
    cash_or_nothing_call, cash_or_nothing_put,
    asset_or_nothing_call, asset_or_nothing_put,
    gap_call, gap_put,
    margrabe_exchange_call,
)

# Sinônimos amigáveis / compatíveis com a DSL
# (mantemos também os nomes "clássicos" que você já usa)
bs_call_price = bs_call
bs_put_price  = bs_put

def digital_cash_call_price(S0, K, r, q, sigma, T, cash=1.0):
    return cash_or_nothing_call(S0, K, r, q, sigma, T, cash)

def digital_cash_put_price(S0, K, r, q, sigma, T, cash=1.0):
    return cash_or_nothing_put(S0, K, r, q, sigma, T, cash)

def digital_asset_call_price(S0, K, r, q, sigma, T):
    return asset_or_nothing_call(S0, K, r, q, sigma, T)

def digital_asset_put_price(S0, K, r, q, sigma, T):
    return asset_or_nothing_put(S0, K, r, q, sigma, T)

def gap_call_price(S0, K, trigger, r, q, sigma, T):
    return gap_call(S0, K, trigger, r, q, sigma, T)

def gap_put_price(S0, K, trigger, r, q, sigma, T):
    return gap_put(S0, K, trigger, r, q, sigma, T)

def margrabe_exchange_call_price(S1, S2, q1, q2, sig1, sig2, rho, T):
    return margrabe_exchange_call(S1, S2, q1, q2, sig1, sig2, rho, T)

__all__ = [
    # nomes já existentes
    "d1d2", "Phi", "bs_call", "bs_put",
    "cash_or_nothing_call", "cash_or_nothing_put",
    "asset_or_nothing_call", "asset_or_nothing_put",
    "gap_call", "gap_put", "margrabe_exchange_call",
    # aliases compatíveis com DSL
    "bs_call_price", "bs_put_price",
    "digital_cash_call_price", "digital_cash_put_price",
    "digital_asset_call_price", "digital_asset_put_price",
    "gap_call_price", "gap_put_price",
    "margrabe_exchange_call_price",
]
