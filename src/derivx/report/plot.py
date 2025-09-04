# src/derivx/report/plot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import matplotlib
matplotlib.use("Agg")  # backend não interativo para CI/servers
import matplotlib.pyplot as plt

from ..dsl.spec import price_from_spec


@dataclass
class PlotReport:
    price: float
    se: float
    inputs: Dict[str, Any]
    payoff_equation_tex: Optional[str]
    pricing_equation_tex: Optional[str]
    figure_path: Optional[str]
    vary_asset: Optional[int]


def _gather_inputs(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Coleta um resumo de inputs amigável para exibir no report."""
    model = spec.get("model", {})
    grid = spec.get("grid", {})
    product = spec.get("product", {})
    return {
        "engine": spec.get("engine"),
        "S0": list(spec.get("S0", [])),
        "grid": {k: grid.get(k) for k in ("T", "steps", "NS", "NT", "Smax_mult", "alpha", "N", "eta")},
        "model": {k: model.get(k) for k in ("name", "r", "q", "sigma", "corr", "kappa", "theta", "xi", "rho", "v0")},
        "product": product,
        "n_paths": spec.get("n_paths"),
        "seed": spec.get("seed"),
    }


def _payoff_equation_tex(product: Dict[str, Any]) -> Tuple[str, Optional[int], str]:
    """
    Retorna (equacao_tex, vary_asset, xlabel) para o payoff.
    Sempre retorna 3 itens (strings podem ser vazias para 'sem plot').
    """
    ptype = str(product.get("type", "")).lower()
    style = str(product.get("style", "european")).lower()
    a = int(product.get("asset", 0))
    xlabel = "Underlying price S"

    if ptype == "european_call":
        K = float(product["K"])
        return rf"g(S)=\max(S-K,0),\ K={K}", a, xlabel
    if ptype == "european_put":
        K = float(product["K"])
        return rf"g(S)=\max(K-S,0),\ K={K}", a, xlabel

    if ptype == "asian_arith_call":
        K = float(product["K"])
        # Atenção ao mathtext: use \frac{1}{n} (com chaves)
        return rf"g(A)=\max(A-K,0),\ A=\frac{{1}}{{n}}\sum S_t,\ K={K}", a, xlabel

    if ptype in ("cash_or_nothing_call", "digital_cash_call", "binary_cash_call"):
        K = float(product["K"]); C = float(product.get("cash", 1.0))
        return rf"g(S)=\mathbf{{1}}_{{S>K}}\cdot {C},\ K={K}", a, xlabel
    if ptype in ("cash_or_nothing_put", "digital_cash_put", "binary_cash_put"):
        K = float(product["K"]); C = float(product.get("cash", 1.0))
        return rf"g(S)=\mathbf{{1}}_{{S<K}}\cdot {C},\ K={K}", a, xlabel
    if ptype in ("asset_or_nothing_call", "digital_asset_call"):
        K = float(product["K"])
        return rf"g(S)=\mathbf{{1}}_{{S>K}}\cdot S,\ K={K}", a, xlabel
    if ptype in ("asset_or_nothing_put", "digital_asset_put"):
        K = float(product["K"])
        return rf"g(S)=\mathbf{{1}}_{{S<K}}\cdot S,\ K={K}", a, xlabel

    if ptype == "gap_call":
        K1 = float(product.get("K1", product.get("payoff_strike")))
        K2 = float(product.get("K2", product.get("trigger")))
        return rf"g(S)=\mathbf{{1}}_{{S>K_2}}\cdot (S-K_1),\ K_1={K1},\ K_2={K2}", a, xlabel
    if ptype == "gap_put":
        K1 = float(product.get("K1", product.get("payoff_strike")))
        K2 = float(product.get("K2", product.get("trigger")))
        return rf"g(S)=\mathbf{{1}}_{{S<K_2}}\cdot (K_1-S),\ K_1={K1},\ K_2={K2}", a, xlabel

    if ptype in ("margrabe_exchange_call", "exchange_call"):
        # 2D → usamos slice 1D variando o 'long' (asset_long/asset1)
        xlabel = "S_1"
        return r"g(S_1,S_2)=\max(S_1-S_2,0)", int(product.get("asset_long", product.get("asset1", 0))), xlabel

    # Produtos de IR (não tem payoff 1D em S):
    if ptype in ("zcb", "fra", "swap", "cap", "floor", "payer_swaption", "receiver_swaption"):
        return "", None, ""

    # fallback
    return "", a, xlabel


def _pricing_equation_tex(spec: Dict[str, Any]) -> str:
    """Escolhe a equação de precificação (fechada) quando aplicável."""
    product = spec.get("product", {})
    ptype = str(product.get("type", "")).lower()

    # Vanillas BS
    if ptype == "european_call":
        return r"\mathrm{BS}:~C=S_0 e^{-qT}\Phi(d_1)-K e^{-rT}\Phi(d_2)"
    if ptype == "european_put":
        return r"\mathrm{BS}:~P=K e^{-rT}\Phi(-d_2)-S_0 e^{-qT}\Phi(-d_1)"

    # Digitais (Haug)
    if ptype in ("cash_or_nothing_call", "digital_cash_call", "binary_cash_call"):
        return r"\mathrm{Haug}:~V=C\,e^{-rT}\Phi(d_2)"
    if ptype in ("cash_or_nothing_put", "digital_cash_put", "binary_cash_put"):
        return r"\mathrm{Haug}:~V=C\,e^{-rT}\Phi(-d_2)"
    if ptype in ("asset_or_nothing_call", "digital_asset_call"):
        return r"\mathrm{Haug}:~V=S_0 e^{-qT}\Phi(d_1)"
    if ptype in ("asset_or_nothing_put", "digital_asset_put"):
        return r"\mathrm{Haug}:~V=S_0 e^{-qT}\Phi(-d_1)"

    # Gap (Haug) – usando d's com trigger K2
    if ptype == "gap_call":
        return r"\mathrm{Haug}:~V=S_0 e^{-qT}\Phi(d_1)-K_1 e^{-rT}\Phi(d_2)"
    if ptype == "gap_put":
        return r"\mathrm{Haug}:~V=K_1 e^{-rT}\Phi(-d_2)-S_0 e^{-qT}\Phi(-d_1)"

    # Margrabe (exchange)
    if ptype in ("margrabe_exchange_call", "exchange_call"):
        return r"\mathrm{Margrabe}:~V=S_{1,0}e^{-q_1T}\Phi(d_1)-S_{2,0}e^{-q_2T}\Phi(d_2)"

    # IR fechadas (Black-76 e afins)
    if ptype == "zcb":
        return r"\mathrm{ZCB}:~P(0,T)=D(0,T)"
    if ptype == "fra":
        return r"\mathrm{FRA}:~\mathrm{PV}=N\big(D(0,T_1)-(1+K\tau)D(0,T_2)\big)"
    if ptype == "swap":
        return r"\mathrm{Swap}:~\mathrm{PV}=N\Big(\sum_i D(0,T_{i+1})\tau_i\,K - \sum_i D(0,T_{i+1})\tau_i\,F_i\Big)"
    if ptype == "cap":
        # Evite \text{} e \big com mathtext
        return r"\mathrm{Cap}=\sum_i N\,D(0,T_{i+1})\,\tau_i\,(F_i\Phi(d_1)-K\Phi(d_2))"
    if ptype == "floor":
        return r"\mathrm{Floor}=\sum_i N\,D(0,T_{i+1})\,\tau_i\,(K\Phi(-d_2)-F_i\Phi(-d_1))"
    if ptype == "payer_swaption":
        return r"\mathrm{PayerSwaption}=N\,D(0,T_0)\big(F\Phi(d_1)-K\Phi(d_2)\big)"
    if ptype == "receiver_swaption":
        return r"\mathrm{ReceiverSwaption}=N\,D(0,T_0)\big(K\Phi(-d_2)-F\Phi(-d_1)\big)"

    # fallback
    return ""


def _payoff_curve_1d(product: Dict[str, Any], x: np.ndarray, S0: List[float]) -> np.ndarray:
    """Curva de payoff 1D (para visualização). Usa aproximações quando necessário."""
    ptype = str(product.get("type", "")).lower()

    if ptype == "european_call":
        K = float(product["K"])
        return np.maximum(x - K, 0.0)
    if ptype == "european_put":
        K = float(product["K"])
        return np.maximum(K - x, 0.0)
    if ptype == "asian_arith_call":
        # Não é exatamente função de S instantâneo; mostramos forma tipo call
        K = float(product["K"])
        return np.maximum(x - K, 0.0)
    if ptype in ("cash_or_nothing_call", "digital_cash_call", "binary_cash_call"):
        K = float(product["K"]); C = float(product.get("cash", 1.0))
        return np.where(x > K, C, 0.0)
    if ptype in ("cash_or_nothing_put", "digital_cash_put", "binary_cash_put"):
        K = float(product["K"]); C = float(product.get("cash", 1.0))
        return np.where(x < K, C, 0.0)
    if ptype in ("asset_or_nothing_call", "digital_asset_call"):
        K = float(product["K"])
        return np.where(x > K, x, 0.0)
    if ptype in ("asset_or_nothing_put", "digital_asset_put"):
        K = float(product["K"])
        return np.where(x < K, x, 0.0)
    if ptype == "gap_call":
        K1 = float(product.get("K1", product.get("payoff_strike")))
        K2 = float(product.get("K2", product.get("trigger")))
        return np.where(x > K2, x - K1, 0.0)
    if ptype == "gap_put":
        K1 = float(product.get("K1", product.get("payoff_strike")))
        K2 = float(product.get("K2", product.get("trigger")))
        return np.where(x < K2, K1 - x, 0.0)
    if ptype in ("margrabe_exchange_call", "exchange_call"):
        # varia S_long, prende S_short em S0[short]
        s_short = float(S0[int(product.get("asset_short", product.get("asset2", 1)))])
        return np.maximum(x - s_short, 0.0)

    # default “linha zero”
    return np.zeros_like(x)


def plot_report(
    spec: Dict[str, Any],
    vary: Optional[int] = None,
    filename: Optional[str] = None,
    dpi: int = 150,
) -> PlotReport:
    """
    Gera um relatório gráfico do payoff + inputs + equações + preço.
    Retorna um PlotReport; se filename for fornecido, salva a figura.

    - vary: índice do ativo a variar (se aplicável). Caso None, é inferido.
    """
    inputs = _gather_inputs(spec)
    product = spec.get("product", {})
    S0 = list(spec.get("S0", []))
    grid = spec.get("grid", {})
    T = grid.get("T", None)

    # preço
    price, se = price_from_spec(spec)

    # equações (payoff e de precificação)
    peq, inferred_vary, xlab = _payoff_equation_tex(product)
    vary_asset = vary if vary is not None else inferred_vary
    pric_eq = _pricing_equation_tex(spec)

    # --- layout
    #   Se dá para plotar 1D: duas linhas -> [plot payoff] e [texto]
    #   Senão: só a área de texto.
    can_plot = (vary_asset is not None) and (len(S0) > 0)

    if can_plot:
        fig = plt.figure(figsize=(9, 6), dpi=dpi)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[3, 2], hspace=0.35, wspace=0.3)
        ax_pay = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[:, 1])  # coluna direita toda para texto
        ax_blank = fig.add_subplot(gs[1, 0])
        ax_blank.axis("off")

        # range de S para plot (heurística)
        base = float(S0[vary_asset])
        K = float(product.get("K", base))
        smin = max(0.0, 0.2 * min(base, K))
        smax = 1.8 * max(base, K)
        if smax <= smin:
            smax = smin + (base if base > 0 else 1.0)

        x = np.linspace(smin, smax, 400)
        y = _payoff_curve_1d(product, x, S0)

        # tipo de traçado
        ptype = str(product.get("type", "")).lower()
        if "cash_or_nothing" in ptype:
            ax_pay.step(x, y, where="post")
        else:
            ax_pay.plot(x, y)
        ax_pay.set_xlabel(xlab)
        ax_pay.set_ylabel("Payoff")
        ax_pay.set_title("Curva de Payoff (slice 1D)")

    else:
        fig = plt.figure(figsize=(9, 4), dpi=dpi)
        ax_info = fig.add_subplot(111)

    # --- bloco de texto: inputs + equações + preço
    ax_info.axis("off")

    lines: List[str] = []
    lines.append(rf"$\mathrm{{Engine}}$: {inputs['engine']}")
    if T is not None:
        lines.append(rf"$T={T}$")
    if S0:
        lines.append(rf"$S_0={S0}$")
    if inputs.get("model"):
        lines.append(rf"$\mathrm{{Model}}$: {inputs['model']}")
    if inputs.get("n_paths"):
        lines.append(rf"$n\_{{paths}}={inputs['n_paths']}$,\ \mathrm{{seed}}={inputs.get('seed')}$")

    if peq:
        lines.append(rf"$\mathrm{{Payoff}}:~{peq}$")
    if pric_eq:
        lines.append(rf"$\mathrm{{Preço~(fechada)}}:~{pric_eq}$")

    # preço numérico
    if se and se > 0:
        lines.append(rf"$\mathrm{{Preço}}:~{price:.6f}\ \pm\ 1.96\cdot {se:.6f}$")
    else:
        lines.append(rf"$\mathrm{{Preço}}:~{price:.6f}$")

    y0 = 0.95
    dy = 0.08
    for i, txt in enumerate(lines):
        fig.text(0.58, y0 - i * dy, txt, fontsize=11, va="top")

    # salvar se pedido
    out_path = None
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        out_path = filename
    plt.close(fig)

    return PlotReport(
        price=price,
        se=se,
        inputs=inputs,
        payoff_equation_tex=peq,
        pricing_equation_tex=pric_eq,
        figure_path=out_path,
        vary_asset=vary_asset,
    )
