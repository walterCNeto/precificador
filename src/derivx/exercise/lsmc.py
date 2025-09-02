# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
from ..models.gbm import RiskNeutralGBM


# ---------------------------------------------------------------------------
# Especificação de exercício (genérica)
# ---------------------------------------------------------------------------
@dataclass
class ExerciseSpec:
    # índices de tempo (inteiros) nas "times" onde o exercício é permitido (inclua t0 e T, se quiser)
    exercise_idx: List[int]
    # função: dado "paths" e o índice de tempo k, retorna payoff imediato (array shape: (n_paths,))
    immediate_payoff: Callable[[Dict[str, np.ndarray], int], np.ndarray]


# ---------------------------------------------------------------------------
# Bases/Features
# ---------------------------------------------------------------------------
def default_basis(features: np.ndarray) -> np.ndarray:
    """
    Base polinomial simples: [1, x, x^2, cruzadas].
    features: shape (n_amostras, n_features)
    """
    x = features
    cols = [np.ones((x.shape[0], 1))]
    cols.append(x)          # grau 1
    cols.append(x ** 2)     # grau 2 (termos quadráticos)
    if x.shape[1] > 1:      # termos cruzados
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                cols.append(x[:, [i]] * x[:, [j]])
    return np.concatenate(cols, axis=1)


def make_features(paths: Dict[str, np.ndarray], t_idx: int) -> np.ndarray:
    """
    Cria features a partir dos caminhos no tempo t_idx.
    Aqui usamos log-preços (clamp para evitar log(0)).
    paths["S"]: shape (n_paths, n_times, n_assets)
    retorna: shape (n_paths, n_assets)
    """
    St = paths["S"][:, t_idx, :]
    return np.log(np.clip(St, 1e-12, None))


def _regress(phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS com pseudo-inversa. Retorna beta."""
    return np.linalg.pinv(phi) @ y


# ---------------------------------------------------------------------------
# LSMC genérico (corrigido): retropropagação por janelas, desconto Delta t correto
# ---------------------------------------------------------------------------
def lsmc_price(
    model: RiskNeutralGBM,
    paths: Dict[str, np.ndarray],
    spec: ExerciseSpec,
    times: np.ndarray,  # shape (n_times,), crescente
    feature_fn: Callable[[Dict[str, np.ndarray], int], np.ndarray] = make_features,
    basis_fn: Callable[[np.ndarray], np.ndarray] = default_basis,
) -> Tuple[float, float]:
    """
    Preço via LSMC para um payoff exercível nas datas spec.exercise_idx.

    Ideia:
      1) Defina V como payoff no vencimento (última data de exercício).
      2) Para cada janela (t_i <- t_j) indo para trás:
           - Y = DF(t_i, t_j) * V  (valor de continuação descontado para t_i)
           - Regressão de Y *apenas nos caminhos ITM* (immediate_payoff>0) para estimar E[Y|S_{t_i}]
           - Decisão: exerce se payoff_i >= continuação_estimada
           - Atualização: V = payoff_i (se exercer) ou V = Y (se continuar)
      3) Ao final, V já está avaliado em t0; preço = média(V), se = std/sqrt(n)

    Retorna: (preço, erro-padrão)
    """
    ex_idx = sorted(list(spec.exercise_idx))
    S = paths["S"]
    n_paths = S.shape[0]

    if len(ex_idx) < 1:
        raise ValueError("ExerciseSpec.exercise_idx vazio.")

    # 1) Valor no vencimento (última data de exercício)
    k_last = ex_idx[-1]
    V = np.asarray(spec.immediate_payoff(paths, k_last), dtype=float)
    V = np.maximum(V, 0.0)  # segurança

    # 2) Retropropagação por janelas (t_i <- t_j)
    for k in range(len(ex_idx) - 2, -1, -1):
        i = ex_idx[k]
        j = ex_idx[k + 1]

        # desconto do bloco j -> i
        disc = float(model.df(float(times[i]), float(times[j])))

        # alvo: valor de continuação já descontado para t_i
        Y = disc * V

        # payoff imediato em t_i
        imm = np.asarray(spec.immediate_payoff(paths, i), dtype=float)
        imm = np.maximum(imm, 0.0)

        # in-the-money (onde regredir faz sentido)
        itm = imm > 1e-12

        # por padrão, continuação = Y (fora do ITM mantemos Y)
        cont = np.array(Y, copy=True)

        if np.any(itm):
            feats_all = feature_fn(paths, i)             # (n_paths, n_features)
            X_itm = basis_fn(feats_all[itm, :])          # só ITM

            # se #amostras >= #colunas, regressão estável
            if X_itm.shape[0] >= X_itm.shape[1]:
                y_itm = Y[itm]
                beta = _regress(X_itm, y_itm)

                # <<< CORREÇÃO: só ITM recebe a predição >>>
                cont_itm = X_itm @ beta
                cont[itm] = cont_itm
            # senão, mantemos cont = Y para todos

        # decisão ótima em t_i
        exer = imm >= cont
        V = np.where(exer, imm, Y)

    # 3) V já está em t0
    price = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0
    return price, se


# ---------------------------------------------------------------------------
# Utilitário: LSMC "cross-fit" para put (exemplo standalone)
# (Mantido; assume T=1.0 -> dt = 1/steps)
# ---------------------------------------------------------------------------
def _poly_basis_default(S: np.ndarray) -> np.ndarray:
    # base simples e estável p/ put: [1, S, S^2]
    return np.column_stack([np.ones_like(S), S, S * S])


def lsmc_put_crossfit(
    S_paths: np.ndarray,          # shape: (n_paths, steps+1)
    K: float,
    r: float,
    exercise_every: int,
    basis_fn=_poly_basis_default,
    seed: int = 7,
) -> Tuple[float, float]:
    """
    LSMC com cross-fit (fora-da-amostra) e desconto por Delta t entre janelas.
    S_paths inclui a coluna t0 e t_steps; passos de tempo UNIFORMES.
    Assume T=1.0 (dt = 1/steps). Para T != 1, adapte dt.

    Retorna (preço, se).
    """
    n_paths, n_cols = S_paths.shape
    steps = n_cols - 1
    dt = 1.0 / steps  # assumir T = 1.0

    # índices de exercício incluindo t0 e T
    ex_idx = list(range(0, steps + 1, exercise_every))
    if ex_idx[-1] != steps:
        ex_idx.append(steps)

    # payoff no vencimento
    V = np.maximum(K - S_paths[:, ex_idx[-1]], 0.0)

    # retropropaga blocado por janelas (ti <- tj)
    for k in range(len(ex_idx) - 2, -1, -1):
        ti = ex_idx[k]
        tj = ex_idx[k + 1]
        disc = math.exp(-r * (tj - ti) * dt)

        St = S_paths[:, ti]
        Y = disc * V
        itm = (K - St) > 0

        if np.any(itm):
            X = basis_fn(St[itm])
            y = Y[itm]

            # split A/B com seed dependente do tempo p/ reprodutibilidade
            idx = np.arange(X.shape[0])
            np.random.default_rng(seed + ti).shuffle(idx)
            A = idx[::2]
            B = idx[1::2]
            XA, yA = X[A], y[A]
            XB, yB = X[B], y[B]

            betaA, *_ = np.linalg.lstsq(XA, yA, rcond=None)
            betaB, *_ = np.linalg.lstsq(XB, yB, rcond=None)

            cont = np.array(Y, copy=True)
            cont_itm = np.empty_like(y)
            cont_itm[A] = XA @ betaB
            cont_itm[B] = XB @ betaA
            cont[itm] = cont_itm
        else:
            cont = Y

        exer = (K - St) >= cont
        V = np.where(exer, K - St, Y)

    price = float(np.mean(V))
    se = float(np.std(V, ddof=1) / math.sqrt(n_paths))
    return price, se
