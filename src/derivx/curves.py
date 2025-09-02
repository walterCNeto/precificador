from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class PiecewiseFlatCurve:
    times: np.ndarray  # nós crescentes (ex.: [0.5, 1.0, 2.0])
    rates: np.ndarray  # taxas contínuas por trecho

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.rates = np.asarray(self.rates, dtype=float)
        assert self.times.ndim == 1 and self.rates.ndim == 1
        assert len(self.times) == len(self.rates)
        assert np.all(np.diff(self.times) > 0.0)

    def r(self, t: float) -> float:
        idx = np.searchsorted(self.times, t, side="right") - 1
        idx = min(max(idx, 0), len(self.rates) - 1)
        return float(self.rates[idx])

    def integral(self, t0: float, t1: float) -> float:
        """Integra r(t) exatamente para curva piecewise-flat no intervalo [t0, t1]."""
        if t1 <= t0:
            return 0.0
        left = float(t0)
        right = float(t1)
        # knots: [0, t1, t2, ...]; o último trecho vai até +inf
        knots = np.concatenate(([0.0], self.times))
        N = len(self.rates)
        total = 0.0
        for i in range(N):
            seg_start = knots[i]
            seg_end = knots[i + 1] if i < N - 1 else float("inf")
            a = max(left, seg_start)
            b = min(right, seg_end)
            if b > a:
                total += float(self.rates[i]) * (b - a)
            if seg_end >= right:
                break
        return total

    def df(self, t0: float, t1: float) -> float:
        return math.exp(-self.integral(t0, t1))
