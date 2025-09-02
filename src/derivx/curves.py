from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class PiecewiseFlatCurve:
    """Curva r(t) piecewise-flat."""
    times: np.ndarray  # nÃ³s crescentes (ex.: [0.5, 1.0, 2.0])
    rates: np.ndarray  # taxas contÃ­nuas por trecho

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
        if t1 <= t0:
            return 0.0
        grid = np.linspace(t0, t1, 17)  # 16 subintervalos
        vals = np.array([self.r(x) for x in grid])
	return float(np.trapezoid(vals, grid))

    def df(self, t0: float, t1: float) -> float:
        return math.exp(-self.integral(t0, t1))


