from typing import Callable
import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


def produto_carteziano(mu_a: FloatArr, mu_b: FloatArr,
                       tnorm: Callable[[FloatArr, FloatArr], FloatArr] = np.multiply
                       ) -> FloatArr:
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    return tnorm(mu_a[:, None], mu_b[None, :])  # shape (len(mu_a), len(mu_b))


def produto_carteziano_scalar(mu_a: FloatArr, mu_b: FloatArr,
                              tnorm_scalar: Callable[[float, float], float]
                              ) -> FloatArr:
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    vec = np.vectorize(tnorm_scalar, otypes=[float])
    return vec(mu_a[:, None], mu_b[None, :])