import numpy as np
from numpy.typing import NDArray
from typing import Union


FloatArray = NDArray[np.float64]


def pertinencia_triangular_scalar(x: float, a: float, b: float, c: float) -> float:
    """Pertinência triangular para um único valor x."""
    x = float(x)
    # lados de subida/descida degenerados são tratados
    if a < b:
        if a < x < b:
            subida = (x - a) / (b - a)
        elif x == b:
            return 1.0
        else:
            subida = 0.0
    else:
        subida = 0.0

    if b < c:
        if b < x < c:
            descida = (c - x) / (c - b)
        elif x == b:
            return 1.0
        else:
            descida = 0.0
    else:
        descida = 0.0

    y = max(0.0, min(1.0, max(subida, descida)))
    return y


def pertinencia_trapezoidal_scalar(
    x: float, a: float, b: float, c: float, d: float
) -> float:
    """Pertinência trapezoidal para um único valor x."""
    x = float(x)
    # subida
    if a < b and a < x < b:
        y = (x - a) / (b - a)
    # patamar
    elif b <= x <= c:
        y = 1.0
    # descida
    elif c < x < d and c < d:
        y = (d - x) / (d - c)
    else:
        y = 0.0
    return float(max(0.0, min(1.0, y)))


def pertinencia_gaussiana_scalar(x: float, c: float, sigma: float) -> float:
    """Pertinência gaussiana para um único valor x (float)."""
    return float(np.exp(-0.5 * ((float(x) - c) / sigma) ** 2))


def _vectorize_scalar(func):
    vec = np.vectorize(func, otypes=[float])

    def wrapped(x: Union[FloatArray, float], *args, **kwargs):
        if np.isscalar(x) or (isinstance(x, (float, int))):
            return float(func(float(x), *args, **kwargs))
        return vec(x, *args, **kwargs).astype(float)

    return wrapped


pertinencia_triangular = _vectorize_scalar(pertinencia_triangular_scalar)
pertinencia_trapezoidal = _vectorize_scalar(pertinencia_trapezoidal_scalar)
pertinencia_gaussiana = _vectorize_scalar(pertinencia_gaussiana_scalar)
