import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def pertinencia_triangular(x: FloatArray, a: float, b: float, c: float) -> FloatArray:
    """Função de pertinência triangular."""
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr, dtype=float)

    if a != b:
        subida = (a < x_arr) & (x_arr < b)
        y[subida] = (x_arr[subida] - a) / (b - a)

    y[x_arr == b] = 1.0

    if b != c:
        descida = (b < x_arr) & (x_arr < c)
        y[descida] = (c - x_arr[descida]) / (c - b)

    return np.clip(y, 0.0, 1.0)


def pertinencia_trapezoidal(
    x: FloatArray,
    a: float,
    b: float,
    c: float,
    d: float,
) -> FloatArray:
    """Função de pertinência trapezoidal."""
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr, dtype=float)

    if a != b:
        subida = (a < x_arr) & (x_arr < b)
        y[subida] = (x_arr[subida] - a) / (b - a)

    patamar = (b <= x_arr) & (x_arr <= c)
    y[patamar] = 1.0

    if c != d:
        descida = (c < x_arr) & (x_arr < d)
        y[descida] = (d - x_arr[descida]) / (d - c)

    return np.clip(y, 0.0, 1.0)


def pertinencia_gaussiana(x: FloatArray, c: float, sigma: float) -> FloatArray:
    """Função de pertinência gaussiana."""
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)
