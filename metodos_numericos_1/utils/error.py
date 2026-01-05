import numpy as np
from numpy.typing import NDArray  # type: ignore


def calculate_matrix_error(xi: NDArray, x0: NDArray) -> np.float64:
    """
    Calcula o erro relativo entre duas soluções.
    :param xi: Solução atual
    :param x0: Solução anterior
    :return: Erro relativo máximo
    """
    # error_array: NDArray = np.abs((xi - x0) / xi)
    error_array: NDArray = np.abs((xi - x0))

    return np.max(error_array)


def is_indertemination(a: np.float64 | float, b: np.float64 | float) -> bool:
    """Esta função verifica se a subtração
    de dois valores do tipo numpy.float64
    gera o valor 0.0.
    Retorna *true* se a - b = 0.0
    Retorna *false* caso contrário

    Args:
        a (np.float64): um dos divisores
        b (np.float64): um dos divisores

    Returns:
        bool: o resultado da subtração é 0.0
    """

    return (a - b) == np.float64(0.0)


def error_to_significant_digits(error: np.float64 | float) -> np.float64:
    """
    Calculate the tolerance based on the error in terms of significant digits.
    The tolerance is calculated as -log10(2 * |error|).
    If the error is zero, the tolerance is set to zero.
    Parameters
    ----------
    error : np.float64
        The error value for which to calculate the tolerance.
    Returns
    -------
    np.float64
        The calculated tolerance in terms of significant digits.
    """

    if error == 0:
        tol = np.float64(0.0)

    tol = -np.log10(2 * np.abs(error))

    return tol


def calcule_absolute_error(
    x_new: np.float64 | float,
    x_old: np.float64 | float,
) -> np.float64:
    """
    Calculate the absolute error between two values.
    Parameters
    ----------
    x_new : np.float64 | float
        The new value.
    x_old : np.float64 | float
        The old value.
    Returns
    -------
    np.float64
        The calculated absolute error.
    """
    return np.abs(x_new - x_old)


def calcule_relative_error(
    x_new: np.float64 | float,
    x_old: np.float64 | float,
) -> np.float64:
    """
    Calculate the relative error between two values.
    Parameters
    ----------
    x_new : np.float64 | float
        The new value.
    x_old : np.float64 | float
        The old value.
    Returns
    -------
    np.float64
        The calculated relative error.
    """
    if x_new == 0:
        return np.float64(0.0)

    return np.abs(calcule_absolute_error(x_new, x_old) / x_new)


if __name__ == "__main__":
    # Example usage - Aula
    # Erro relativo de 0.0001 (0.01%)
    # Raiz 10.000, Raiz calculada 10.001
    print("Error to significant digits:")
    print("Tolerance for 0.0001:", error_to_significant_digits(0.0001))
