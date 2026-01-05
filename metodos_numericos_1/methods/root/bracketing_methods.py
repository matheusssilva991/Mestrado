import sys
import os

# Adicionar o diretório raiz do projeto ao Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from utils.parser import evaluate_one_variable
from typing import Union, Callable
from utils.error import is_indertemination


def bissection(
    func: Union[str, Callable],
    xl: np.float64 | float,
    xu: np.float64 | float,
    tol: np.float64 | float,
    max_iter: int,
) -> np.float64 | float:
    iter: int = 0
    relative_error: np.float64 = np.float64(100.0)
    x: np.float64 | float = xl

    fl: np.float64 | float = evaluate_one_variable(func, xl)

    while (relative_error > tol) and (iter < max_iter):
        x_old: np.float64 | float = x

        x = (xl + xu) / 2
        fx: np.float64 | float = evaluate_one_variable(func, x)

        iter = iter + 1

        if x != np.float64(0.0):
            relative_error = np.abs((x - x_old) / x) * 100
        else:
            return np.inf

        if (fl * fx) < 0:
            xu = x
        else:
            xl = x
            fl = fx

    return x


def false_position(
    func: Union[str, Callable],
    xl: np.float64 | float,
    xu: np.float64 | float,
    tol: np.float64 | float,
    max_iter: int,
) -> np.float64 | float:
    x: np.float64 | float = np.float64(0.0)
    iter: int = 0
    relative_error: np.float64 | float = np.float64(100.0)

    fl: np.float64 | float = evaluate_one_variable(func, xl)
    fu: np.float64 | float = evaluate_one_variable(func, xu)

    if abs(fl) < abs(fu):
        x = xl
    else:
        x = xu

    while (relative_error > tol) and (iter < max_iter):
        x_old: np.float64 | float = x
        iter = iter + 1

        if is_indertemination(fu, fl):
            return np.inf

        x = xu + (fu * (xl - xu)) / (fu - fl)

        fX: np.float64 | float = evaluate_one_variable(func, x)

        if x != np.float64(0.0):
            relative_error = abs((x - x_old) / x) * 100

        if abs(fl) < abs(fu):
            xl = x
            fl = fX
        else:
            xu = x
            fu = fX

    return x


if __name__ == "__main__":

    # Example usage
    func = "exp(-x) - x"  # Example function
    xl = np.float64(0.0)
    xu = np.float64(1.0)
    tol = np.float64(1e-5)
    max_iter = 100

    print("Finding roots using bracketing methods...")
    print(f"Function: {func}")
    print(f"Interval: [{xl}, {xu}]")
    print(f"Tolerance: {tol}")
    print(f"Max iterations: {max_iter}\n")

    root_bissection = bissection(func, xl, xu, tol, max_iter)
    print(f"Root found using bisection method: {root_bissection}")

    root_false_position = false_position(func, xl, xu, tol, max_iter)
    print(f"Root found using false position method: {root_false_position}")
