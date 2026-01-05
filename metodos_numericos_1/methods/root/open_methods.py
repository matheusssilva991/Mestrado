import sys
import os

# Adicionar o diretório raiz do projeto ao Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from utils.parser import evaluate_one_variable
from typing import Union, Callable
from utils.error import is_indertemination
from utils.parser import get_derivative

def fixed_point(
    func: Union[str, Callable], x0: np.float64 | float, tol: np.float64 | float, max_iter: int
) -> np.float64 | float:
    """
    Método do Ponto Fixo para encontrar raízes de uma função.

    Parâmetros:
        func (Union[str, Callable]): A função a ser avaliada.
        x0 (np.float64): O ponto inicial.
        tol (np.float64): A tolerância para o erro relativo.
        max_iter (int): O número máximo de iterações.

    Retorna:
        - A raiz encontrada (np.float64)
        - O último valor de x (np.float64)
        - O erro relativo (np.float64)
        - O número de iterações (int)
    """

    iter: int = 0
    relative_error: np.float64 | float = np.float64(100.0)
    x: np.float64 | float = x0
    x_old: np.float64 | float = x

    while (relative_error > tol) and (iter < max_iter):
        x_old: np.float64 | float = x
        iter = iter + 1

        x = evaluate_one_variable(func, x_old)

        if x != np.float64(0.0):
            relative_error = abs((x - x_old) / x) * 100

    return x


def newton_raphson(
    func: Union[str, Callable],
    derivative: Union[str, Callable],
    x0: np.float64 | float,
    tol: np.float64 | float,
    max_iter: int,
) -> np.float64 | float:
    """
    Método de Newton-Raphson para encontrar raízes de uma função.

    Parâmetros:
        func (Union[str, Callable]): A função a ser avaliada.
        derivative (Union[str, Callable]): A derivada da função.
        x0 (np.float64): O ponto inicial.
        tol (np.float64): A tolerância para o erro relativo.
        max_iter (int): O número máximo de iterações.

    Retorna:
        - A raiz encontrada (np.float64)
        - O último valor de x (np.float64)
        - O erro relativo (np.float64)
        - O número de iterações (int)
    """

    iter: int = 0
    relative_error: np.float64 | float = np.float64(100.0)
    x: np.float64 | float = x0

    while (relative_error > tol) and (iter < max_iter):
        x0 = x
        iter = iter + 1

        x = x0 - (
            evaluate_one_variable(func, x0) / evaluate_one_variable(derivative, x0)
        )

        if x != np.float64(0.0):
            relative_error = abs((x - x0) / x) * 100

    return x


def secant(
    func: Union[str, Callable],
    x0: np.float64 | float,
    x1: np.float64 | float,
    tol: np.float64 | float,
    max_iter: int,
) -> np.float64 | float:
    """
    Método da Secante para encontrar raízes de uma função.

    Parâmetros:
        func (Union[str, Callable]): A função a ser avaliada.
        x0 (np.float64): O ponto inicial.
        x1 (np.float64): O segundo ponto inicial.
        tol (np.float64): A tolerância para o erro relativo.
        max_iter (int): O número máximo de iterações.

    Retorna:
        - A raiz encontrada (np.float64)
        - O último valor de x0 (np.float64)
        - O último valor de x1 (np.float64)
        - O erro relativo (np.float64)
        - O número de iterações (int)
    """

    relative_error: np.float64 | float = np.float64(100.0)
    iter: int = 0
    x: np.float64 | float= np.float64(0.0)

    while (relative_error > tol) and (iter < max_iter):
        iter = iter + 1

        f0: np.float64 | float = evaluate_one_variable(func, x0)
        f1: np.float64 | float = evaluate_one_variable(func, x1)

        if is_indertemination(f0, f1):
            print("Indetermination encountered, returning infinity.")
            return np.inf

        x: np.float64 | float = x1 - f1 * ((x0 - x1) / (f0 - f1))

        if x != np.float64(0.0):
            relative_error = abs((x - x1) / x) * 100

        x0 = x1
        x1 = x

    return x


if __name__ == "__main__":

    # Example usage
    func = "exp(-x) - x"  # Example function
    func_g = "exp(-x)"  # Rearranged function for fixed point method
    derivative = get_derivative(func)  # Derivative of the function
    xl = np.float64(0.0)
    xu = np.float64(1.0)
    tol = np.float64(1e-5)
    max_iter = 100

    print("Finding roots using open methods...")
    print(f"Function: {func}")
    print(f"Interval: [{xl}, {xu}]")
    print(f"Tolerance: {tol}")
    print(f"Max iterations: {max_iter}\n")

    root_fixed_point = fixed_point(func_g, (xl + xu) / 2, tol, max_iter)
    print(f"Root found using fixed point method: {root_fixed_point}")

    root_newton_raphson = newton_raphson(func, derivative, (xl + xu) / 2, tol, max_iter)
    print(f"Root found using Newton-Raphson method: {root_newton_raphson}")

    root_secant = secant(func, xl, xu, tol, max_iter)
    print(f"Root found using secant method: {root_secant}")
