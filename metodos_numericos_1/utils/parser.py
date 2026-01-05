import numpy as np
import sympy as sy
from typing import Union, Callable, Any


def parser_to_sy_expr(function: str) -> sy.Expr:
    """
    Converte uma string de função em um objeto sympy.
    A função deve ser uma string que representa uma expressão matemática.
    Exemplo: "np.sin(x) + np.exp(x)".
    """
    function = function.replace("np.", "")

    local_dict = {
        # SymPy functions
        "exp": sy.exp,
        "sin": sy.sin,
        "cos": sy.cos,
        "tan": sy.tan,
        "log": sy.log,
        "ln": sy.ln,
        "sqrt": sy.sqrt,
        "pi": sy.pi,
        "E": sy.E,
        
        # NumPy functions
        "abs": np.abs,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "arctan2": np.arctan2,
        "exp2": np.exp2,
        "log2": np.log2,
        "log10": np.log10,
        "log1p": np.log1p,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "arcsinh": np.arcsinh,
        "arccosh": np.arccosh,
        "arctanh": np.arctanh,
        "floor": np.floor,
        "ceil": np.ceil,
        "round": np.round
    }
    parsed_function: sy.Expr = sy.parse_expr(function, local_dict=local_dict)
    return parsed_function


def evaluate_one_variable(
    function: Union[str, Callable], x0: np.float64 | float
) -> np.float64 | Any:
    """
    Avalia uma função de uma variável em um ponto específico.
    A função pode ser passada como uma string ou um callable.
    Se for uma string, deve conter a variável 'x'.
    """
    result: np.float64 | float | Any = np.float64(0.0)

    if isinstance(function, str):
        if "x" not in function:
            raise ValueError("A função deve conter a variável 'x'.")

        symbol_x: sy.Symbol = sy.Symbol("x")
        parsed_function: sy.Expr = parser_to_sy_expr(function)
        result = parsed_function.subs(symbol_x, x0).evalf(17)  # type: ignore

    elif isinstance(function, Callable):
        result = function(x0)
    else:
        raise ValueError("A função deve ser uma string ou um callable.")

    return np.float64(result)


def evaluate_one_variable_vector(
    function: Union[str, Callable], points: np.ndarray | list[np.float64 | float]
) -> np.ndarray:
    """
    Avalia uma função de uma variável em um vetor de pontos.
    A função pode ser passada como uma string ou um callable.
    Se for uma string, deve conter a variável 'x'.
    """

    if isinstance(points, list):
        points = np.array(points)
    if not isinstance(points, np.ndarray):
        raise TypeError("Os pontos devem ser um array numpy ou uma lista.")

    if isinstance(function, str):
        if "x" not in function:
            raise ValueError("A função deve conter a variável 'x'.")

        symbol_x: sy.Symbol = sy.Symbol("x")
        parsed_function: sy.Expr = parser_to_sy_expr(function)
        return sy.lambdify(symbol_x, parsed_function, "numpy")(points)

    elif isinstance(function, Callable):
        return function(points)

    else:
        raise ValueError("A função deve ser uma string ou um callable.")


def get_derivative(function: Union[str, Callable]) -> Callable:
    """
    Retorna a derivada de uma função de uma variável.
    A função pode ser passada como uma string ou um callable.
    Se for uma string, deve conter a variável 'x'.
    """
    if isinstance(function, str):
        symbol_x: sy.Symbol = sy.Symbol("x")
        parsed_function: sy.Expr = parser_to_sy_expr(function)
        derivative: sy.Expr = sy.diff(parsed_function, symbol_x)
        return sy.lambdify(symbol_x, derivative, "numpy")

    elif isinstance(function, Callable):
        return sy.lambdify(
            sy.Symbol("x"), sy.diff(sy.sympify(function), sy.Symbol("x")), "numpy"
        )

    else:
        raise ValueError("A função deve ser uma string ou um callable.")


def get_symbolic_function_and_derivative(function: str):
    """
    Retorna a função original e sua derivada como expressões simbólicas do SymPy.

    Parameters
    ----------
    function : str
        A função como string, contendo a variável 'x'.

    Returns
    -------
    tuple
        (função original como sy.Expr, derivada como sy.Expr)
    """
    if "x" not in function:
        raise ValueError("A função deve conter a variável 'x'.")

    symbol_x = sy.Symbol("x")
    parsed_function = parser_to_sy_expr(function)
    derivative = sy.diff(parsed_function, symbol_x)

    return parsed_function, derivative
