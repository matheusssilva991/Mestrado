import numpy as np
from numpy.typing import NDArray
from typing import Callable


def linear_least_squares(x: NDArray, y: NDArray) -> NDArray:
    """
    Regressão linear por mínimos quadrados: y = a + bx

    Args:
        x: Array com valores independentes
        y: Array com valores dependentes

    Returns:
        Tuple (a, b) com os coeficientes da regressão linear
    """
    n = len(x)

    # Cálculo das somas
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)

    # Cálculo dos coeficientes
    a = (sum_y * sum_x2 - sum_x * sum_xy) / (n * sum_x2 - sum_x**2)
    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

    return np.array([a, b], dtype=np.float64)


def polynomial_least_squares(x: NDArray, y: NDArray, degree: int) -> NDArray:
    """
    Regressão polinomial por mínimos quadrados: y = a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n

    Args:
        x: Array com valores independentes
        y: Array com valores dependentes
        degree: Grau do polinômio

    Returns:
        Array com os coeficientes do polinômio (do termo independente ao termo de maior grau)
    """
    return np.polynomial.Polynomial.fit(x, y, degree).convert().coef


def regression_multiple_least_squares(x: NDArray, y: NDArray) -> NDArray:
    """
    Regressão linear múltipla por mínimos quadrados: y = a_0 + a_1*x_1 + a_2*x_2 + ... + a_n*x_n

    Args:
        x: Array 2D com valores independentes (cada coluna representa uma variável)
        y: Array com valores dependentes

    Returns:
        Array com os coeficientes da regressão (termo independente seguido dos coeficientes de cada variável)
    """
    # Garantir arrays NumPy
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 1:
        x = x[:, np.newaxis]  # transforma em matriz coluna

    if len(x) != len(y):
        raise ValueError("Os arrays x e y devem ter o mesmo número de observações")

    # Adiciona coluna de 1s para o termo independente
    X = np.hstack((np.ones((x.shape[0], 1)), x))

    try:
        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(X) @ y

    return coefficients


def get_regression_func(coefficients: NDArray) -> Callable[[float], float]:
    """
    Avalia o polinômio de regressão em um ponto específico.

    Args:
        coefficients: Coeficientes do polinômio (a_0, a_1, ..., a_n)
        x: Ponto onde avaliar o polinômio

    Returns:
        Valor do polinômio no ponto x
    """

    def regression_function(x: np.float64 | float) -> np.float64 | float:
        result = 0.0
        for i, coef in enumerate(coefficients):
            result += coef * (x**i)
        return result

    return regression_function


def get_regression_func_multiple(coefficients: NDArray) -> Callable[[NDArray], float]:
    """
    Avalia a regressão linear múltipla em um ponto específico.

    Args:
        coefficients: Coeficientes da regressão (a_0, a_1, ..., a_n)

    Returns:
        Função que avalia o modelo de regressão em um ponto específico
    """

    def regression_function(x: NDArray[np.float64] | list[float | np.float64 | int]) -> float:
        return float(np.dot(coefficients, np.hstack(([1], x))))

    return regression_function


def mean_squared_error(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> float:
    """
    Calcula o erro quadrático médio entre os valores reais e previstos.

    Args:
        y_true: Valores reais
        y_pred: Valores previstos

    Returns:
        Erro quadrático médio
    """
    return float(np.mean((y_true - y_pred) ** 2))


def r_squared(x: NDArray, y: NDArray, coefficients: NDArray) -> float:
    """
    Calcula o coeficiente de determinação (R²) da regressão.

    Args:
        x: Valores independentes
        y: Valores dependentes reais
        coefficients: Coeficientes do polinômio de regressão

    Returns:
        Coeficiente R²
    """
    y_mean = np.mean(y)
    regression_func = get_regression_func(coefficients)
    y_pred = np.array([regression_func(xi) for xi in x])

    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)

    return float(1 - (ss_residual / ss_total))


def correlation_coefficient(
    x: NDArray[np.float64], y: NDArray[np.float64], coefficients: NDArray[np.float64]
) -> float:
    """
    Calcula o coeficiente de correlação (r) entre os valores independentes e dependentes.

    Args:
        x: Valores independentes
        y: Valores dependentes
        coefficients: Coeficientes do polinômio de regressão

    Returns:
        Coeficiente de correlação r
    """
    r_squared_value = r_squared(x, y, coefficients)
    return np.sqrt(r_squared_value)


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    y = np.array([0.5, 2.5, 2, 4, 3.5, 6, 5.5])

    linear_coefficients = linear_least_squares(x, y)
    print("Linear Regression Coefficients:", linear_coefficients)
    func_linear = get_regression_func(linear_coefficients)

    r_2 = r_squared(x, y, linear_coefficients)
    print("R² for Linear Regression:", r_2)
    correlation = correlation_coefficient(x, y, linear_coefficients)
    print("Correlation Coefficient for Linear Regression:", correlation)

    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([2.10, 7.70, 13.60, 27.20, 40.90, 61.10])

    degree = 2
    polynomial_coefficients = polynomial_least_squares(x, y, degree)
    print(
        f"\nPolynomial Regression Coefficients (Degree {degree}):",
        polynomial_coefficients,
    )
    func_polynomial = get_regression_func(polynomial_coefficients)
    r_2_poly = r_squared(x, y, polynomial_coefficients)
    print(f"R² for Polynomial Regression (Degree {degree}):", r_2_poly)
    correlation_poly = correlation_coefficient(x, y, polynomial_coefficients)
    print(
        f"Correlation Coefficient for Polynomial Regression (Degree {degree}):",
        correlation_poly,
    )
