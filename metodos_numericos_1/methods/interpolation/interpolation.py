import sys
import os

from numpy.typing import NDArray
import numpy as np
from scipy.special import factorial
import sympy as sy
from typing import Callable

# Adicionar o diretório raiz do projeto ao Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.parser import (
    parser_to_sy_expr,
    evaluate_one_variable_vector,
    evaluate_one_variable,
)


def taylor_interpolation(func: str | Callable, x0, xi, order):
    """
    Expansão em série de Taylor para uma função f usando SymPy.

    Args:
        func: Função a ser expandida, pode ser uma string ou uma função callable
        x0: Ponto onde a expansão é centrada
        xi: Ponto onde queremos avaliar a função
        order: Ordem da expansão

    Returns:
        Valor aproximado de f(xi)
    """
    x = sy.Symbol("x")

    if isinstance(func, str):
        # Converter a string da função em uma expressão simbólica
        f_expr = parser_to_sy_expr(func)
    elif callable(func):
        # Se for uma função, converte para expressão simbólica
        try:
            f_expr = func(x)  # Tenta usar a função diretamente com símbolo SymPy
        except Exception:
            raise ValueError(
                "A função callable deve aceitar símbolos SymPy como entrada."
            )
    else:
        raise ValueError("A função deve ser uma string ou uma função callable.")

    # Calcular as derivadas simbolicamente
    derivatives = [f_expr]
    for k in range(1, order + 1):
        derivatives.append(sy.diff(derivatives[-1], x))

    # Calcular a expansão de Taylor
    h = xi - x0
    result = float(derivatives[0].subs(x, x0))  # type: ignore # Termo inicial f(x0)

    for k in range(1, order + 1):
        # Adiciona o termo (h^k/k!) * f^(k)(x0)
        term = (h**k / factorial(k)) * float(derivatives[k].subs(x, x0))  # type: ignore
        result += term

    return result


def lagrange_interpolation(
    pointwise_matrix: NDArray | list, xi: np.float64 | float
) -> np.float64 | float:
    """
    Interpolação de Lagrange para um conjunto de pontos.
    Args:
        pointwise_matrix: Matriz 2D onde a primeira linha contém os valores de x e a segunda linha contém os valores de y
        xi: Ponto onde queremos avaliar o polinômio interpolador
    Returns:
        Valor do polinômio interpolador de Lagrange em xi
    """
    if isinstance(pointwise_matrix, list):
        pointwise_matrix = np.array(pointwise_matrix)

    # Obter os valores x e y separadamente para melhor legibilidade
    x_values = pointwise_matrix[0, :]
    y_values = pointwise_matrix[1, :]
    n = x_values.size - 1

    # Verificar duplicatas nos pontos x (causariam divisão por zero)
    if len(set(x_values)) != len(x_values):
        raise ValueError(
            "Pontos x duplicados encontrados. A interpolação de Lagrange requer pontos distintos."
        )

    result = np.float64(0.0)

    for i in range(n + 1):
        # Calcular o polinômio de base de Lagrange L_i(x)
        L_i = np.float64(1.0)

        for j in range(n + 1):
            if j != i:
                L_i *= (xi - x_values[j]) / (x_values[i] - x_values[j])

        # Acumular a contribuição deste termo
        result += L_i * y_values[i]

    return result


def cubic_spline_coefficients(pointwise_matrix: NDArray | list):
    """
    Calcula os coeficientes para splines cúbicos naturais.

    Args:
        pointwise_matrix: Matriz 2D onde a primeira linha contém os valores de x
                         e a segunda linha contém os valores de y

    Returns:
        Tupla de arrays (a, b, c, d) com os coeficientes para cada segmento
    """
    if isinstance(pointwise_matrix, list):
        pointwise_matrix = np.array(pointwise_matrix)

    # Extrair valores x e y
    x = pointwise_matrix[0, :]
    f = pointwise_matrix[1, :]
    n = len(x) - 1  # Número de segmentos

    # Passo 1: Calcular os valores a (são simplesmente os valores de f)
    a = f[:-1]  # a_j = f_j para j=0,...,n-1

    # Passo 2: Calcular os valores c resolvendo o sistema de equações
    # Primeiro, calcular os tamanhos dos intervalos h_j = x_{j+1} - x_j
    h = np.diff(x)

    # Montar a matriz tridiagonal e o vetor do lado direito
    A = np.zeros((n + 1, n + 1))
    r = np.zeros(n + 1)

    # Primeira e última linha para splines naturais (c_0 = c_n = 0)
    A[0, 0] = 1
    A[n, n] = 1

    # Equações para pontos internos
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

        r[i] = 3 * ((f[i + 1] - f[i]) / h[i] - (f[i] - f[i - 1]) / h[i - 1])

    # Resolver o sistema para obter os valores c
    c = np.linalg.solve(A, r)

    # Passo 3: Calcular os valores b usando a fórmula
    b = np.zeros(n)
    for j in range(n):
        b[j] = (f[j + 1] - f[j]) / h[j] - h[j] * (2 * c[j] + c[j + 1]) / 3

    # Passo 4: Calcular os valores d usando a fórmula
    d = np.zeros(n)
    for j in range(n):
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    # Retornar os coeficientes para cada segmento
    return a, b, c[:-1], d


def evaluate_cubic_spline(x_value, x_knots, coefficients):
    """
    Avalia um spline cúbico em um ponto específico.

    Args:
        x_value: Ponto onde avaliar o spline
        x_knots: Pontos de interpolação (nós)
        coefficients: Tupla (a, b, c, d) com os coeficientes

    Returns:
        Valor do spline no ponto x_value
    """
    a, b, c, d = coefficients

    # Encontrar o segmento correto
    if x_value <= x_knots[0]:
        j = 0
    elif x_value >= x_knots[-1]:
        j = len(x_knots) - 2
    else:
        j = np.searchsorted(x_knots, x_value) - 1

    # Calcular o valor do spline usando a fórmula
    dx = x_value - x_knots[j]
    return a[j] + b[j] * dx + c[j] * dx**2 + d[j] * dx**3


def manual_cubic_spline_interpolation(
    pointwise_matrix: NDArray | list, xi: np.float64 | float
) -> np.float64 | float:
    """
    Interpolação por splines cúbicos implementada manualmente.

    Args:
        pointwise_matrix: Matriz de pontos onde a primeira linha contém valores x
                         e a segunda linha contém valores y
        xi: Ponto onde queremos avaliar o spline

    Returns:
        Valor interpolado no ponto xi
    """
    if isinstance(pointwise_matrix, list):
        pointwise_matrix = np.array(pointwise_matrix)

    x_values = pointwise_matrix[0, :]

    # Verificar se os pontos são distintos
    if len(set(x_values)) != len(x_values):
        raise ValueError(
            "Pontos x duplicados encontrados. A interpolação por splines cúbicos requer pontos distintos."
        )

    # Calcular os coeficientes
    coefficients = cubic_spline_coefficients(pointwise_matrix)

    # Avaliar o spline no ponto xi
    return evaluate_cubic_spline(xi, x_values, coefficients)


def regression_multiple_least_squares(x: NDArray, y: NDArray) -> NDArray:
    """
    Regressão linear múltipla por mínimos quadrados: y = a_0 + a_1*x_1 + a_2*x_2 + ... + a_n*x_n

    Args:
        x: Array 2D com valores independentes (cada coluna representa uma variável)
        y: Array com valores dependentes

    Returns:
        Array com os coeficientes da regressão (termo independente seguido dos coeficientes de cada variável)
    """
    # Verificar dimensões de entrada
    if x.ndim == 1:
        x = x[:, np.newaxis]  # Transformar em 2D se necessário

    # Verificar compatibilidade de dimensões
    if len(x) != len(y):
        raise ValueError("Os arrays x e y devem ter o mesmo número de observações")

    # Adicionar coluna de uns para o termo independente
    X = np.hstack((np.ones((x.shape[0], 1)), x))

    # Calcular os coeficientes usando método mais estável numericamente
    try:
        # Usar resolução direta de sistema linear em vez de inversa explícita
        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        # Fallback para método de pseudo-inversa se o sistema for mal condicionado
        coefficients = np.linalg.pinv(X) @ y

    return coefficients


if __name__ == "__main__":
    # Exemplo de uso da interpolação de Lagrange
    func = "1/x"
    x_values = [2, 2.5, 4]
    y_values = evaluate_one_variable_vector(func, x_values)
    points = np.array([x_values, y_values])  # Matriz 2D com pontos (x, y)

    xi = np.float64(2.2)  # Ponto onde queremos avaliar o polinômio interpolador
    result = lagrange_interpolation(points, xi)
    print(f"Valor do polinômio interpolador de Lagrange em x = {xi}: {result}")

    xi = np.float64(3.5)  # Ponto onde queremos avaliar o polinômio interpolador
    result = lagrange_interpolation(points, xi)
    print(f"Valor do polinômio interpolador de Lagrange em x = {xi}: {result}")

    # Exemplo de uso da expansão de Taylor
    func = "(1+x)**(1/2)"
    x0 = np.float64(0.0)  # Ponto de expansão
    xi = np.float64(2.5)  # Ponto onde queremos avaliar a função
    order = 3  # Ordem da expansão
    taylor_result = taylor_interpolation(func, x0, xi, order)
    print(
        f"Valor da expansão de Taylor de sin(x) em x = {xi} (centro em {x0}, ordem {order}): {taylor_result}"
    )

    # Exemplo de uso da interpolação por splines cúbicos
    x_values = np.array([0, 1, 2, 3])
    y_values = np.array([0, 1, 4, 9])
    points = np.array([x_values, y_values])  # Matriz 2D com pontos (x, y)

    func = "exp(x)"
    # points = np.array([x_values, evaluate_one_variable_vector(func, x_values)])
    coefficients = cubic_spline_coefficients(points)

    print("Coeficientes do spline cúbico:")
    print(f"a: {coefficients[0]}")
    print(f"b: {coefficients[1]}")
    print(f"c: {coefficients[2]}")
    print(f"d: {coefficients[3]}")

    xi = np.float64(1.5)  # Ponto onde queremos avaliar o spline
    spline_result = manual_cubic_spline_interpolation(points, xi)
    print(f"Valor do spline cúbico em x = {xi}: {spline_result}")
    print(f"Valor da função em x = {x_values}: {y_values}")
    print(f"Valor da função em x = {xi}: {evaluate_one_variable(func, xi)}")

    print(
        f"Erro da interpolação por splines cúbicos em x = {xi}: {abs(spline_result - evaluate_one_variable(func, xi))}"
    )
