import numpy as np


def triangular_membership(x, a, b, c):
    """
    Função de pertinência triangular.
    Parâmetros
    ----------
    x : np.ndarray
        Valores de entrada.
    a : float
        Ponto inicial do triângulo.
    b : float
        Ponto do pico do triângulo.
    c : float
        Ponto final do triângulo.
    Retorna
    -------
    mu : np.ndarray
        Valores de pertinência.
    """
    mu = np.zeros_like(x, dtype=float)

    # Crescente
    mask1 = (x >= a) & (x < b)
    mu[mask1] = (x[mask1] - a) / (b - a)

    # Decrescente
    mask2 = (x >= b) & (x <= c)
    mu[mask2] = (c - x[mask2]) / (c - b)

    return mu


def trapezoidal_membership(x, a, b, c, d):
    """
    Função de pertinência trapezoidal.
    Parâmetros
    ----------
    x : np.ndarray
        Valores de entrada.
    a : float
        Início da subida.
    b : float
        Início do topo.
    c : float
        Fim do topo.
    d : float
        Fim da descida.
    Retorna
    -------
    mu : np.ndarray
        Valores de pertinência.
    """
    mu = np.zeros_like(x, dtype=float)

    # Subida
    mask1 = (x >= a) & (x < b)
    mu[mask1] = (x[mask1] - a) / (b - a)

    # Topo
    mask2 = (x >= b) & (x <= c)
    mu[mask2] = 1.0

    # Descida
    mask3 = (x > c) & (x <= d)
    mu[mask3] = (d - x[mask3]) / (d - c)

    return mu


def gaussian_membership(x, c, sigma):
    """
    Função de pertinência gaussiana.
    Parâmetros
    ----------
    x : np.ndarray
        Valores de entrada.
    c : float
        Centro da função gaussiana.
    sigma : float
        Desvio padrão da função gaussiana.
    Retorna
    -------
    mu : np.ndarray
        Valores de pertinência.
    """
    mu = np.exp(-0.5 * ((x - c) / sigma) ** 2)
    return mu


def bell_membership(x, a, b, c):
    """
    Função de pertinência em sino (bell-shaped).
    Parâmetros
    ----------
    x : np.ndarray
        Valores de entrada.
    a : float
        Largura da função.
    b : float
        Inclinação da função.
    c : float
        Centro da função.
    Retorna
    -------
    mu : np.ndarray
        Valores de pertinência.
    """
    mu = 1 / (1 + np.abs((x - c) / a) ** (2 * b))
    return mu


def sigmoid_membership(x, a, c):
    """
    Função de pertinência sigmoidal.
    Parâmetros
    ----------
    x : np.ndarray
        Valores de entrada.
    a : float
        Inclinação da função.
    c : float
        Centro da função.
    Retorna
    -------
    mu : np.ndarray
        Valores de pertinência.
    """
    mu = 1 / (1 + np.exp(-a * (x - c)))
    return mu
