import numpy as np


def tanh(x):
    """Função de ativação tanh"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivada de tanh(x) = 1 - tanh(x)^2"""
    t = np.tanh(x)
    return 1 - t**2


def mexican_hat_wavelet(z):
    """Uma função wavelet-mãe 'Mexican Hat' real."""
    return (1 - z**2) * np.exp(-z**2 / 2)

def mexican_hat_wavelet_derivative(z):
    """Derivada da função wavelet-mãe 'Mexican Hat'."""
    return z * (z**2 - 3) * np.exp(-z**2 / 2)
