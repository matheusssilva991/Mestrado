import numpy as np


def centro_gravidade(valores, pertinencias):
    """Calcula o valor defuzzificado usando o método do centro de gravidade."""
    numerador = np.sum(valores * pertinencias)
    denominador = np.sum(pertinencias)
    if denominador == 0:
        return 0.0
    return numerador / denominador


def centro_maximos(valores, pertinencias):
    """Calcula o valor defuzzificado usando o método do centro máximo."""
    max_pertinencia = np.max(pertinencias)
    indices_valores_maximo = np.argwhere(pertinencias == max_pertinencia)
    conjunto_valores_maximo = valores[indices_valores_maximo].flatten()

    i = min(conjunto_valores_maximo)
    s = max(conjunto_valores_maximo)
    return (i + s) / 2


def media_maximos(valores, pertinencias):
    """Calcula o valor defuzzificado usando o método da média dos máximos."""
    max_pertinencia = np.max(pertinencias)
    indices_valores_maximo = np.argwhere(pertinencias == max_pertinencia)
    conjunto_valores_maximo = valores[indices_valores_maximo].flatten()

    return np.mean(conjunto_valores_maximo)


DEFUZZICADORES_DISPONIVEIS = {
    "centro_gravidade": centro_gravidade,
    "centro_maximos": centro_maximos,
    "media_maximos": media_maximos,
}