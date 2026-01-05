import numpy as np
from .neuron import neuron


def hidden_forward(*features, neurons_weights, activation_fn=np.tanh):
    """
    Rede neural simples com uma camada oculta de N neurônios e uma camada de saída.

    Parâmetros
    ----------
    *features : arrays
        Features de entrada (x1, x2, ..., xn)
    neurons_weights : list of np.ndarray
        Lista com pesos dos neurônios:
            [w1, w2, ..., w_hiddenN, w_output]
    activation_fn : callable
        Função de ativação (ex: np.tanh)

    Retorna
    -------
    y_hat : np.ndarray
        Saída final da rede.
    """

    hidden_outputs = [
        neuron(*features, weights=weights, activation_fn=activation_fn)
        for weights in neurons_weights[:-1]
    ]
    # Passa as saídas ocultas como features da camada de saída
    y_hat = neuron(*hidden_outputs, weights=neurons_weights[-1], activation_fn=activation_fn)

    return y_hat


