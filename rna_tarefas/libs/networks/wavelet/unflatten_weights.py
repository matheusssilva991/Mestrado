import numpy as np

def unflatten_weights(weights_flat, n_inputs, n_neurons):
    """
    Reconstrói parâmetros da rede a partir de um vetor achatado.

    Parâmetros
    ----------
    weights_flat : np.ndarray
        Vetor achatado com todos os parâmetros
    n_inputs : int
        Número de features de entrada
    n_neurons : int
        Número de neurônios wavelon

    Retorna
    -------
    neurons_params : list of dict
        Lista com parâmetros de cada neurônio
    output_weights : np.ndarray
        Pesos da camada de saída
    """
    neurons_params = []
    idx = 0

    for _ in range(n_neurons):
        translation = weights_flat[idx : idx + n_inputs]
        idx += n_inputs
        scale = weights_flat[idx : idx + n_inputs]
        idx += n_inputs

        neurons_params.append({"translation": translation, "scale": scale})

    output_weights = weights_flat[idx:]

    return neurons_params, output_weights
