import numpy as np

def unflatten_weights(weights_flat, n_inputs, n_neurons):
    """Reconstrói os pesos da rede a partir de um vetor achatado."""
    neurons_weights = []
    idx = 0

    for _ in range(n_neurons):
        w_hidden = weights_flat[idx:idx + (n_inputs + 1)]
        neurons_weights.append(w_hidden)
        idx += n_inputs + 1

    w_output = weights_flat[idx:]
    neurons_weights.append(w_output)

    return neurons_weights
