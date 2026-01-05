import numpy as np
from .wavelon import wavelon

def hidden_forward(*features, neurons_params, wavelet_fn, output_weights):
    """
    Rede neural com camada oculta de neurônios Wavelon.

    Parâmetros
    ----------
    *features : np.ndarray
        Features de entrada
    neurons_params : list of dict
        Lista com parâmetros de cada neurônio {'translation': [...], 'scale': [...]}
    wavelet_fn : callable
        Função wavelet-mãe
    output_weights : np.ndarray
        Pesos da camada de saída (incluindo bias)
    """
    hidden_outputs = []

    for params in neurons_params:
        h = wavelon(
            *features,
            translations=params["translation"],
            scales=params["scale"],
            wavelet_fn=wavelet_fn,
        )
        hidden_outputs.append(h)

    H = np.column_stack(hidden_outputs)
    H = np.column_stack([H, np.ones(H.shape[0])])
    y_hat = H @ output_weights

    return y_hat
