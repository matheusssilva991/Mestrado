import numpy as np
from .forward import hidden_forward
from .unflatten_weights import unflatten_weights
from activations.activations import mexican_hat_wavelet

def make_residuals_fn(*features, Y: np.ndarray, n_neurons=2, wavelet_fn=mexican_hat_wavelet):
    """
    Cria função de resíduos para qualquer número de features.
    """
    n_inputs = len(features)

    def residuals_fn(weights_flat) -> np.ndarray:
        neurons_params, output_weights = unflatten_weights(
            weights_flat, n_inputs=n_inputs, n_neurons=n_neurons
        )
        y_hat = hidden_forward(
            *features,
            neurons_params=neurons_params,
            wavelet_fn=wavelet_fn,
            output_weights=output_weights,
        )
        return Y - y_hat

    return residuals_fn


def make_mse_loss_for_network(*features, Y, n_neurons=2, wavelet_fn=mexican_hat_wavelet):
    """
    Cria função de perda MSE para qualquer número de features.
    """
    residuals_fn = make_residuals_fn(
        *features, Y=Y, n_neurons=n_neurons, wavelet_fn=wavelet_fn
    )

    def loss_fn(weights_flat):
        residuals = residuals_fn(weights_flat)
        return np.mean(residuals**2)

    return loss_fn
