import numpy as np
from .forward import hidden_forward
from .unflatten_weights import unflatten_weights

def make_residuals_fn(*features, Y: np.ndarray, n_neurons=2, activation_fn=np.tanh):
    n_inputs = len(features)

    def residuals_fn(weights_flat) -> np.ndarray:
        neurons_weights = unflatten_weights(weights_flat, n_inputs=n_inputs, n_neurons=n_neurons)
        y_hat = hidden_forward(*features, neurons_weights=neurons_weights, activation_fn=activation_fn)
        return Y - y_hat
    return residuals_fn


def make_mse_loss_for_network(*features, Y, n_neurons=2, activation_fn=np.tanh):
    residuals_fn = make_residuals_fn(*features, Y=Y, n_neurons=n_neurons, activation_fn=activation_fn)

    def loss_fn(weights_flat):
        residuals = residuals_fn(weights_flat)
        return np.mean(residuals**2)

    return loss_fn
