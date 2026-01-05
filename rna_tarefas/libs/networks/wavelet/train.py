import numpy as np
from .loss import make_mse_loss_for_network, make_residuals_fn
from .gradients import make_jacobian_fn
from activations.activations import mexican_hat_wavelet, mexican_hat_wavelet_derivative
from optimizers.levenberg_marquadt import levenberg_marquadt
from .initialization import weights_initialization_wavelet


def optimize_network_weights(*features, y, **kwargs):
    """
    Otimiza pesos da rede wavelon.
    """
    n_neurons = kwargs.get("n_neurons", 10)
    activation_fn = kwargs.get("activation_fn", mexican_hat_wavelet)
    n_iterations = kwargs.get("n_iterations", 1000)
    tolerance = kwargs.get("tolerance", 1e-6)
    alpha = kwargs.get("alpha", 1e-3)
    alpha_variability = kwargs.get("alpha_variability", 10)
    stopping_criteria = kwargs.get("stopping_criteria", [1, 3])

    n_inputs = len(features)
    total_params = n_neurons * 2 * n_inputs + n_neurons + 1

    initial_weights = kwargs.get(
        "initial_weights",
        np.random.uniform(-1, 1, size=total_params),
    )

    loss_function = make_mse_loss_for_network(
        *features, Y=y, wavelet_fn=activation_fn, n_neurons=n_neurons
    )

    residuals_fn = make_residuals_fn(
        *features, Y=y, n_neurons=n_neurons, wavelet_fn=activation_fn
    )

    jacobian_fn = make_jacobian_fn(
        *features,
        n_neurons=n_neurons,
        wavelet_fn=activation_fn,
        wavelet_deriv=mexican_hat_wavelet_derivative,
    )

    neurons_weights, losses, n_iters = levenberg_marquadt(
        initial_weights,
        residuals_fn,
        loss_function,
        jacobian_fn,
        alpha=alpha,
        alpha_variability=alpha_variability,
        max_iter=n_iterations,
        tolerance=tolerance,
        stopping_criteria=stopping_criteria,
    )

    return neurons_weights, losses, n_iters


def train_network_wavelet(*features, y, **kwargs):
    """
    Treina rede wavelet para qualquer número de features.
    """
    n_neurons = kwargs.get("n_neurons", 10)
    activation_fn = kwargs.get("activation_fn", mexican_hat_wavelet)
    n_iterations = kwargs.get("n_iterations", 1000)
    tolerance = kwargs.get("tolerance", 1e-6)
    alpha = kwargs.get("alpha", 1e-3)
    weights_init_method = kwargs.get("weights_init_method", "heuristic")
    alpha_variability = kwargs.get("alpha_variability", 10)
    stopping_criteria = kwargs.get("stopping_criteria", [1, 3])

    # Inicializar pesos
    initial_weights = weights_initialization_wavelet(
        *features,
        y=y,
        n_neurons=n_neurons,
        method=weights_init_method,
        wavelet_fn=activation_fn
    )

    neurons_weights, losses, n_iters = optimize_network_weights(
        *features,
        y=y,
        n_neurons=n_neurons,
        activation_fn=activation_fn,
        n_iterations=n_iterations,
        tolerance=tolerance,
        alpha=alpha,
        initial_weights=initial_weights,
        alpha_variability=alpha_variability,
        stopping_criteria=stopping_criteria,
    )

    return neurons_weights[-1], losses, n_iters
