import numpy as np
from .loss import make_mse_loss_for_network, make_residuals_fn
from .gradients import make_jacobian_fn
from activations.activations import tanh_derivative
from optimizers.levenberg_marquadt import levenberg_marquadt


def optimize_network_weights(*features, y, **kwargs):
    """
    Otimiza os pesos da rede neural para qualquer número de features.

    Parâmetros
    ----------
    *features : np.ndarray
        Features de entrada (X1, X2, ..., Xn)
    y : np.ndarray
        Valores alvo
    **kwargs : dict
        Parâmetros adicionais (n_neurons, activation_fn, etc.)
    """
    n_neurons = kwargs.get("n_neurons", 10)
    activation_fn = kwargs.get("activation_fn", np.tanh)
    n_iterations = kwargs.get("n_iterations", 1000)
    tolerance = kwargs.get("tolerance", 1e-6)
    alpha = kwargs.get("alpha", 1e-3)

    n_inputs = len(features)
    total_params = n_neurons * (n_inputs + 1) + (n_neurons + 1)

    initial_weights = kwargs.get(
        "initial_weights",
        np.random.uniform(-1, 1, size=total_params),
    )

    # Criar funções usando *features
    loss_function = make_mse_loss_for_network(
        *features, Y=y, activation_fn=activation_fn, n_neurons=n_neurons
    )

    residuals_fn = make_residuals_fn(
        *features, Y=y, n_neurons=n_neurons, activation_fn=activation_fn
    )

    jacobian_fn = make_jacobian_fn(
        *features,
        n_neurons=n_neurons,
        activation_fn=activation_fn,
        activation_deriv=tanh_derivative,
    )

    neurons_weights, losses, n_iters = levenberg_marquadt(
        initial_weights,
        residuals_fn,
        loss_function,
        jacobian_fn,
        alpha=alpha,
        alpha_variability=10,
        max_iter=n_iterations,
        tolerance=tolerance,
        stopping_criteria=[1, 3],
    )

    return neurons_weights, losses, n_iters


def train_network(*features, y, **kwargs):
    """
    Treina a rede neural para qualquer número de features.

    Parâmetros
    ----------
    *features : np.ndarray
        Features de entrada (X1, X2, ..., Xn)
    y : np.ndarray
        Valores alvo
    **kwargs : dict
        Parâmetros adicionais (n_neurons, n_epochs, etc.)
    """
    n_neurons = kwargs.get("n_neurons", 10)
    activation_fn = kwargs.get("activation_fn", np.tanh)
    n_epochs = kwargs.get("n_epochs", 100)
    n_iterations_per_epoch = kwargs.get("n_iterations_per_epoch", 1000)
    tolerance = kwargs.get("tolerance", 1e-6)
    alpha = kwargs.get("alpha", 1e-3)

    n_inputs = len(features)

    # Inicializar pesos dinamicamente
    hidden_weights = [np.random.uniform(-1, 1, size=n_inputs + 1) for _ in range(n_neurons)]
    output_weights = np.random.uniform(-1, 1, size=n_neurons + 1)
    initial_weights = np.concatenate([w.flatten() for w in hidden_weights] + [output_weights.flatten()])

    all_losses = []
    total_iters = 0

    for _ in range(n_epochs):
        initial_weights, losses, n_iters = optimize_network_weights(
            *features,
            y=y,
            n_neurons=n_neurons,
            activation_fn=activation_fn,
            n_iterations=n_iterations_per_epoch,
            tolerance=tolerance,
            alpha=alpha,
            initial_weights=initial_weights,
        )
        initial_weights = initial_weights[-1]
        all_losses.extend(losses)
        total_iters += n_iters

    return initial_weights, all_losses, total_iters
