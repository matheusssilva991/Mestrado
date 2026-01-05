import numpy as np
from .unflatten_weights import unflatten_weights
from activations.activations import tanh_derivative

def make_jacobian_fn(*features, n_neurons=2, activation_fn=np.tanh, activation_deriv=tanh_derivative):
    """
    Cria uma função que calcula a Jacobiana da rede neural.

    Parâmetros
    ----------
    *features : np.ndarray
        Features de entrada (X1, X2, ..., Xn)
    n_neurons : int
        Número de neurônios na camada oculta
    activation_fn : callable
        Função de ativação
    activation_deriv : callable
        Derivada da função de ativação
    """
    n_inputs = len(features)

    def jacobian_fn(weights_flat):
        neurons_weights = unflatten_weights(weights_flat, n_inputs=n_inputs, n_neurons=n_neurons)

        # Empilhar todas as features e adicionar bias
        features_arrays = [np.atleast_1d(f) for f in features]
        X = np.column_stack(features_arrays + [np.ones_like(features_arrays[0])])
        n_samples = X.shape[0]

        num_neurons = len(neurons_weights) - 1
        w_out = neurons_weights[-1]
        hidden_z = []
        hidden_h = []

        # Forward pass na camada oculta
        for j in range(num_neurons):
            z_j = X @ neurons_weights[j]
            h_j = activation_fn(z_j)
            hidden_z.append(z_j)
            hidden_h.append(h_j)

        hidden_h = np.stack(hidden_h, axis=1)
        z_out = np.column_stack([hidden_h, np.ones(n_samples)]) @ w_out

        # Dimensões da Jacobiana
        total_params = num_neurons * X.shape[1] + len(w_out)
        J = np.zeros((n_samples, total_params))

        dy_dz_out = activation_deriv(z_out)
        idx = 0

        # Derivadas em relação aos pesos da camada oculta
        for j in range(num_neurons):
            output_weight_for_neuron_j = w_out[j]
            dz_hidden = activation_deriv(hidden_z[j])

            for i in range(X.shape[1]):
                if i < n_inputs:  # Features (não bias)
                    J[:, idx] = -dy_dz_out * output_weight_for_neuron_j * dz_hidden * X[:, i]
                else:  # Bias
                    J[:, idx] = -dy_dz_out * output_weight_for_neuron_j * dz_hidden
                idx += 1

        # Derivadas em relação aos pesos da camada de saída
        for j in range(num_neurons):
            J[:, idx] = -dy_dz_out * hidden_h[:, j]
            idx += 1

        # Derivada em relação ao bias da saída
        J[:, idx] = -dy_dz_out

        return J

    return jacobian_fn
