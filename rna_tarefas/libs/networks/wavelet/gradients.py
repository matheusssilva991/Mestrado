import numpy as np
from .unflatten_weights import unflatten_weights
from activations.activations import mexican_hat_wavelet, mexican_hat_wavelet_derivative

def make_jacobian_fn(
    *features,
    n_neurons=2,
    wavelet_fn=mexican_hat_wavelet,
    wavelet_deriv=mexican_hat_wavelet_derivative,
):
    """
    Cria função Jacobiana para rede Wavelon.
    """
    n_inputs = len(features)

    def jacobian_fn(weights_flat):
        neurons_params, output_weights = unflatten_weights(
            weights_flat, n_inputs=n_inputs, n_neurons=n_neurons
        )

        features_arrays = [np.atleast_1d(f) for f in features]
        X = np.column_stack(features_arrays)
        n_samples = X.shape[0]

        hidden_z = []
        hidden_h = np.zeros((n_samples, n_neurons))

        for j, params in enumerate(neurons_params):
            t_j = np.array(params["translation"])
            s_j = np.array(params["scale"])

            z_j = (X - t_j) / s_j
            psi_z_j = wavelet_fn(z_j)
            h_j = np.prod(psi_z_j, axis=1)

            hidden_h[:, j] = h_j
            hidden_z.append((z_j, psi_z_j, t_j, s_j))

        params_per_neuron = 2 * n_inputs
        total_params = n_neurons * params_per_neuron + len(output_weights)
        J = np.zeros((n_samples, total_params))

        idx = 0

        for j in range(n_neurons):
            z_j, psi_z_j, t_j, s_j = hidden_z[j]
            h_j = hidden_h[:, j]
            output_weight_j = output_weights[j]

            # Derivadas em relação a t_j
            for i in range(n_inputs):
                psi_zji = psi_z_j[:, i]
                psi_zji_derivative = wavelet_deriv(z_j[:, i])

                dh_dtji = h_j * (psi_zji_derivative / psi_zji) * (-1 / s_j[i])
                J[:, idx] = -output_weight_j * dh_dtji
                idx += 1

            # Derivadas em relação a s_j
            for i in range(n_inputs):
                psi_zji = psi_z_j[:, i]
                psi_zji_derivative = wavelet_deriv(z_j[:, i])

                dh_dsji = h_j * (psi_zji_derivative / psi_zji) * (-(X[:, i] - t_j[i]) / (s_j[i] ** 2))
                J[:, idx] = -output_weight_j * dh_dsji
                idx += 1

        # Derivadas em relação aos pesos de saída
        for j in range(n_neurons):
            J[:, idx] = -hidden_h[:, j]
            idx += 1

        J[:, idx] = -np.ones(n_samples)

        return J

    return jacobian_fn
