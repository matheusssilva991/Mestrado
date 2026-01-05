import numpy as np
from activations.activations import mexican_hat_wavelet


def wavelon(*features, translations, scales, wavelet_fn=mexican_hat_wavelet):
    """
    Implementa um neurônio do tipo Wavelon.

    Parâmetros
    ----------
    *features : arrays
        Features de entrada (x1, x2, ..., xn)
    translations : array_like
        Centro da wavelet (um valor por feature)
    scales : array_like
        Escala da wavelet (um valor por feature)
    wavelet_fn : callable
        Função wavelet-mãe
    """
    X = np.column_stack([np.atleast_1d(f) for f in features])
    t = np.atleast_1d(translations)
    s = np.abs(np.atleast_1d(scales)) + 1e-8

    if t.shape[0] != X.shape[1]:
        raise ValueError(
            f"Dimensão do centro 't' ({t.shape[0]}) deve ser igual ao número de features ({X.shape[1]})."
        )

    if s.shape[0] != X.shape[1]:
        raise ValueError(
            f"Dimensão da escala 's' ({s.shape[0]}) deve ser igual ao número de features ({X.shape[1]})."
        )

    z = (X - t) / s
    psi_z = wavelet_fn(z)
    y = np.prod(psi_z, axis=1)

    return y
