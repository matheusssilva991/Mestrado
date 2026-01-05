import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        """
        Armazena média e desvio padrão por feature.
        X: matriz (n_amostras, n_features)
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Você deve chamar 'fit' antes de normalizar.")
        return (X - self.mean) / self.std

    def denormalize(self, X_std: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Você deve chamar 'fit' antes de desnormalizar.")
        return X_std * self.std + self.mean

    def desnormalize_weights(self, w: np.ndarray) -> np.ndarray:
        """
        Converte pesos de um modelo linear treinado em dados padronizados
        para a escala original.
        w = [w1, w2, ..., bias]
        """
        if self.mean is None or self.std is None:
            raise ValueError("Você deve chamar 'fit' antes de desnormalizar pesos.")

        w_no_bias = w[:-1]
        b = w[-1]

        w_orig = w_no_bias / self.std
        b_orig = b - np.sum((w_no_bias * self.mean) / self.std)

        return np.concatenate([w_orig, [b_orig]])
