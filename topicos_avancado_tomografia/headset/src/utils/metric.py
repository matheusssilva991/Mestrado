import numpy as np


def calculate_snr(signal_roi: np.ndarray, noise_roi: np.ndarray) -> float:
    """
    Calcula o SNR (Signal-to-Noise Ratio) entre uma ROI de sinal e uma ROI de ruído.

    O SNR quantifica a qualidade da reconstrução comparando a potência do sinal
    (região de interesse no objeto) com a potência do ruído (região de fundo/background).

    Formula: SNR = P_signal / P_noise

    onde:
        - P_signal = mean(signal_roi²)  → potência média do sinal
        - P_noise  = var(noise_roi)     → variância do ruído de fundo

    Parameters
    ----------
    signal_roi : np.ndarray
        Região de interesse contendo o sinal (objeto/material reconstruído).
    noise_roi : np.ndarray
        Região de interesse contendo apenas ruído (fundo/ar sem objeto).

    Returns
    -------
    float
        Valor do SNR (adimensional). Quanto maior, melhor a qualidade.
        Retorna inf se noise_roi tiver variância zero (ruído ausente).

    Notes
    -----
    - SNR linear: valores típicos variam de 10 a 1000+ para boas reconstruções.
    - Para converter para dB: SNR_dB = 10 × log10(SNR_linear)
    - Certifique-se que noise_roi está em uma região verdadeiramente vazia.

    Examples
    --------
    >>> signal_roi = volume[150:250, 100:200, 100:200]
    >>> noise_roi = volume[:, :50, :50]
    >>> snr = calculate_snr(signal_roi, noise_roi)
    >>> print(f"SNR: {snr:.2f}")
    """
    signal = np.asarray(signal_roi, dtype=np.float64)
    noise = np.asarray(noise_roi, dtype=np.float64)

    # Potência do sinal
    signal_power = np.mean(signal**2)

    # Potência do ruído
    noise_power = np.var(noise)

    if noise_power == 0:
        return float("inf")

    # SNR linear (razão direta)
    snr = signal_power / noise_power

    return float(snr)


def calculate_cnr(signal_roi: np.ndarray, background_roi: np.ndarray) -> float:
    """
    Calcula o CNR (Contrast-to-Noise Ratio) entre duas ROIs.

    O CNR quantifica a capacidade de distinguir um objeto de seu fundo,
    levando em conta tanto o contraste quanto o ruído presente na imagem.

    Formula: CNR = |μ_signal - μ_background| / σ_background

    onde:
        - μ_signal = mean(signal_roi)        → intensidade média do sinal
        - μ_background = mean(background_roi) → intensidade média do fundo
        - σ_background = std(background_roi)  → desvio padrão do fundo (ruído)

    Parameters
    ----------
    signal_roi : np.ndarray
        Região de interesse contendo o sinal (objeto/material reconstruído).
        Pode ser um subvolume 3D ou região 2D do volume completo.
    background_roi : np.ndarray
        Região de interesse contendo o fundo de referência (área próxima ao objeto).
        Usada para estimar o contraste e o nível de ruído.

    Returns
    -------
    float
        Valor do CNR (adimensional). Quanto maior, melhor a visibilidade do objeto.
        Retorna inf se background_roi tiver desvio padrão zero (ruído ausente).

    Notes
    -----
    - Em CT, CNR > 3-5 geralmente indica boa visibilidade do objeto.
    - CNR difere de SNR por considerar o contraste relativo entre objeto e fundo.
    - Ideal para avaliar capacidade de distinguir diferentes materiais/tecidos.

    Examples
    --------
    >>> signal_roi = volume[150:250, 100:200, 100:200]  # ROI no objeto
    >>> background_roi = volume[:, :50, :50]  # Região de fundo
    >>> cnr = calculate_cnr(signal_roi, background_roi)
    >>> print(f"CNR: {cnr:.2f}")
    """
    signal = np.asarray(signal_roi, dtype=np.float64)
    background = np.asarray(background_roi, dtype=np.float64)

    # Intensidades médias das duas regiões
    mean_signal = np.mean(signal)
    mean_background = np.mean(background)

    # Ruído do fundo
    std_background = np.std(background)

    if std_background == 0:
        return float("inf")

    # CNR: contraste normalizado pelo ruído
    cnr = abs(mean_signal - mean_background) / std_background

    return float(cnr)
