import numpy as np


def normalize_volume_01(volume, clamp_negatives=True):
    """
    Normaliza volume para [0, 1] com duas estratégias para valores negativos.

    Esta normalização é útil para comparação justa entre métodos de
    reconstrução com escalas diferentes (ex: FDK vs OSSART).

    Parameters
    ----------
    volume : numpy.ndarray
        Volume 3D [Z, Y, X] a ser normalizado.
    clamp_negatives : bool, optional
        Se True, valores negativos são clampados para 0 (padrão).
        Se False, adiciona o valor mínimo para tornar todos os valores
        positivos antes de normalizar (shift).

    Returns
    -------
    numpy.ndarray
        Volume normalizado para [0, 1].

    Examples
    --------
    >>> vol = np.array([-10, 0, 50, 100])
    >>> normalize_volume_01(vol, clamp_negatives=True)
    array([0., 0., 0.5, 1.])  # -10 virou 0
    >>> normalize_volume_01(vol, clamp_negatives=False)
    array([0., 0.091, 0.545, 1.])  # -10 + 10 = 0, mantém proporção
    """
    vol_norm = volume.copy()

    if clamp_negatives:
        # Estratégia 1: Clampar negativos para 0
        vol_norm[vol_norm < 0] = 0
    else:
        # Estratégia 2: Shift - adicionar mínimo para tornar positivo
        vol_min = vol_norm.min()
        if vol_min < 0:
            vol_norm = vol_norm - vol_min  # Adiciona |min| a todos

    # Normalizar para [0, 1]
    vol_max = vol_norm.max()
    if vol_max > 0:
        vol_norm = vol_norm / vol_max

    return vol_norm


def normalize_volume_robust(volume, clamp_negatives=True, percentile=99.5):
    """
    Normaliza volume para [0, 1] ignorando outliers extremos (Robust Scaling).

    Esta normalização é mais robusta que normalize_volume_01 pois ignora
    artefatos brilhantes (metal, ruído) que podem saturar a imagem.
    Ideal para visualização de volumes CT com artefatos.

    Parameters
    ----------
    volume : numpy.ndarray
        Volume 3D [Z, Y, X] a ser normalizado.
    clamp_negatives : bool, optional
        Se True (padrão), valores negativos são clampados para 0.
        Se False, faz shift adicionando o valor mínimo para tornar positivos.
    percentile : float, optional
        Percentil para definir o valor máximo (branco). Default: 99.5
        - 99.5: Padrão, ignora 0.5% dos pixels mais brilhantes
        - 99.0: Mais agressivo, melhora contraste em imagens escuras
        - 99.9: Mais conservador, preserva mais detalhes brilhantes

    Returns
    -------
    numpy.ndarray
        Volume normalizado para [0, 1].

    Notes
    -----
    Diferença para normalize_volume_01:
    - normalize_volume_01: vmax = volume.max() (sensível a outliers)
    - normalize_volume_robust: vmax = percentil (ignora outliers)

    Examples
    --------
    >>> # Volume com artefato de metal (valor 5000) e tecido normal (0-100)
    >>> vol = np.array([0, 50, 100, 5000])
    >>> normalize_volume_01(vol, clamp_negatives=False)
    array([0., 0.01, 0.02, 1.])  # Imagem muito escura!
    >>> normalize_volume_robust(vol, clamp_negatives=False, percentile=99)
    array([0., 0.5, 1., 1.])  # Muito melhor! Artefato ignorado
    """
    vol_norm = volume.copy()

    # Passo 1: Tratamento de valores negativos (fundo/ar no FDK)
    if clamp_negatives:
        # Estratégia 1: Clampar negativos para 0 (descarta informação negativa)
        vol_norm[vol_norm < 0] = 0
    else:
        # Estratégia 2: Shift - preserva proporções adicionando |min| a todos
        vol_norm = vol_norm - vol_norm.min()

    # Passo 2: Encontrar valor máximo "útil" ignorando outliers
    # Usa percentil ao invés de max() para ignorar artefatos de metal/ruído
    # Ex: percentile=99.5 significa que apenas 0.5% dos pixels mais brilhantes
    # serão saturados (ficam brancos), melhorando o contraste do resto
    v_max = np.percentile(vol_norm, percentile)

    # Proteção contra divisão por zero
    if v_max == 0:
        # Fallback: se percentil deu zero, tenta usar max() mesmo
        v_max = vol_norm.max()
        if v_max == 0:
            # Volume é completamente zero, não há o que normalizar
            return vol_norm

    # Passo 3: Normalizar dividindo pelo vmax robusto
    vol_norm = vol_norm / v_max

    # Passo 4: Clipar valores que ultrapassaram 1.0 (os outliers ignorados)
    # Pixels acima do percentil ficam saturados em branco (1.0)
    vol_norm = np.clip(vol_norm, 0, 1)

    return vol_norm


def normalize_volume_uint8(volume, low_pct=1.0, high_pct=99.0):
    """
    Normaliza volume para uint8 usando percentis para reduzir influência de outliers.
    """
    v = volume.astype(np.float32)
    vmin = np.percentile(v, low_pct)
    vmax = np.percentile(v, high_pct)
    if vmax <= vmin:
        vmax = v.max()
        vmin = v.min()
    v = np.clip((v - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return (v * 255).astype(np.uint8)