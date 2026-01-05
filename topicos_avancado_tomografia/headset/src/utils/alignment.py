import numpy as np


def displace_cont(x, b):
    """
    Displace the array by a continuous offset.

    Parameters
    ----------
    x : numpy.ndarray
        The input array to displace.
    b : float
        The displacement value.

    Returns
    -------
    numpy.ndarray
        The displaced array.
    """
    x_displaced = np.zeros(x.shape)

    # Percorre cada posição do array
    for i in range(len(x)):
        # Calcula a posição deslocada (pode ser não-inteira)
        t = i + b
        t1 = np.floor(t)  # Posição inferior (arredonda para baixo)
        t2 = np.ceil(t)  # Posição superior (arredonda para cima)
        w = t - t1  # Peso para interpolação linear

        # Obtém valores nas posições inferior e superior (com verificação de limites)
        x1 = x[int(t1)] if (int(t1) < len(x)) and (int(t1) >= 0) else 0.0
        x2 = x[int(t2)] if (int(t2) < len(x)) and (int(t2) >= 0) else 0.0

        # Interpolação linear entre x1 e x2
        x_val = (1.0 - w) * x1 + w * x2
        x_displaced[i] = x_val

    return x_displaced


def find_error(d_value, pf1, pf2):
    """
    Compute the error between two profiles, with a displacement applied to one of them.

    Parameters
    ----------
    d_value : float
        Displacement value.
    pf1 : numpy.ndarray
        First profile array.
    pf2 : numpy.ndarray
        Second profile array.

    Returns
    -------
    float
        The mean squared error between the profiles.
    """
    # Aplica deslocamento contínuo ao segundo perfil
    f2_ = displace_cont(pf2, d_value)

    # Remove bordas afetadas pelo deslocamento para comparação justa
    delta = int(abs(d_value))
    cf1 = pf1[delta : pf1.shape[0] - delta]
    cf2 = f2_[delta : f2_.shape[0] - delta]

    # Calcula erro quadrático médio entre os perfis
    e = np.mean((cf1 - cf2) ** 2)
    return e


def find_min_pos(pf1, pf2, scan):
    """
    Find the position of minimum error between two profiles.

    Parameters
    ----------
    pf1 : numpy.ndarray
        First profile array.
    pf2 : numpy.ndarray
        Second profile array.
    scan : int
        Range of displacements to scan.

    Returns
    -------
    float
        The displacement value that minimizes the error.
    """
    e1 = 1e12  # Inicializa com valor muito alto
    min_val_disp = 0

    # Varre deslocamentos de -scan até +scan
    for t in np.linspace(-scan, scan, 2 * scan + 1):
        e = find_error(t, pf1, pf2)
        # Atualiza se encontrar erro menor
        if e < e1:
            e1 = e
            min_val_disp = t
    return min_val_disp


def divide_sino_line(sino_p):
    """
    Divide a sinogram line into two halves.

    Parameters
    ----------
    sino_p : numpy.ndarray
        The input sinogram profile.

    Returns
    -------
    tuple of numpy.ndarray
        Two halves of the sinogram, with the second half reversed.
    """
    # Soma o sinograma para criar perfil 1D
    _F = np.sum(np.sum(sino_p, axis=2), axis=1)
    M = _F.shape[0]

    # Divide o perfil ao meio
    _F1 = _F[: M // 2]  # Primeira metade (0° a 180°)
    _F2 = _F[M // 2:]   # Segunda metade (180° a 360°)
    _F2 = _F2[::-1]     # Inverte a segunda metade

    return _F1, _F2


def extract_width_displacement(sino_p, scan=20):
    """
    Extract the width displacement from a sinogram.

    Parameters
    ----------
    sino_p : numpy.ndarray
        The input sinogram profile.
    scan : int, optional
        The range of displacements to scan.

    Returns
    -------
    float
        The width displacement value.
    """
    # Divide sinograma em duas metades
    F1, F2 = divide_sino_line(sino_p)
    print("Divide Sino", F1.shape, F2.shape)

    # Encontra deslocamento que minimiza diferença entre as metades
    PosMin = find_min_pos(F1, F2, scan)
    return PosMin


def detect_and_fix_misalignment(sino, scan):
    """
    Detect and fix misalignment in a sinogram - CEPEDI standard procedure.

    Parameters
    ----------
    sino : numpy.ndarray
        The input sinogram.
    scan : int
        Initial scan range for detecting misalignment.

    Returns
    -------
    numpy.ndarray
        The corrected sinogram.
    """
    Result = scan

    # Aumenta range de busca até encontrar desalinhamento ou atingir limite
    while (abs(Result) == scan) and (scan < 500):
        Result = extract_width_displacement(sino, scan)
        if abs(Result) == scan:
            scan = 2 * scan  # Dobra o range de busca
            Result = scan

    print("Misalignment found: ", Result)

    # Se não há desalinhamento, retorna sinograma original
    if Result == 0:
        print("No misalignment detected, returning original sinogram")
        return sino
    else:
        print("Correcting misalignment...")

        # Calcula correção (metade do deslocamento encontrado)
        posMiss = int(abs(Result) // 2)
        if Result > 0:
            tmpstr = "left side."
        else:
            tmpstr = "right side."
            posMiss = -posMiss
        print(f"Shifting {posMiss} pixels to the {tmpstr}")

        # Aplica correção circular ao sinograma
        sino = np.roll(sino, posMiss, axis=0)
        print("Misalignment correction finished")
        return sino