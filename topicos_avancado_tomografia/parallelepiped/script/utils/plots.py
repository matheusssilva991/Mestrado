import matplotlib.pyplot as plt
import numpy as np


def plot_image(image: np.ndarray, **kwargs) -> None:
    """
    Displays the given image using matplotlib.

    Parameters:
    - image: numpy array, the image data.
    - kwargs: additional keyword arguments for customization (e.g., cmap, title).
    """
    cmap = kwargs.get('cmap', 'gray')
    title = kwargs.get('title', 'Image')
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    fontsize = kwargs.get('fontsize', 12)

    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=fontsize)
    plt.axis('off')
    plt.show()


def plot_projection_mosaic(
    projections: np.ndarray,
    **kwargs
):
    """
    Exibe um mosaico de projeções de um array 3D, ajustando automaticamente a quantidade de subplots.

    Parâmetros:
    - projections: array 3D (nx, ny, n_proj)
    - kwargs: argumentos opcionais:
        - num_projections: número de projeções a serem exibidas (default=20)
        - step: passo entre as projeções (default=None, calcula espaçamento igual)
        - cmap: mapa de cores (default="gray")
        - figsize: tamanho da figura (default ajustado)
        - title: título do mosaico (default="Mosaico de Projeções", se None não exibe título)
        - fontsize: tamanho da fonte do título (default=12)
    """
    num_projections = kwargs.get('num_projections', 20)
    step = kwargs.get('step', None)
    cmap = kwargs.get('cmap', "gray")
    figsize = kwargs.get('figsize', None)
    title = kwargs.get('title', None)
    fontsize = kwargs.get('fontsize', 12)
    title_fontsize = kwargs.get('title_fontsize', 14)
    projection_titles = kwargs.get('projection_titles', 'Projeção')

    n_proj = projections.shape[2]

    if step is not None:
        indices = [i*step for i in range(num_projections) if i*step < n_proj]
        if not indices:
            indices = list(range(n_proj))  # Se não houver índices válidos, mostra todas
    else:
        # Espaçamento igual entre índices
        if num_projections >= n_proj:
            indices = list(range(n_proj))
        else:
            indices = np.linspace(0, n_proj-1, num_projections, dtype=int).tolist()

    n_imgs = len(indices)
    n_cols = min(5, n_imgs)
    n_rows = int(np.ceil(n_imgs / n_cols))

    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # Flatten para facilitar o acesso

    vmax = projections.max()
    vmin = projections.min()

    for idx, proj_idx in enumerate(indices):
        axes[idx].imshow(projections[:, :, proj_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        if projection_titles:
            axes[idx].set_title(f'{projection_titles} {proj_idx + 1}', fontsize=fontsize)
        axes[idx].axis('off')

    # Desativa e esconde subplots não usados
    for idx in range(n_imgs, len(axes)):
        axes[idx].axis('off')

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.show()


def plot_spectrum(spectrum: np.ndarray, energy_bins: np.ndarray, **kwargs) -> None:
    """
    Plota o espectro de energia com intensidade no eixo x e energia no eixo y.

    Parâmetros:
    - spectrum: array 1D, valores do espectro (intensidade).
    - energy_bins: array 1D, valores dos bins de energia.
    - kwargs: argumentos adicionais para personalização (e.g., title, xlabel, ylabel).
    """
    title = kwargs.get('title', 'Espectro de Energia')
    xlabel = kwargs.get('xlabel', 'Intensidade')
    ylabel = kwargs.get('ylabel', 'Energia (keV)')
    fontsize = kwargs.get('fontsize', 12)

    plt.plot(spectrum, energy_bins)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.grid()
    plt.show()


def plot_four_projections(projections, ids=[30, 60, 120, 170], cmap="gray", vmin=None, vmax=None):
    """
    Plota quatro projeções selecionadas de um array 3D.

    Parâmetros:
    - projections: array 3D (nx, ny, n_proj)
    - ids: lista dos índices das projeções a serem exibidas
    - cmap: mapa de cores
    - vmin, vmax: limites de intensidade para exibição
    """
    if len(ids) != 4:
        raise ValueError("A lista 'ids' deve conter exatamente quatro índices.")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, idx in enumerate(ids):
        axes[i].imshow(projections[:, :, idx], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Projeção {idx}', fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def plot_four_projections_by_angle(projections, **kwargs):
    """
    Plota quatro projeções selecionadas de um array 3D, escolhidas pelos ângulos, em formato 2x2.

    Parâmetros:
    - projections: array 3D (nx, ny, n_proj)
    - kwargs: argumentos adicionais, incluindo:
        - angles: lista dos ângulos das projeções a serem exibidas
        - cmap: mapa de cores
        - vmin, vmax: limites de intensidade para exibição
        - figsize: tamanho da figura
    """
    angles = kwargs.get('angles', [30, 60, 120, 170])
    cmap = kwargs.get('cmap', "gray")
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    figsize = kwargs.get('figsize', (8, 8))

    if len(angles) != 4:
        raise ValueError("A lista 'angles' deve conter exatamente quatro ângulos.")

    n_proj = projections.shape[2]
    idxs = [int(round(angle / 360 * n_proj)) % n_proj for angle in angles]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for i, (idx, angle) in enumerate(zip(idxs, angles)):
        row, col = divmod(i, 2)
        axes[row, col].imshow(projections[:, :, idx], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, col].set_title(f'Ângulo {angle}° (idx {idx})', fontsize=10)
        axes[row, col].axis('off')
    #plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.show()


def plot_two_sino_planes(projs, lines=[128, 400], vmin=0, vmax=None, cmap="gray", figsize=(12, 5)):
    """
    Plota duas linhas (planos) do sinograma.

    Parâmetros:
    - projs: array 3D (nx, ny, n_proj)
    - lines: lista com os índices das linhas a serem exibidas
    - vmin, vmax: limites de intensidade para exibição
    - cmap: mapa de cores
    - figsize: tamanho da figura
    """
    if vmax is None:
        vmax = projs.max()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for i, line in enumerate(lines):
        axes[i].imshow(projs[:, line, :], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axes[i].set_title(f'Linha {line}', fontsize=12)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


