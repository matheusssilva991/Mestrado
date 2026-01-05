import matplotlib.pyplot as plt
import k3d
import numpy as np
import os
from .processing import normalize_volume_robust


def print_img_data(img, desc=None):
    """
    Print the details of an image after reading.

    Parameters
    ----------
    img : numpy.ndarray
        Image matrix.
    desc : str, optional
        Optional comment about the image.
    """
    if desc is not None:
        print(f"{desc}:: ", end="")
    print(
        f"Shape:{img.shape}, Min:{img.min():.4e}, Max:{img.max():.4e}, Mean:{img.mean():.4e}, Type:{img.dtype}"
    )


def plot_two_sino_planes(sino, lines=[100, 130], min_v=0, max_v=1):
    """
    Generate a mosaic with 2 selected planes (detector lines) from the sinogram.

    Parameters
    ----------
    sino : numpy.ndarray
        Sinogram with projections [angles, lines, columns].
    lines : list of int, optional
        Detector lines to display, default [100, 130] for a 200-line detector.
    min_v : float, optional
        Minimum value of the grayscale.
    max_v : float, optional
        Maximum value of the grayscale.
    """
    fig, ax = plt.subplot_mosaic([[0, 1]], figsize=(10, 6))

    ax[0].imshow(sino[:, lines[0], :], cmap="gray", vmin=min_v, vmax=max_v)
    ax[0].set_title(f"Linha {lines[0]}", fontsize=12)

    ax[1].imshow(sino[:, lines[1], :], cmap="gray", vmin=min_v, vmax=max_v)
    ax[1].set_title(f"Linha {lines[1]}", fontsize=12)


def generate_views(volume, mini=0, maxi=1000):
    """
    Generate three orthogonal views of a 3D volume.

    Parameters
    ----------
    volume : numpy.ndarray
        3D volume with shape [slices, height, width].
    mini : float, optional
        Minimum value for grayscale display.
    maxi : float, optional
        Maximum value for grayscale display.
    """
    # Selecionar a fatia central para análise
    num_slices, height, width = volume.shape
    print(volume.shape)

    fig, ax = plt.subplots(figsize=(15, 7))
    plt.imshow(volume[num_slices // 2, :, :], cmap="gray", vmin=mini, vmax=maxi)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 7))
    plt.imshow(volume[:, height // 2, :], cmap="gray", vmin=mini, vmax=maxi)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 7))
    plt.imshow(volume[:, :, width // 2], cmap="gray", vmin=mini, vmax=maxi)
    plt.show()


# Função para visualizar múltiplas projeções em mosaico
def fun_mosaic_prjs(
    prjs_data, step_p=20, nro_p=None, axes_fontsize=12, return_fig=False
):
    """
    Visualiza múltiplas projeções em mosaico.

    Parameters
    ----------
    prjs_data : numpy.ndarray
        Projeções com shape [num_projeções, altura, largura].
    step_p : int, optional
        Intervalo entre projeções exibidas (default: 20).
    nro_p : int, optional
        Quantidade máxima de projeções a considerar; por padrão usa prjs_data.shape[0].
    axes_fontsize : int, optional
        Tamanho da fonte dos títulos dos eixos.
    return_fig : bool, optional
        Se True, retorna a figura e os eixos criados.

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, numpy.ndarray
        Figura e array de eixos (se return_fig=True), caso contrário None.
    """
    if prjs_data.ndim != 3:
        raise ValueError(
            f"Esperado prjs_data 3D [N,H,W], recebido shape={prjs_data.shape}"
        )
    if prjs_data.shape[0] == 0:
        raise ValueError("prjs_data está vazio (N=0)")

    total = prjs_data.shape[0] if nro_p is None else min(nro_p, prjs_data.shape[0])
    num_panels = max(1, total // step_p)

    # Calcula layout balanceado (preferência por ~4-5 colunas)
    cols = min(4, num_panels)  # máximo 5 colunas
    rows = (num_panels + cols - 1) // cols  # divisão com arredondamento para cima

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    # Exibe projeções em intervalos regulares
    for i in range(num_panels):
        idx = i * step_p
        axes[i].imshow(prjs_data[idx, :, :], cmap="gray")
        axes[i].set_title(f"Projeção {idx}", fontsize=axes_fontsize)
        axes[i].axis("off")

    # Desativa eixos extras
    for j in range(num_panels, axes.size):
        axes[j].axis("off")

    fig.tight_layout()
    plt.show()
    return (fig, axes) if return_fig else None


def plot_k3d_volume(volume, alpha_coef=30, downsample=4, color_range=None):
    """
    Renderiza um volume 3D no K3D com downsampling e normalização.

    Parameters
    ----------
    volume : numpy.ndarray
        Volume 3D [Z, Y, X] ou [slices, height, width].
    alpha_coef : int
        Transparência do volume.
    downsample : int
        Fator de redução (default=4). Use 1 para desativar.
    color_range : list or tuple, optional
        Intervalo de cores [min, max]. Se None, usa [0, volume_max].

    Returns
    -------
    k3d.plot
        Objeto K3D com o volume renderizado.
    """
    volume_small = volume[::downsample, ::downsample, ::downsample]
    vol_norm = (volume_small - volume_small.min()) / (
        volume_small.max() - volume_small.min()
    )

    if color_range is None:
        color_range = [0.0, float(volume_small.max())]

    plot = k3d.plot()  # type: ignore
    vol_obj = k3d.volume(
        vol_norm.astype(np.float32),
        color_map=k3d.colormaps.matplotlib_color_maps.gray,  # type: ignore
        alpha_coef=alpha_coef,
        color_range=color_range,
    )
    plot += vol_obj
    plot.display()

    return plot


def save_k3d_volume(volume, output_path, alpha_coef=30, downsample=4, color_range=None):
    """
    Salva um volume 3D como arquivo HTML usando K3D.

    Parameters
    ----------
    volume : numpy.ndarray
        Volume 3D [Z, Y, X] ou [slices, height, width].
    output_path : str
        Caminho completo para salvar o arquivo HTML (ex: 'volume.html').
    alpha_coef : int
        Transparência do volume (default=30).
    downsample : int
        Fator de redução (default=4). Use 1 para desativar.
    color_range : list or tuple, optional
        Intervalo de cores [min, max]. Se None, usa [0, volume_max].
    """
    volume_small = volume[::downsample, ::downsample, ::downsample]
    vol_norm = (volume_small - volume_small.min()) / (
        volume_small.max() - volume_small.min()
    )

    if color_range is None:
        color_range = [0.0, float(volume_small.max())]

    plot = k3d.plot()  # type: ignore
    vol_obj = k3d.volume(
        vol_norm.astype(np.float32),
        color_map=k3d.colormaps.matplotlib_color_maps.gray,  # type: ignore
        alpha_coef=alpha_coef,
        color_range=color_range,
    )
    plot += vol_obj

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(plot.get_snapshot())

    print(f"Volume salvo em: {output_path}")


def compare_volume_slices(
    volume1,
    volume2,
    slices=None,
    axis=2,
    titles=["Volume 1", "Volume 2"],
    save_path=None,
    clamp_negatives=True,
    transpose_layout=False,
):
    """
    Compara fatias de dois volumes lado a lado com normalização automática.

    Os volumes são automaticamente normalizados para [0, 1] permitindo
    comparação justa entre métodos com escalas diferentes (ex: FDK vs OSSART).

    Parameters
    ----------
    volume1 : numpy.ndarray
        Primeiro volume 3D [Z, Y, X].
    volume2 : numpy.ndarray
        Segundo volume 3D [Z, Y, X].
    slices : list of int, optional
        Índices das fatias a visualizar. Se None, seleciona 3 fatias
        distribuídas uniformemente (início, meio, fim).
    axis : int, optional
        Eixo ao longo do qual cortar (0=axial, 1=coronal, 2=sagital).
        Default: 2 (sagital).
    titles : list of str, optional
        Títulos para cada volume. Default: ["Volume 1", "Volume 2"].
    save_path : str, optional
        Caminho para salvar a figura (ex: 'comparison.png').
    clamp_negatives : bool, optional
        Se True (padrão), valores negativos são clampados para 0.
        Se False, faz shift adicionando o valor mínimo para tornar positivos.
    transpose_layout : bool, optional
        Se False (padrão), layout é N_slices linhas x 2 colunas (volumes).
        Se True, layout é 2 linhas (volumes) x N_slices colunas (slices).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Objeto da figura criada.
    """
    if volume1.shape != volume2.shape:
        raise ValueError(
            f"Volumes devem ter mesmo shape. "
            f"volume1: {volume1.shape}, volume2: {volume2.shape}"
        )

    # Normalizar ambos os volumes para [0, 1] usando normalização robusta
    # Ignora outliers (artefatos de metal, ruído) usando percentil 99.5
    vol1_norm = normalize_volume_robust(volume1, clamp_negatives=clamp_negatives)
    vol2_norm = normalize_volume_robust(volume2, clamp_negatives=clamp_negatives)

    # Selecionar fatias automaticamente se não fornecidas
    if slices is None:
        n_slices = volume1.shape[axis]
        # Pega 3 fatias: início (20%), meio (50%), fim (80%)
        slices = [
            int(n_slices * 0.20),
            int(n_slices * 0.50),
            int(n_slices * 0.80)
        ]

    # Garantir que slices é uma lista de inteiros
    slices = [int(s) for s in slices]
    n_slices_plot = len(slices)

    # Dimensionamento: 5 polegadas por painel
    panel_size = 5.0

    # Definir layout baseado em transpose_layout
    if transpose_layout:
        # Layout transposto: 2 linhas (volumes) x N colunas (slices)
        n_rows, n_cols = 2, n_slices_plot
        width = n_cols * panel_size
        height = n_rows * panel_size
    else:
        # Layout padrão: N linhas (slices) x 2 colunas (volumes)
        n_rows, n_cols = n_slices_plot, 2
        width = n_cols * panel_size
        height = n_rows * panel_size

    figsize = (width, height)

    # Criar figura
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
    )

    # Garantir que axes seja sempre 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plotar fatias
    axis_names = {0: "Axial", 1: "Coronal", 2: "Sagital"}
    axis_name = axis_names.get(axis, "Unknown")

    for i, slice_idx in enumerate(slices):
        # Extrair fatia baseado no eixo
        if axis == 0:  # Axial
            slice1 = vol1_norm[slice_idx, :, :]
            slice2 = vol2_norm[slice_idx, :, :]
        elif axis == 1:  # Coronal
            slice1 = vol1_norm[:, slice_idx, :]
            slice2 = vol2_norm[:, slice_idx, :]
        else:  # Sagital
            slice1 = vol1_norm[:, :, slice_idx]
            slice2 = vol2_norm[:, :, slice_idx]

        if transpose_layout:
            # Layout transposto: linha 0 = volume1, linha 1 = volume2
            # Volume 1 (normalizado [0, 1]) - primeira linha
            axes[0, i].imshow(slice1, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(
                f"{titles[0]} - {axis_name} (slice {slice_idx})",
                fontsize=13,
                fontweight='bold',
                pad=10
            )
            axes[0, i].axis("off")

            # Volume 2 (normalizado [0, 1]) - segunda linha
            axes[1, i].imshow(slice2, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(
                f"{titles[1]} - {axis_name} (slice {slice_idx})",
                fontsize=13,
                fontweight='bold',
                pad=10
            )
            axes[1, i].axis("off")
        else:
            # Layout padrão: coluna 0 = volume1, coluna 1 = volume2
            # Volume 1 (normalizado [0, 1]) - primeira coluna
            axes[i, 0].imshow(slice1, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(
                f"{titles[0]} - {axis_name} (slice {slice_idx})",
                fontsize=13,
                fontweight='bold',
                pad=10
            )
            axes[i, 0].axis("off")

            # Volume 2 (normalizado [0, 1]) - segunda coluna
            axes[i, 1].imshow(slice2, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(
                f"{titles[1]} - {axis_name} (slice {slice_idx})",
                fontsize=13,
                fontweight='bold',
                pad=10
            )
            axes[i, 1].axis("off")

    # Salvar figura se especificado
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Figura salva em: {save_path}")

    return fig


def compare_multiple_volumes(
    volumes,
    slices=None,
    axis=2,
    titles=None,
    save_path=None,
    clamp_negatives=True,
):
    """
    Compara fatias de múltiplos volumes lado a lado com normalização automática.

    Os volumes são automaticamente normalizados para [0, 1] permitindo
    comparação justa entre diferentes métodos e iterações.

    Parameters
    ----------
    volumes : list of numpy.ndarray
        Lista de volumes 3D [Z, Y, X] para comparar.
    slices : list of int, optional
        Índices das fatias a visualizar. Se None, seleciona 3 fatias
        distribuídas uniformemente (início, meio, fim).
    axis : int, optional
        Eixo ao longo do qual cortar (0=axial, 1=coronal, 2=sagital).
        Default: 2 (sagital).
    titles : list of str, optional
        Títulos para cada volume. Default: ["Volume 1", "Volume 2", ...].
    save_path : str, optional
        Caminho para salvar a figura (ex: 'comparison.png').
    clamp_negatives : bool, optional
        Se True (padrão), valores negativos são clampados para 0.
        Se False, faz shift adicionando o valor mínimo para tornar positivos.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Objeto da figura criada.
    """
    if not volumes or len(volumes) == 0:
        raise ValueError("A lista de volumes não pode estar vazia")

    # Verificar se todos os volumes têm o mesmo shape
    base_shape = volumes[0].shape
    for i, vol in enumerate(volumes[1:], 1):
        if vol.shape != base_shape:
            raise ValueError(
                f"Todos os volumes devem ter mesmo shape. "
                f"volume[0]: {base_shape}, volume[{i}]: {vol.shape}"
            )

    n_volumes = len(volumes)

    # Títulos padrão se não fornecidos
    if titles is None:
        titles = [f"Volume {i+1}" for i in range(n_volumes)]
    elif len(titles) != n_volumes:
        raise ValueError(
            f"Número de títulos ({len(titles)}) deve corresponder "
            f"ao número de volumes ({n_volumes})"
        )

    # Normalizar todos os volumes para [0, 1] usando normalização robusta
    # Ignora outliers (artefatos de metal, ruído) usando percentil 99.5
    volumes_norm = [normalize_volume_robust(vol, clamp_negatives=clamp_negatives) for vol in volumes]

    # Selecionar fatias automaticamente se não fornecidas
    if slices is None:
        n_slices = volumes[0].shape[axis]
        # Pega 3 fatias: início (20%), meio (50%), fim (80%)
        slices = [
            int(n_slices * 0.20),
            int(n_slices * 0.50),
            int(n_slices * 0.80)
        ]

    # Garantir que slices é uma lista de inteiros
    slices = [int(s) for s in slices]
    n_slices_plot = len(slices)

    # Dimensionamento: 5 polegadas por painel
    panel_size = 5.0
    width = n_volumes * panel_size
    height = n_slices_plot * panel_size
    figsize = (width, height)

    # Criar figura
    fig, axes = plt.subplots(
        n_slices_plot,
        n_volumes,
        figsize=figsize,
        constrained_layout=True,
    )

    # Garantir que axes seja sempre 2D
    if n_slices_plot == 1 and n_volumes == 1:
        axes = np.array([[axes]])
    elif n_slices_plot == 1:
        axes = axes.reshape(1, -1)
    elif n_volumes == 1:
        axes = axes.reshape(-1, 1)

    # Plotar fatias
    axis_names = {0: "Axial", 1: "Coronal", 2: "Sagital"}
    axis_name = axis_names.get(axis, "Unknown")

    for i, slice_idx in enumerate(slices):
        for j, vol_norm in enumerate(volumes_norm):
            # Extrair fatia baseado no eixo
            if axis == 0:  # Axial
                slice_data = vol_norm[slice_idx, :, :]
            elif axis == 1:  # Coronal
                slice_data = vol_norm[:, slice_idx, :]
            else:  # Sagital
                slice_data = vol_norm[:, :, slice_idx]

            # Plotar fatia normalizada [0, 1]
            axes[i, j].imshow(slice_data, cmap='gray', vmin=0, vmax=1)
            axes[i, j].set_title(
                f"{titles[j]} - {axis_name} (slice {slice_idx})",
                fontsize=13,
                fontweight='bold',
                pad=10
            )
            axes[i, j].axis("off")

    # Salvar figura se especificado
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Figura salva em: {save_path}")

    return fig
