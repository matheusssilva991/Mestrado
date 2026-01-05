import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_random_samples(df, n_samples=3, figsize=(12, 4),
                       path_col='crop_path', label_col='label',
                       subset_col='subset', seed=None, save_path=None):
    """
    Plota amostras aleatórias do dataset.

    Args:
        df: DataFrame com os dados
        n_samples: número de amostras a visualizar
        figsize: tamanho da figura
        path_col: nome da coluna com o caminho da imagem
        label_col: nome da coluna com as labels
        subset_col: nome da coluna com os subsets (opcional)
        seed: seed para reprodutibilidade
        save_path: caminho para salvar a figura (opcional)

    Returns:
        fig, axes: objetos matplotlib
    """
    if seed is not None:
        np.random.seed(seed)

    # Selecionar amostras aleatórias
    sample_indices = np.random.choice(len(df), size=n_samples, replace=False)

    fig, axes = plt.subplots(1, n_samples, figsize=figsize)

    # Garantir que axes seja sempre uma lista
    if n_samples == 1:
        axes = [axes]

    for idx, ax in zip(sample_indices, axes):
        crop_path = df.iloc[idx][path_col]
        label = df.iloc[idx][label_col]

        # Verificar se existe coluna subset
        if subset_col and subset_col in df.columns:
            subset = df.iloc[idx][subset_col]
            title = f'Label: {label}\nSubset: {subset}\nIdx: {idx}'
        else:
            title = f'Label: {label}\nIdx: {idx}'

        # Carregar imagem original (sem processamento)
        img = Image.open(crop_path)

        # Visualizar
        ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figura salva em: {save_path}")

    plt.show()

    return fig, axes


def plot_samples_by_class(df, path_col='crop_path', label_col='label',
                          samples_per_class=3, figsize=None, save_path=None):
    """
    Plota amostras de cada classe do dataset.

    Args:
        df: DataFrame com os dados
        path_col: nome da coluna com o caminho da imagem
        label_col: nome da coluna com as labels
        samples_per_class: número de amostras por classe
        figsize: tamanho da figura (auto se None)
        save_path: caminho para salvar a figura (opcional)

    Returns:
        fig, axes: objetos matplotlib
    """
    classes = sorted(df[label_col].unique())
    n_classes = len(classes)

    # Calcular tamanho da figura automaticamente
    if figsize is None:
        figsize = (samples_per_class * 4, n_classes * 3)

    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=figsize)

    # Garantir que axes seja sempre 2D
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    if samples_per_class == 1:
        axes = axes.reshape(-1, 1)

    for class_idx, class_label in enumerate(classes):
        # Filtrar amostras da classe
        df_class = df[df[label_col] == class_label]

        # Selecionar amostras aleatórias
        n_available = min(samples_per_class, len(df_class))
        sample_indices = np.random.choice(len(df_class), size=n_available, replace=False)

        for sample_idx in range(samples_per_class):
            ax = axes[class_idx, sample_idx]

            if sample_idx < n_available:
                idx = sample_indices[sample_idx]
                crop_path = df_class.iloc[idx][path_col]

                # Carregar imagem
                img = Image.open(crop_path)

                # Visualizar
                ax.imshow(img, cmap='gray' if img.mode == 'L' else None)

                # Título apenas na primeira coluna
                if sample_idx == 0:
                    ax.set_ylabel(f'Classe {class_label}', fontsize=12, weight='bold')

                ax.axis('off')
            else:
                # Se não houver amostras suficientes, deixar em branco
                ax.axis('off')

    plt.suptitle('Amostras por Classe', fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figura salva em: {save_path}")

    plt.show()

    return fig, axes
