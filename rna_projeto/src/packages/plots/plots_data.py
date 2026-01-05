import matplotlib.pyplot as plt


def plot_class_distribution(df, label_col='label', subset_col='subset',
                            figsize=(14, 5), save_path=None):
    """
    Plota a distribuição de classes do dataset.

    Args:
        df: DataFrame com os dados
        label_col: nome da coluna com as labels
        subset_col: nome da coluna com os subsets (train/val/test)
        figsize: tamanho da figura
        save_path: caminho para salvar a figura (opcional)

    Returns:
        fig, axes: objetos matplotlib
    """
    # Contagem por classe
    class_counts = df[label_col].value_counts().sort_index()

    # Criar figura
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Gráfico 1: Valores absolutos
    class_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Distribuição de Classes - Valores Absolutos',
                      fontsize=12, weight='bold')
    axes[0].set_xlabel('Classe')
    axes[0].set_ylabel('Número de Amostras')
    axes[0].grid(axis='y', alpha=0.3)

    # Adicionar valores nas barras
    for i, v in enumerate(class_counts):
        axes[0].text(i, v + 5, str(v), ha='center', va='bottom',
                     fontsize=11, weight='bold')

    # Gráfico 2: Percentual
    class_percentages = (class_counts / len(df) * 100).round(2)
    class_percentages.plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_title('Distribuição de Classes - Percentual',
                      fontsize=12, weight='bold')
    axes[1].set_xlabel('Classe')
    axes[1].set_ylabel('Percentual (%)')
    axes[1].grid(axis='y', alpha=0.3)

    # Adicionar percentuais nas barras
    for i, v in enumerate(class_percentages):
        axes[1].text(i, v + 0.5, f'{v}%', ha='center', va='bottom',
                     fontsize=11, weight='bold')

    plt.tight_layout()

    # Salvar se necessário
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Figura salva em: {save_path}")

    plt.show()
    return fig, axes


def plot_subset_distribution(df, label_col='label', subset_col='subset',
                             figsize=(16, 5), save_path=None):
    """
    Plota a distribuição de classes por subset (train/val/test).

    Args:
        df: DataFrame com os dados
        label_col: nome da coluna com as labels
        subset_col: nome da coluna com os subsets
        figsize: tamanho da figura
        save_path: caminho para salvar a figura (opcional)

    Returns:
        fig, axes: objetos matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    subsets = df[subset_col].unique()
    colors = ['steelblue', 'coral', 'lightgreen']

    for idx, (subset, color) in enumerate(zip(subsets, colors)):
        df_subset = df[df[subset_col] == subset]
        counts = df_subset[label_col].value_counts().sort_index()

        counts.plot(kind='bar', ax=axes[idx], color=color, edgecolor='black')
        axes[idx].set_title(f'Subset: {subset.upper()}',
                           fontsize=12, weight='bold')
        axes[idx].set_xlabel('Classe')
        axes[idx].set_ylabel('Número de Amostras')
        axes[idx].grid(axis='y', alpha=0.3)

        # Adicionar valores nas barras
        for i, v in enumerate(counts):
            axes[idx].text(i, v + 2, str(v), ha='center', va='bottom',
                          fontsize=10, weight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figura salva em: {save_path}")

    plt.show()
    return fig, axes
