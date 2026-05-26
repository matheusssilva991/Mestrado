import matplotlib.pyplot as plt
from numpy.typing import NDArray
from pathlib import Path
from typing import Any


def plot_conjuntos_fuzzy(
    universo: NDArray,
    nome_universo: str,
    conjuntos: dict[str, NDArray],
    **kwargs: Any,
) -> None:
    """Plota os conjuntos fuzzy para um dado universo de discurso."""

    figsize = kwargs.get("figsize", (10, 5))
    title = kwargs.get("title", f"Função de Pertinência para {nome_universo}")
    xlabel = kwargs.get("xlabel", "X")
    ylabel = kwargs.get("ylabel", "Grau de Pertinência")

    fig, ax = plt.subplots(figsize=figsize)
    for nome_conjunto, conjunto in conjuntos.items():
        ax.plot(universo, conjunto, label=nome_conjunto)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()
    plt.show()


def plot_conjunto_saida_fuzzy(
    valores_risco: NDArray,
    pertinencias: NDArray,
    amostra_idx: int,
    caminho_saida: Path | None = None,
    exibir: bool = False,
    **kwargs: Any,
) -> Path | None:
    """Plota o conjunto fuzzy de saída agregado para uma amostra."""

    figsize = kwargs.get("figsize", (10, 5))
    title = kwargs.get("title", f"Conjunto fuzzy de saída - amostra {amostra_idx}")
    xlabel = kwargs.get("xlabel", "Risco de Fadiga (%)")
    ylabel = kwargs.get("ylabel", "Grau de Pertinência")
    label = kwargs.get("label", "saída agregada")
    color = kwargs.get("color", "tab:blue")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        valores_risco,
        pertinencias,
        color=color,
        linewidth=2,
        label=label,
    )
    ax.fill_between(
        valores_risco,
        pertinencias,
        color=color,
        alpha=0.2,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    ax.legend()

    if caminho_saida is not None:
        fig.savefig(caminho_saida, dpi=150, bbox_inches="tight")

    if exibir:
        plt.show()
    else:
        plt.close(fig)

    return caminho_saida


def salvar_figura_conjunto_saida(
    agregacao_df,
    figuras_saida_dir: Path,
    amostra_idx: int,
    exibir: bool = False,
) -> Path:
    """Salva (ou exibe) a figura do conjunto fuzzy de saída para uma amostra.

    Parâmetros:
    - agregacao_df: DataFrame com colunas ['amostra', 'valor_risco_fadiga', 'pertinencia_agregada']
    - figuras_saida_dir: diretório onde salvar as figuras (Path ou str)
    - amostra_idx: índice da amostra a plotar
    - exibir: se True, exibe a figura após salvar

    Retorna o caminho da figura gerada (Path).
    """
    saida_amostra = agregacao_df[agregacao_df["amostra"] == amostra_idx].sort_values(
        "valor_risco_fadiga"
    )

    if saida_amostra.empty:
        raise ValueError(f"Amostra {amostra_idx} não encontrada em agregacao_df")

    figuras_saida_dir = Path(figuras_saida_dir)
    figuras_saida_dir.mkdir(parents=True, exist_ok=True)
    caminho_figura = figuras_saida_dir / f"conjunto_saida_amostra_{amostra_idx}.png"

    return plot_conjunto_saida_fuzzy(
        saida_amostra["valor_risco_fadiga"].to_numpy(),
        saida_amostra["pertinencia_agregada"].to_numpy(),
        amostra_idx=amostra_idx,
        caminho_saida=caminho_figura,
        exibir=exibir,
    )
