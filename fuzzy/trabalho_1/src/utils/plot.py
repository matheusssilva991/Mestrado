import matplotlib.pyplot as plt
import numpy as np
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

    figsize = kwargs.get("figsize", (12, 6))
    title = kwargs.get("title", f"Função de Pertinência para {nome_universo}")
    xlabel = kwargs.get("xlabel", "X")
    ylabel = kwargs.get("ylabel", "Grau de Pertinência")
    title_fontsize = kwargs.get("title_fontsize", 14)
    label_fontsize = kwargs.get("label_fontsize", 13)
    tick_labelsize = kwargs.get("tick_labelsize", 10)
    legend_fontsize = kwargs.get("legend_fontsize", 10)
    caminho_saida = kwargs.get("caminho_saida")
    dpi = kwargs.get("dpi", 600)
    xticks = kwargs.get("xticks")
    xtick_step = kwargs.get("xtick_step")
    num_xticks = kwargs.get("num_xticks")
    xtick_rotation = kwargs.get("xtick_rotation", 0)
    minor_grid = kwargs.get("minor_grid", True)

    fig, ax = plt.subplots(figsize=figsize)
    for nome_conjunto, conjunto in conjuntos.items():
        ax.plot(universo, conjunto, label=nome_conjunto)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if xticks is not None:
        ax.set_xticks(xticks)
    elif xtick_step is not None:
        inicio = np.floor(float(np.min(universo)) / xtick_step) * xtick_step
        fim = np.ceil(float(np.max(universo)) / xtick_step) * xtick_step
        ax.set_xticks(np.arange(inicio, fim + xtick_step, xtick_step))
    elif num_xticks is not None:
        ax.set_xticks(
            np.linspace(float(np.min(universo)), float(np.max(universo)), num_xticks)
        )

    ax.tick_params(axis="x", rotation=xtick_rotation, labelsize=tick_labelsize)
    ax.tick_params(axis="y", labelsize=tick_labelsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, which="major", alpha=0.45)
    if minor_grid:
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.15)

    if caminho_saida is not None:
        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")

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

    figsize = kwargs.get("figsize", (12, 6))
    title = kwargs.get("title", f"Conjunto fuzzy de saída - amostra {amostra_idx}")
    xlabel = kwargs.get("xlabel", "Risco de Fadiga (%)")
    ylabel = kwargs.get("ylabel", "Grau de Pertinência")
    title_fontsize = kwargs.get("title_fontsize", 15)
    label_fontsize = kwargs.get("label_fontsize", 14)
    tick_labelsize = kwargs.get("tick_labelsize", 11)
    legend_fontsize = kwargs.get("legend_fontsize", 11)
    label = kwargs.get("label", "saída agregada")
    color = kwargs.get("color", "tab:blue")
    dpi = kwargs.get("dpi", 600)
    xticks = kwargs.get("xticks")
    xtick_step = kwargs.get("xtick_step", 10)
    yticks = kwargs.get("yticks", np.arange(0, 1.01, 0.1))
    minor_grid = kwargs.get("minor_grid", True)
    destacar_maximo = kwargs.get("destacar_maximo", True)

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
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_ylim(0, 1.05)

    if xticks is not None:
        ax.set_xticks(xticks)
    elif xtick_step is not None:
        inicio = np.floor(float(np.min(valores_risco)) / xtick_step) * xtick_step
        fim = np.ceil(float(np.max(valores_risco)) / xtick_step) * xtick_step
        ax.set_xticks(np.arange(inicio, fim + xtick_step, xtick_step))

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(axis="x", labelsize=tick_labelsize)
    ax.tick_params(axis="y", labelsize=tick_labelsize)

    if destacar_maximo and len(pertinencias) > 0:
        indice_maximo = int(np.argmax(pertinencias))
        valor_maximo = valores_risco[indice_maximo]
        pertinencia_maxima = pertinencias[indice_maximo]
        ax.scatter(
            [valor_maximo],
            [pertinencia_maxima],
            color="tab:red",
            zorder=3,
            label=f"pico: {valor_maximo:.1f}%",
        )

    ax.grid(True, which="major", alpha=0.45)
    if minor_grid:
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.15)
    ax.legend(fontsize=legend_fontsize)

    if caminho_saida is not None:
        fig.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")

    if exibir:
        plt.show()
    else:
        plt.close(fig)

    return caminho_saida


def plot_implicacoes_fuzzy(
    implicacoes_df,
    amostra_idx: int,
    caminho_saida: Path | None = None,
    exibir: bool = False,
    **kwargs: Any,
) -> Path | None:
    """Plota as implicações fuzzy das regras ativadas para uma amostra."""

    implicacoes_amostra = implicacoes_df[
        implicacoes_df["amostra"] == amostra_idx
    ].sort_values(["regra", "valor_risco_fadiga"])

    if implicacoes_amostra.empty:
        raise ValueError(f"Amostra {amostra_idx} não encontrada em implicacoes_df")

    figsize = kwargs.get("figsize", (12, 6))
    title = kwargs.get("title", f"Implicações fuzzy - amostra {amostra_idx}")
    xlabel = kwargs.get("xlabel", "Risco de Fadiga (%)")
    ylabel = kwargs.get("ylabel", "Grau de Pertinência")
    title_fontsize = kwargs.get("title_fontsize", 15)
    label_fontsize = kwargs.get("label_fontsize", 14)
    tick_labelsize = kwargs.get("tick_labelsize", 11)
    legend_fontsize = kwargs.get("legend_fontsize", 9)
    dpi = kwargs.get("dpi", 600)
    xtick_step = kwargs.get("xtick_step", 10)
    yticks = kwargs.get("yticks", np.arange(0, 1.01, 0.1))
    minor_grid = kwargs.get("minor_grid", True)

    fig, ax = plt.subplots(figsize=figsize)

    for (regra, risco_fadiga), implicacao in implicacoes_amostra.groupby(
        ["regra", "risco_fadiga"],
        sort=True,
    ):
        forca_ativacao = float(implicacao["forca_ativacao"].iloc[0])
        ax.plot(
            implicacao["valor_risco_fadiga"],
            implicacao["pertinencia_implicacao"],
            linewidth=1.8,
            alpha=0.9,
            label=f"regra {regra} -> {risco_fadiga} (w={forca_ativacao:.3f})",
        )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_ylim(0, 1.05)

    if xtick_step is not None:
        valores_risco = implicacoes_amostra["valor_risco_fadiga"]
        inicio = np.floor(float(valores_risco.min()) / xtick_step) * xtick_step
        fim = np.ceil(float(valores_risco.max()) / xtick_step) * xtick_step
        ax.set_xticks(np.arange(inicio, fim + xtick_step, xtick_step))

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(axis="x", labelsize=tick_labelsize)
    ax.tick_params(axis="y", labelsize=tick_labelsize)
    ax.grid(True, which="major", alpha=0.45)
    if minor_grid:
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.15)
    ax.legend(fontsize=legend_fontsize, loc="upper right")

    if caminho_saida is not None:
        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")

    if exibir:
        plt.show()
    else:
        plt.close(fig)

    return caminho_saida


def salvar_figura_implicacoes(
    implicacoes_df,
    figuras_implicacoes_dir: Path,
    amostra_idx: int,
    exibir: bool = False,
    dpi: int = 600,
) -> Path:
    """Salva (ou exibe) as curvas de implicação fuzzy de uma amostra."""

    figuras_implicacoes_dir = Path(figuras_implicacoes_dir)
    figuras_implicacoes_dir.mkdir(parents=True, exist_ok=True)
    caminho_figura = figuras_implicacoes_dir / f"implicacoes_amostra_{amostra_idx}.png"

    return plot_implicacoes_fuzzy(
        implicacoes_df,
        amostra_idx=amostra_idx,
        caminho_saida=caminho_figura,
        exibir=exibir,
        dpi=dpi,
    )


def salvar_figura_conjunto_saida(
    agregacao_df,
    figuras_saida_dir: Path,
    amostra_idx: int,
    exibir: bool = False,
    dpi: int = 600,
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
        dpi=dpi,
    )
