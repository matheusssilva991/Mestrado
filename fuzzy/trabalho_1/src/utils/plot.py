import matplotlib.pyplot as plt
from numpy.typing import NDArray
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
