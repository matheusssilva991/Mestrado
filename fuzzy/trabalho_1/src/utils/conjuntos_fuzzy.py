from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .pertinencias import (
    pertinencia_trapezoidal,
    pertinencia_trapezoidal_scalar,
    pertinencia_triangular,
    pertinencia_triangular_scalar,
)


FloatArray = NDArray[np.float64]

EspecificacaoConjunto = tuple[Callable[..., float | FloatArray], tuple[float, ...]]
EspecificacoesConjuntos = dict[str, EspecificacaoConjunto]


ESPECIFICACOES_CONJUNTOS_FUZZY: dict[str, EspecificacoesConjuntos] = {
    "velocidade_vento": {
        "baixa": (pertinencia_trapezoidal, (0, 0, 5, 8)),
        "media": (pertinencia_triangular, (6, 10, 14)),
        "media_alta": (pertinencia_triangular, (12, 16, 20)),
        "alta": (pertinencia_triangular, (18, 21.5, 25)),
        "altissimo": (pertinencia_trapezoidal, (22, 30, 40, 40)),
    },
    "umidade_relativa": {
        "muito_baixa": (pertinencia_trapezoidal, (0, 0, 10, 30)),
        "baixa": (pertinencia_triangular, (20, 35, 50)),
        "moderada": (pertinencia_triangular, (40, 55, 70)),
        "alta": (pertinencia_triangular, (60, 75, 90)),
        "altissima": (pertinencia_trapezoidal, (80, 90, 100, 100)),
    },
    "vibracao_torre": {
        "baixa": (pertinencia_trapezoidal, (0, 0, 1, 2)),
        "media": (pertinencia_triangular, (1.5, 2.75, 4)),
        "media_alta": (pertinencia_triangular, (3.5, 4.75, 6)),
        "alta": (pertinencia_triangular, (5.5, 6.75, 8)),
        "altissimo": (pertinencia_trapezoidal, (7.5, 9.5, 15, 15)),
    },
    "risco_fadiga": {
        "muito_baixo": (pertinencia_trapezoidal, (0.0, 0.0, 10.0, 20.0)),
        "baixo": (pertinencia_triangular, (15.0, 30.0, 45.0)),
        "moderado": (pertinencia_triangular, (40.0, 52.5, 65.0)),
        "alto": (pertinencia_triangular, (60.0, 72.5, 85.0)),
        "altissimo": (pertinencia_trapezoidal, (80.0, 90.0, 100.0, 100.0)),
    },
}


def _construir_conjunto_fuzzy(
    universo: FloatArray,
    especificacoes: EspecificacoesConjuntos,
) -> dict[str, FloatArray]:
    """Cria um dicionário de conjuntos fuzzy a partir das especificações dadas."""
    return {
        nome: np.asarray(func(universo, *parametros), dtype=float)
        for nome, (func, parametros) in especificacoes.items()
    }


def gerar_conjuntos_fuzzy_velocidade_vento(
    universo: FloatArray,
) -> dict[str, FloatArray]:
    """Retorna os conjuntos fuzzy da velocidade do vento."""
    return _construir_conjunto_fuzzy(
        universo, ESPECIFICACOES_CONJUNTOS_FUZZY["velocidade_vento"]
    )


def gerar_conjuntos_fuzzy_umidade_relativa(
    universo: FloatArray,
) -> dict[str, FloatArray]:
    """Retorna os conjuntos fuzzy da umidade relativa."""
    return _construir_conjunto_fuzzy(
        universo, ESPECIFICACOES_CONJUNTOS_FUZZY["umidade_relativa"]
    )


def gerar_conjuntos_fuzzy_vibracao_torre(universo: FloatArray) -> dict[str, FloatArray]:
    """Retorna os conjuntos fuzzy da vibração da torre."""
    return _construir_conjunto_fuzzy(
        universo, ESPECIFICACOES_CONJUNTOS_FUZZY["vibracao_torre"]
    )


def gerar_conjuntos_fuzzy_risco_fadiga(universo: FloatArray) -> dict[str, FloatArray]:
    """Retorna os conjuntos fuzzy para risco de fadiga.

    Observação: as especificações usam a escala [0, 1]; escolha um universo compatível.
    """
    return _construir_conjunto_fuzzy(
        universo, ESPECIFICACOES_CONJUNTOS_FUZZY["risco_fadiga"]
    )


def conjuntos_fuzzy_por_variavel(
    nome_variavel: str,
    universo: FloatArray,
) -> dict[str, FloatArray]:
    """Seleciona o conjunto fuzzy correto para uma variável conhecida."""
    mapeamento: dict[str, Callable[[FloatArray], dict[str, FloatArray]]] = {
        "velocidade_vento": gerar_conjuntos_fuzzy_velocidade_vento,
        "umidade_relativa": gerar_conjuntos_fuzzy_umidade_relativa,
        "vibracao_torre": gerar_conjuntos_fuzzy_vibracao_torre,
        "risco_fadiga": gerar_conjuntos_fuzzy_risco_fadiga,
    }

    if nome_variavel not in mapeamento:
        raise ValueError(f"Variável desconhecida: {nome_variavel}")

    return mapeamento[nome_variavel](universo)


def pertinencia_no_valor(
    x: float,
    universo: FloatArray,
    func_pertinencia: Callable[..., float | FloatArray],
    *parametros: float,
) -> float:
    """Calcula a pertinência escalar de um valor x no conjunto definido pela função."""
    if func_pertinencia is pertinencia_triangular:
        return float(pertinencia_triangular_scalar(float(x), *parametros))
    if func_pertinencia is pertinencia_trapezoidal:
        return float(pertinencia_trapezoidal_scalar(float(x), *parametros))

    valores = np.asarray(func_pertinencia(universo, *parametros), dtype=float)
    return float(np.interp(float(x), universo, valores))


def gerar_conjuntos_fuzzy_por_amostra(
    amostra: dict[str, float] | Any,
) -> dict[str, dict[str, float]]:
    """Gera os graus de pertinência de uma amostra para todas as variáveis conhecidas."""
    resultado: dict[str, dict[str, float]] = {}
    for nome_variavel, especificacoes in ESPECIFICACOES_CONJUNTOS_FUZZY.items():
        # se a amostra não tiver a variável (ex.: 'risco_fadiga'), pule
        if nome_variavel not in amostra:
            continue

        try:
            valor = float(amostra[nome_variavel])
        except Exception:
            # não é possível extrair valor numérico da amostra; pule a variável
            continue

        graus: dict[str, float] = {}
        for nome_conjunto, (func_pertinencia, parametros) in especificacoes.items():
            # as funções em ESPECIFICACOES_CONJUNTOS_FUZZY (wrappers) aceitam float
            try:
                mu = float(func_pertinencia(valor, *parametros))
            except Exception:
                # fallback: calcular via interp no universo definido pelos parâmetros
                mu = float(
                    pertinencia_no_valor(
                        valor, np.array([], dtype=float), func_pertinencia, *parametros
                    )
                )
            graus[nome_conjunto] = max(0.0, min(1.0, mu))

        resultado[nome_variavel] = graus

    return resultado
