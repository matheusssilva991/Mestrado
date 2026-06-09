from __future__ import annotations

import numpy as np

from .conjuntos_fuzzy import ESPECIFICACOES_CONJUNTOS_FUZZY, pertinencia_no_valor
from .universos import UNIVERSOS_DISCURSO


ROTULOS_RISCO_FADIGA = {
    "muito_baixo": "Muito baixo",
    "baixo": "Baixo",
    "moderado": "Moderado",
    "alto": "Alto",
    "altissimo": "Altíssimo",
}

DECISOES_OPERACIONAIS = {
    "Muito baixo": "Manter turbinas",
    "Baixo": "Manter turbinas",
    "Moderado": "Manter turbinas",
    "Alto": "Parar turbinas",
    "Altíssimo": "Parar turbinas",
}


def categorizar_risco_fadiga(
    valor_defuzzificado: float,
    universo_risco_fadiga: np.ndarray | None = None,
) -> str:
    """Retorna o rótulo linguístico com maior pertinência para o risco defuzzificado."""
    universo = (
        UNIVERSOS_DISCURSO["risco_fadiga"]
        if universo_risco_fadiga is None
        else universo_risco_fadiga
    )
    especificacoes_risco = ESPECIFICACOES_CONJUNTOS_FUZZY["risco_fadiga"]

    pertinencias_rotulos = {
        nome_conjunto: pertinencia_no_valor(
            valor_defuzzificado,
            universo,
            func_pertinencia,
            *parametros,
        )
        for nome_conjunto, (func_pertinencia, parametros) in especificacoes_risco.items()
    }
    nome_conjunto_maior_pertinencia = max(
        pertinencias_rotulos,
        key=pertinencias_rotulos.get,
    )
    return ROTULOS_RISCO_FADIGA[nome_conjunto_maior_pertinencia]


def decidir_operacao_turbinas(categoria_risco_fadiga: str) -> str:
    """Retorna a decisão operacional associada à categoria de risco."""
    return DECISOES_OPERACIONAIS[categoria_risco_fadiga]
