from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def carregar_regras(caminho_regras: str | Path) -> list[tuple[str, str, str, str]]:
    caminho_regras = Path(caminho_regras)
    with caminho_regras.open(encoding="utf-8") as arquivo_regras:
        base_regras = json.load(arquivo_regras)

    return [
        (
            regra["velocidade_vento"],
            regra["umidade_relativa"],
            regra["vibracao_torre"],
            regra["risco_fadiga"],
        )
        for regra in base_regras
    ]


def carregar_coletas(caminho_coletas: str | Path) -> list[list[float]]:
    caminho_coletas = Path(caminho_coletas)
    coletas_df = pd.read_csv(caminho_coletas)
    return coletas_df[
        ["velocidade_vento", "umidade_relativa", "vibracao_torre"]
    ].values.tolist()
