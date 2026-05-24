from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz

from utils import (
    DEFAULT_TCONORNA,
    DEFAULT_TNORMA,
    DEFAULT_DEFUZZIFICADOR,
    DEFUZZICADORES_DISPONIVEIS,
    TNORMAS_DISPONIVEIS,
    TCONORMAS_DISPONIVEIS,
    UNIVERSOS_DISCURSO,
    carregar_coletas,
    carregar_regras,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def criar_diretorios_saida(operador_fuzzy: str) -> Path:
    output_dir = PROJECT_DIR / "output" / operador_fuzzy / "script"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def calcular_pertinencia(
    valor: float, universo: np.ndarray, conjunto: np.ndarray
) -> float:
    return float(fuzz.interp_membership(universo, conjunto, valor))


def selecionar_operadores(nome_operador: str):
    tnorma = TNORMAS_DISPONIVEIS.get(nome_operador, DEFAULT_TNORMA)
    tconorma = TCONORMAS_DISPONIVEIS.get(nome_operador, DEFAULT_TCONORNA)
    return np.vectorize(tnorma, otypes=[float]), np.vectorize(tconorma, otypes=[float])


def criar_conjuntos_fuzzy(universos_discurso: dict[str, np.ndarray]):
    conjuntos_vento = {
        "baixa": fuzz.trapmf(universos_discurso["velocidade_vento"], [0, 0, 5, 8]),
        "media": fuzz.trimf(universos_discurso["velocidade_vento"], [6, 10, 14]),
        "media_alta": fuzz.trimf(universos_discurso["velocidade_vento"], [12, 16, 20]),
        "alta": fuzz.trimf(universos_discurso["velocidade_vento"], [18, 21.5, 25]),
        "altissimo": fuzz.trapmf(
            universos_discurso["velocidade_vento"], [22, 30, 40, 40]
        ),
    }

    conjuntos_umidade = {
        "muito_baixa": fuzz.trapmf(
            universos_discurso["umidade_relativa"], [0, 0, 10, 30]
        ),
        "baixa": fuzz.trimf(universos_discurso["umidade_relativa"], [20, 35, 50]),
        "moderada": fuzz.trimf(universos_discurso["umidade_relativa"], [40, 55, 70]),
        "alta": fuzz.trimf(universos_discurso["umidade_relativa"], [60, 75, 90]),
        "altissima": fuzz.trapmf(
            universos_discurso["umidade_relativa"], [80, 90, 100, 100]
        ),
    }

    conjuntos_vibracao = {
        "baixa": fuzz.trapmf(universos_discurso["vibracao_torre"], [0, 0, 1, 2]),
        "media": fuzz.trimf(universos_discurso["vibracao_torre"], [1.5, 2.75, 4]),
        "media_alta": fuzz.trimf(universos_discurso["vibracao_torre"], [3.5, 4.75, 6]),
        "alta": fuzz.trimf(universos_discurso["vibracao_torre"], [5.5, 6.75, 8]),
        "altissimo": fuzz.trapmf(
            universos_discurso["vibracao_torre"], [7.5, 9.5, 15, 15]
        ),
    }

    conjuntos_risco = {
        "muito_baixo": fuzz.trapmf(universos_discurso["risco_fadiga"], [0, 0, 10, 20]),
        "baixo": fuzz.trimf(universos_discurso["risco_fadiga"], [15, 30, 45]),
        "moderado": fuzz.trimf(universos_discurso["risco_fadiga"], [40, 52.5, 65]),
        "alto": fuzz.trimf(universos_discurso["risco_fadiga"], [60, 72.5, 85]),
        "altissimo": fuzz.trapmf(
            universos_discurso["risco_fadiga"], [80, 90, 100, 100]
        ),
    }

    return conjuntos_vento, conjuntos_umidade, conjuntos_vibracao, conjuntos_risco


def executar(
    operador_fuzzy: str = "hamacher",
    defuzzificacao: str = "centro_gravidade",
) -> None:
    universos_discurso = UNIVERSOS_DISCURSO
    output_dir = criar_diretorios_saida(operador_fuzzy)
    regras = carregar_regras(DATA_DIR / "regras.json")
    coletas = carregar_coletas(DATA_DIR / "coletas.csv")
    (
        conjuntos_vento,
        conjuntos_umidade,
        conjuntos_vibracao,
        conjuntos_risco,
    ) = criar_conjuntos_fuzzy(universos_discurso)
    aplicar_tnorma, aplicar_tconorma = selecionar_operadores(operador_fuzzy)
    defuzzificador = DEFUZZICADORES_DISPONIVEIS.get(
        defuzzificacao,
        DEFAULT_DEFUZZIFICADOR,
    )

    print(
        f"--- RESULTADOS DA DEFUZZIFICAÇÃO ({operador_fuzzy.upper()} | {defuzzificacao}) ---"
    )
    inferencia_regras_registros = []
    inferencia_implicacoes_registros = []
    agregacao_registros = []
    saida_fuzzificacao_registros = []

    for indice, (v_vento, v_umidade, v_vibracao) in enumerate(coletas, start=1):
        agregacao_final = np.zeros_like(universos_discurso["risco_fadiga"])
        registros_regras_amostra = []

        for indice_regra, (r_vento, r_umidade, r_vibracao, r_risco) in enumerate(
            regras,
            start=1,
        ):
            mu_ven = calcular_pertinencia(
                v_vento,
                universos_discurso["velocidade_vento"],
                conjuntos_vento[r_vento],
            )
            mu_umi = calcular_pertinencia(
                v_umidade,
                universos_discurso["umidade_relativa"],
                conjuntos_umidade[r_umidade],
            )
            mu_vib = calcular_pertinencia(
                v_vibracao,
                universos_discurso["vibracao_torre"],
                conjuntos_vibracao[r_vibracao],
            )

            w_temp = aplicar_tnorma(mu_ven, mu_umi)
            w_final = aplicar_tnorma(w_temp, mu_vib)

            if w_final > 0:
                vetor_implicado = aplicar_tnorma(w_final, conjuntos_risco[r_risco])
                agregacao_final = aplicar_tconorma(agregacao_final, vetor_implicado)

                for valor_risco, pertinencia_implicacao in zip(
                    universos_discurso["risco_fadiga"],
                    vetor_implicado,
                ):
                    inferencia_implicacoes_registros.append(
                        {
                            "amostra": indice,
                            "regra": indice_regra,
                            "risco_fadiga": r_risco,
                            "valor_risco_fadiga": valor_risco,
                            "pertinencia_risco_fadiga": pertinencia_implicacao,
                            "forca_ativacao": w_final,
                            "pertinencia_implicacao": pertinencia_implicacao,
                        }
                    )

            registros_regras_amostra.append(
                {
                    "amostra": indice,
                    "regra": indice_regra,
                    "proposicao": (
                        f"SE velocidade_vento É {r_vento} E umidade_relativa É {r_umidade} "
                        f"E vibracao_torre É {r_vibracao} ENTÃO risco_fadiga É {r_risco}"
                    ),
                    "valor_velocidade_vento": v_vento,
                    "conjunto_velocidade_vento": r_vento,
                    "pertinencia_velocidade_vento": mu_ven,
                    "valor_umidade_relativa": v_umidade,
                    "conjunto_umidade_relativa": r_umidade,
                    "pertinencia_umidade_relativa": mu_umi,
                    "valor_vibracao_torre": v_vibracao,
                    "conjunto_vibracao_torre": r_vibracao,
                    "pertinencia_vibracao_torre": mu_vib,
                    "forca_ativacao": w_final,
                    "risco_fadiga": r_risco,
                }
            )

        inferencia_regras_registros.extend(registros_regras_amostra)

        for valor_risco, pertinencia_agregada in zip(
            universos_discurso["risco_fadiga"],
            agregacao_final,
        ):
            agregacao_registros.append(
                {
                    "amostra": indice,
                    "valor_risco_fadiga": valor_risco,
                    "pertinencia_agregada": pertinencia_agregada,
                }
            )

        soma_pertinencias = np.sum(agregacao_final)
        if soma_pertinencias == 0:
            print(
                f"Coleta {indice} -> Erro na defuzzificação (Nenhuma regra ativada fortemente)"
            )
            continue

        resultado_risco = defuzzificador(
            universos_discurso["risco_fadiga"],
            agregacao_final,
        )
        saida_fuzzificacao_registros.append(
            {
                "amostra": indice,
                "velocidade_vento": v_vento,
                "umidade_relativa": v_umidade,
                "vibracao_torre": v_vibracao,
                "valor_defuzzificado": resultado_risco,
            }
        )
        print(
            f"Coleta {indice} (Ven:{v_vento}, Umi:{v_umidade}, Vib:{v_vibracao}) -> Risco de Fadiga: {resultado_risco:.2f}%"
        )

    pd.DataFrame(inferencia_regras_registros).to_csv(
        output_dir / "inferencia_regras.csv",
        index=False,
    )
    pd.DataFrame(inferencia_implicacoes_registros).to_csv(
        output_dir / "inferencia_implicacoes.csv",
        index=False,
    )
    pd.DataFrame(agregacao_registros).to_csv(
        output_dir / "agregacao.csv",
        index=False,
    )
    pd.DataFrame(saida_fuzzificacao_registros).to_csv(
        output_dir / "saida_fuzzificacao.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o sistema fuzzy do trabalho 1."
    )
    parser.add_argument(
        "--operador",
        choices=sorted(TNORMAS_DISPONIVEIS.keys()),
        default="hamacher",
        help="Operador fuzzy para t-norma e t-conorma.",
    )
    parser.add_argument(
        "--defuzzificacao",
        choices=sorted(DEFUZZICADORES_DISPONIVEIS.keys()),
        default="centro_gravidade",
        help="Método de defuzzificação para a saída agregada.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    argumentos = parse_args()
    executar(
        operador_fuzzy=argumentos.operador,
        defuzzificacao=argumentos.defuzzificacao,
    )
