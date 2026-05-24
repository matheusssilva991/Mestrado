from .pertinencias import (
    pertinencia_gaussiana,
    pertinencia_trapezoidal,
    pertinencia_triangular,
)
from .conjuntos_fuzzy import (
    conjuntos_fuzzy_por_variavel,
    gerar_conjuntos_fuzzy_por_amostra,
    gerar_conjuntos_fuzzy_velocidade_vento,
    gerar_conjuntos_fuzzy_umidade_relativa,
    gerar_conjuntos_fuzzy_vibracao_torre,
    gerar_conjuntos_fuzzy_risco_fadiga,
    pertinencia_no_valor,
)
from .defuzzificadores import (
    centro_gravidade,
    centro_maximos,
    media_maximos,
    DEFUZZICADORES_DISPONIVEIS,
)

from .plot import (
    plot_conjuntos_fuzzy,
    plot_conjunto_saida_fuzzy,
    salvar_figura_conjunto_saida,
)
from .operadores import (
    TNORMAS_DISPONIVEIS,
    TCONORMAS_DISPONIVEIS,
)
from .relacoes import (
    produto_carteziano,
    produto_carteziano_scalar,
)

from .tnormas import tnorma_min
from .tcornomas import tconorma_max

# Valores padrão exportados para uso global/compartilhado
DEFAULT_TNORMA = tnorma_min
DEFAULT_TCONORNA = tconorma_max
DEFAULT_DEFUZZIFICADOR = centro_gravidade

__all__ = [
    # pertinências
    "pertinencia_gaussiana",
    "pertinencia_trapezoidal",
    "pertinencia_triangular",
    # conjuntos fuzzy
    "conjuntos_fuzzy_por_variavel",
    "gerar_conjuntos_fuzzy_por_amostra",
    "gerar_conjuntos_fuzzy_velocidade_vento",
    "gerar_conjuntos_fuzzy_umidade_relativa",
    "gerar_conjuntos_fuzzy_vibracao_torre",
    "gerar_conjuntos_fuzzy_risco_fadiga",
    "pertinencia_no_valor",
    # plot
    "plot_conjuntos_fuzzy",
    "plot_conjunto_saida_fuzzy",
    "salvar_figura_conjunto_saida",
    # operadores
    "TNORMAS_DISPONIVEIS",
    "TCONORMAS_DISPONIVEIS",
    # padrões selecionáveis
    "DEFAULT_TNORMA",
    "DEFAULT_TCONORNA",
    # defuzzyficadores
    "centro_gravidade",
    "centro_maximos",
    "media_maximos",
    "DEFUZZICADORES_DISPONIVEIS",
    # relações
    "produto_carteziano",
    "produto_carteziano_scalar",
]
