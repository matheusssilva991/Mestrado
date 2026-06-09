# Trabalho 1 - Sistema fuzzy para risco de fadiga

Este projeto implementa um sistema de inferencia fuzzy para estimar o risco de
fadiga em turbinas a partir de tres variaveis de entrada:

- velocidade do vento;
- umidade relativa;
- vibracao da torre.

A saida do modelo e o risco de fadiga em uma escala de 0 a 100, acompanhado de
uma decisao operacional para manter a turbina em funcionamento ou parar as pas.
O fluxo principal esta no notebook `src/main.ipynb`, e uma versao automatizada
para gerar CSVs tambem esta disponivel em `src/main.py`.

## Ferramentas

- Python 3.13 ou superior.
- Jupyter Notebook, via `ipykernel`.
- NumPy, para operacoes numericas e vetorizacao.
- Pandas, para montar tabelas e exportar resultados em CSV.
- Matplotlib, para gerar e salvar os graficos.
- uv, usado para gerenciar o ambiente e as dependencias do projeto.

## Estrutura

```text
trabalho_1/
├── data/
│   ├── coletas.csv          # amostras de entrada do sistema fuzzy
│   ├── metadata.json        # metadados auxiliares das variaveis
│   └── regras.json          # base de regras fuzzy
├── doc/
│   └── Projeto.pdf          # especificacao do trabalho
├── output/
│   └── <operador>/          # CSVs e figuras geradas por operador fuzzy
├── src/
│   ├── main.ipynb           # notebook principal do experimento
│   ├── main.py              # execucao automatizada por linha de comando
│   └── utils/
│       ├── carregamento.py       # leitura das coletas e regras
│       ├── conjuntos_fuzzy.py    # montagem dos conjuntos fuzzy
│       ├── defuzzificadores.py   # metodos de defuzzificacao
│       ├── operadores.py         # operadores disponiveis
│       ├── pertinencias.py       # funcoes de pertinencia
│       ├── plot.py               # funcoes de visualizacao
│       ├── relacoes.py           # funcoes para relacoes fuzzy
│       ├── tcornomas.py          # t-conormas duais
│       ├── tnormas.py            # t-normas
│       └── universos.py          # universos de discurso
├── pyproject.toml
└── uv.lock
```

## Fluxo do modelo

1. Carrega as coletas em `data/coletas.csv`.
2. Carrega a base de regras em `data/regras.json`.
3. Define os universos de discurso e os conjuntos fuzzy das entradas e da saida.
4. Calcula os graus de pertinencia de cada amostra.
5. Identifica as regras ativadas para cada caso.
6. Aplica a t-norma selecionada para calcular a forca de ativacao das regras.
7. Aplica a implicacao fuzzy sobre os conjuntos de risco de fadiga.
8. Agrega as implicacoes com a t-conorma dual do operador escolhido.
9. Defuzzifica a saida agregada.
10. Classifica a saida em uma decisao operacional.

## Como rodar

Instale ou sincronize o ambiente:

```bash
uv sync
```

### Notebook

Abra `src/main.ipynb` no Jupyter ou no VS Code e execute as celulas em ordem.
O notebook gera tabelas intermediarias, graficos dos conjuntos fuzzy, graficos
de saida por amostra e a tabela final de defuzzificacao.

### Script

Para executar o fluxo automatizado e gerar os CSVs:

```bash
uv run python src/main.py --operador hamacher --defuzzificacao centro_gravidade
```

Tambem e possivel usar o interpretador do ambiente local:

```bash
.venv/bin/python src/main.py --operador hamacher --defuzzificacao centro_gravidade
```

## Parametros

Operadores fuzzy disponiveis:

- `min`
- `prod`
- `lukasiewicz`
- `hamacher`
- `sugeno-weber`
- `dombi`

Metodos de defuzzificacao disponiveis:

- `centro_gravidade`
- `centro_maximos`
- `media_maximos`

## Saidas

As saidas do notebook sao organizadas por operador em `output/<operador>/`.
Para o operador `hamacher`, por exemplo:

```text
output/hamacher/
├── agregacao.csv
├── fuzzyficacao_ativa.csv
├── fuzzyficacao_completa.csv
├── inferencia_implicacoes.csv
├── inferencia_regras.csv
├── regras.csv
├── saida_fuzzificacao.csv
└── figuras/
    ├── conjuntos/
    │   ├── conjuntos_fuzzy_risco_fadiga.png
    │   ├── conjuntos_fuzzy_umidade_relativa.png
    │   ├── conjuntos_fuzzy_velocidade_vento.png
    │   └── conjuntos_fuzzy_vibracao_torre.png
    └── saida/
        └── conjunto_saida_amostra_<n>.png
```

Quando o fluxo e executado pelo script, os arquivos ficam em
`output/<operador>/script/`.

## Arquivos principais

- `src/main.ipynb`: caminho recomendado para analisar o modelo passo a passo.
- `src/main.py`: execucao reproduzivel para gerar os CSVs principais.
- `data/regras.json`: base de regras usada na inferencia.
- `src/utils/operadores.py`: ponto central para consultar t-normas e
  t-conormas disponiveis.
- `src/utils/plot.py`: funcoes usadas para salvar os graficos do notebook.
