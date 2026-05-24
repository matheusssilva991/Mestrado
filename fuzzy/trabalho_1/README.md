# Trabalho fuzzy

Implementação do trabalho 1 de lógica fuzzy, com notebook e script Python para
inferência e defuzzificação do risco de fadiga.

## Visão geral

- Leitura das regras em `data/regras.json`.
- Leitura das coletas em `data/coletas.csv`.
- Definição centralizada dos universos de discurso em `src/utils/universos.py`.
- Seleção de t-norma, t-conorma e defuzzificador via código ou linha de comando.

## Estrutura

```text
data/
output/
src/
 main.ipynb
 main.py
 utils/
```

## Como executar

### Notebook

Abra e execute o notebook em `src/main.ipynb`.

### Script

Use o script para gerar os CSVs da execução:

```bash
.venv/bin/python src/main.py --operador hamacher --defuzzificacao centro_gravidade
```

## Parâmetros

### Operadores fuzzy

- `min`
- `prod`
- `lukasiewicz`
- `hamacher`
- `sugeno-weber`
- `dombi`

### Defuzzificação

- `centro_gravidade`
- `centro_maximos`
- `media_maximos`

## Saídas

- Notebook: `output/<operador>/`
- Script: `output/<operador>/script/`

Arquivos gerados pelo script:

- `inferencia_regras.csv`
- `inferencia_implicacoes.csv`
- `agregacao.csv`
- `saida_fuzzificacao.csv`

## Módulos úteis

- `src/utils/plot.py`: funções de visualização e salvamento de figuras.
- `src/utils/carregamento.py`: carregamento de regras e coletas.
- `src/utils/operadores.py`: mapa dos operadores fuzzy disponíveis.
- `src/utils/defuzzificadores.py`: métodos de defuzzificação disponíveis.
