# Tarefa 1: Funções de Pertinência de Idade e Operadores Fuzzy

## Visão Geral

Esta tarefa implementa funções de pertinência fuzzy para classificar idades em categorias linguísticas (jovem, adulto) e operadores fuzzy para combinar e modificar esses conjuntos fuzzy.

## Estrutura do Projeto

```text
tarefa_1/
├── main.ipynb                 # Notebook com análises e visualizações
├── README.md                  # Este arquivo
└── utils/
    ├── __init__.py
    ├── age_membership/        # Funções de pertinência por idade
    │   ├── __init__.py
    │   └── age_membership.py
    └── fuzzy_operators/       # Operadores fuzzy (T-normas, T-conormas, negações)
        ├── __init__.py
        └── fuzzy_operators.py
```

## Funções de Pertinência de Idade

Localizadas em `utils/age_membership/age_membership.py`

### young_membership(age, tail_end=30)

Define o grau de pertinência para a categoria **"jovem"**.

**Faixas:**

- **age ≤ 18**: retorna 1.0 (totalmente jovem)
- **18 < age ≤ 21**: desce linearmente de 1.0 para 0.7
  - Fórmula: `1.0 - 0.1 * (age - 18.0)`
- **21 < age ≤ 25**: desce linearmente de 0.7 para 0.3
  - Fórmula: `0.7 - 0.1 * (age - 21.0)`
- **25 < age < tail_end**: desce linearmente de 0.3 para 0.0
  - Fórmula: `0.3 * (tail_end - age) / (tail_end - 25.0)`
- **age ≥ tail_end**: retorna 0.0

**Parâmetro:**

- `tail_end` (padrão: 30): idade onde a queda após 25 anos atinge zero

### adult_membership(age, tail_end=70)

Define o grau de pertinência para a categoria **"adulto"**.

**Faixas:**

- **age ≤ 18**: retorna 0.0 (não é adulto)
- **18 < age ≤ 21**: sobe linearmente de 0.0 para 0.3
  - Fórmula: `0.1 * (age - 18.0)`
- **21 < age ≤ 25**: sobe linearmente de 0.3 para 0.7
  - Fórmula: `0.3 + 0.1 * (age - 21.0)`
- **25 < age ≤ 30**: sobe linearmente de 0.7 para 1.0
  - Fórmula: `0.7 + 0.06 * (age - 25.0)`
- **30 < age ≤ 55**: retorna 1.0 (plenamente adulto)
- **55 < age ≤ 60**: desce linearmente de 1.0 para 0.7
  - Fórmula: `1.0 - 0.06 * (age - 55.0)`
- **60 < age ≤ 65**: desce linearmente de 0.7 para 0.3
  - Fórmula: `0.7 - 0.08 * (age - 60.0)`
- **65 < age < tail_end**: desce linearmente de 0.3 para 0.0
  - Fórmula: `0.3 * (tail_end - age) / (tail_end - 65.0)`
- **age ≥ tail_end**: retorna 0.0

**Parâmetro:**

- `tail_end` (padrão: 70): idade onde a queda após 65 anos atinge zero

## Operadores Fuzzy

Localizados em `utils/fuzzy_operators/fuzzy_operators.py`

### T-normas (Interseção Fuzzy)

Representam a interseção de conjuntos fuzzy (operação E).

#### min_tnorma(a, b)

Retorna o mínimo entre dois graus de pertinência.

- `min(a, b)`
- Conservadora, adequada para situações rigorosas

#### prod_tnorma(a, b)

Retorna o produto entre dois graus de pertinência.

- `a * b`
- Menos rigorosa que min, penaliza valores baixos

### T-conormas (União Fuzzy)

Representam a união de conjuntos fuzzy (operação OU).

#### max_tconorma(a, b)

Retorna o máximo entre dois graus de pertinência.

- `max(a, b)`

#### lukasiewicz_tconorma(a, b)

T-conorma de Lukasiewicz.

- `min(1, a + b)`
- Menos pessimista que max

### Negações Fuzzy (Complemento)

Representam o complemento de um conjunto fuzzy (operação NÃO).

#### zadeh_negation(a)

Negação clássica (também chamada de negação de Zadeh).

- `1 - a`
- Simples e intuitiva

#### sugeno_negation(a, lamb=0)

Negação parametrizada de Sugeno.

- Fórmula: $(1 - a) / (1 + \lambda \cdot a)$
- Quando $\lambda = 0$, reduz à negação de Zadeh
- $\lambda$ controla a curvatura da negação

**Parâmetro:**

- `lamb` (padrão: 0): parâmetro que controla o comportamento da negação

## Como Usar

### Importação

```python
from utils import (
    young_membership,
    adult_membership,
    min_tnorma,
    prod_tnorma,
    max_tconorma,
    lukasiewicz_tconorma,
    zadeh_negation,
    sugeno_negation,
)
```

### Exemplo: Calcular Pertinência

```python
age = 28
grau_jovem = young_membership(age)      # Quanto é "jovem"
grau_adulto = adult_membership(age)     # Quanto é "adulto"

print(f"Idade {age}: {grau_jovem:.2f} jovem, {grau_adulto:.2f} adulto")
```

### Exemplo: Usar Operadores

```python
# Interseção (E): "Jovem E Adulto"
interseccao = min_tnorma(grau_jovem, grau_adulto)

# União (OU): "Jovem OU Adulto"
uniao = max_tconorma(grau_jovem, grau_adulto)

# Negação (NÃO): "NÃO Jovem"
nao_jovem = zadeh_negation(grau_jovem)
```

## Visualização

Abra `main.ipynb` no Jupyter para visualizar gráficos das funções de pertinência e operadores fuzzy aplicados.
