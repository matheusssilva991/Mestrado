# Métodos Numéricos Final

## Descrição

Biblioteca Python para implementação e aplicação de métodos numéricos, desenvolvida como projeto final da disciplina de Métodos Numéricos. Esta biblioteca fornece implementações eficientes de diversos algoritmos para resolver problemas matemáticos comuns em engenharia e ciências.

## Funcionalidades

A biblioteca oferece implementações para:

- Interpolação
  - Interpolação de Lagrange
  - Expansão em Série de Taylor
  - Splines Cúbicos

- Busca de Raízes
  - Método da Bisseção
  - Método da Posição Falsa (Regula Falsi)
  - Método do Ponto Fixo
  - Método de Newton-Raphson
  - Método da Secante

- Regressão
  - Regressão Linear
  - Regressão Polinomial
  - Cálculo de coeficientes de correlação e determinação

## Instalação

```bash
# Clone o repositório
git clone https://github.com/matheusssilva991/metodos_numericos_final.git

# Entre no diretório
cd metodos_numericos_final

# Crie e ative o ambiente conda
conda env create -f environment.yml
conda activate metodos_numericos
```

## Como usar

Interpolação

```python
import numpy as np
from methods.interpolation import lagrange_interpolation, taylor_interpolation

# Interpolação de Lagrange
points = np.array([[0, 1, 2], [1, 2, 0]])  # Matriz de pontos (x, y)
xi = 1.5  # Ponto a interpolar
result = lagrange_interpolation(points, xi)
print(f"Valor interpolado em x = {xi}: {result}")

# Expansão em Série de Taylor
func = "sin(x)"
x0 = 0.0  # Ponto de expansão
xi = 0.5  # Ponto a avaliar
order = 3  # Ordem da expansão
taylor_result = taylor_interpolation(func, x0, xi, order)
print(f"Expansão de Taylor em x = {xi}: {taylor_result}")
```

Busca de Raízes


```python
from methods.root import bisection, newton_raphson

# Método da bisseção
func = "x^3 - 2*x - 5"
a, b = 2, 3  # Intervalo inicial
tol = 1e-6  # Tolerância
max_iter = 100  # Número máximo de iterações
root = bisection(func, a, b, tol, max_iter)
print(f"Raiz encontrada: {root}")

# Método de Newton-Raphson
func = "cos(x) - x"
x0 = 0.5  # Estimativa inicial
root = newton_raphson(func, x0, tol, max_iter)
print(f"Raiz encontrada: {root}")
```

Regressão

```python
import numpy as np
from methods.regression import linear_least_squares, polynomial_least_squares

# Dados de exemplo
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.8, 6.2, 8.1, 9.9])

# Regressão linear
a, b = linear_least_squares(x, y)
print(f"Equação da reta: y = {a:.4f} + {b:.4f}x")

# Regressão polinomial
degree = 2  # Grau do polinômio
coefs = polynomial_least_squares(x, y, degree)
print(f"Coeficientes do polinômio: {coefs}")
```

## Estrutura do Projeto

```
metodos_numericos_final/
│
├── methods/
│   ├── interpolation/
│   │   ├── __init__.py
│   │   └── interpolation.py
│   ├── root/
│   │   ├── __init__.py
│   │   ├── bracketing_methods.py
│   │   └── open_methods.py
│   └── regression/
│       ├── __init__.py
│       └── regression.py
│
├── utils/
│   ├── __init__.py
│   ├── parser.py
│   ├── error.py
│   └── matrix.py
│
├── tests/
│   └── ...
│
├── environment.yml
└── README.md
```


## Requisitos
- Python 3.8 ou superior
- NumPy
- SciPy
- SymPy
- Matplotlib (para visualização)

## Contribuição
Contribuições são bem-vindas! Por favor, sinta-se à vontade para enviar um Pull Request.

## Licença
Este projeto está licenciado sob a licença MIT - consulte o arquivo LICENSE para obter detalhes.

## Autor
Matheus - Mestrando em Modelagem Computacional