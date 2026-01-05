# Redes Neurais Artificiais - Tarefas de Mestrado

Implementação de redes neurais artificiais (Perceptron e Wavelon) com otimização por Levenberg-Marquardt para tarefas de ajuste de curva e regressão.

## 📋 Descrição

Este projeto implementa duas arquiteturas de redes neurais:
- **Perceptron Multicamadas (MLP)**: Com função de ativação tangente hiperbólica
- **Rede Wavelon**: Utilizando wavelets como funções de ativação (Mexican Hat)

Ambas as redes são treinadas usando o algoritmo de **Levenberg-Marquardt** para otimização de pesos.

## 🚀 Funcionalidades

- ✅ Treinamento de redes com qualquer número de features de entrada
- ✅ Múltiplos métodos de inicialização de pesos (Heurístico, RBS/K-Means)
- ✅ Normalização de dados (MinMax e Standard Scaler)
- ✅ Métricas de avaliação (MSE, RMSE, MAE, R²)
- ✅ Validação cruzada (K-Fold)
- ✅ Visualizações interativas (Plotly)
- ✅ Suporte para ajuste de superfícies 2D e 3D

## 📂 Estrutura do Projeto

```
rna_tarefas_mestrado/
├── data/                           # Datasets
│   ├── Trabalho4dados.xlsx
│   ├── Trabalho5dados.xlsx
│   └── ...
├── libs/                           # Bibliotecas implementadas
│   ├── activations/                # Funções de ativação
│   │   └── activations.py
│   ├── metrics/                    # Métricas de avaliação
│   │   └── metrics.py
│   ├── networks/                   # Arquiteturas de redes
│   │   ├── perceptron/
│   │   │   ├── __init__.py
│   │   │   ├── neuron.py           # Neurônio básico
│   │   │   ├── forward.py          # Forward pass
│   │   │   ├── loss.py             # Funções de perda
│   │   │   ├── gradients.py        # Jacobiana
│   │   │   ├── train.py            # Treinamento
│   │   │   └── unflatten_weights.py
│   │   └── wavelet/
│   │       ├── __init__.py
│   │       ├── wavelon.py          # Neurônio wavelon
│   │       ├── forward.py          # Forward pass
│   │       ├── loss.py             # Funções de perda
│   │       ├── gradients.py        # Jacobiana
│   │       ├── train.py            # Treinamento
│   │       ├── initialization.py   # Inicialização de pesos
│   │       └── unflatten_weights.py
│   ├── normalizers/                # Normalizadores
│   │   ├── min_max_scaler.py
│   │   └── stantard_scaler.py
│   └── optimizers/                 # Algoritmos de otimização
│       └── levenberg_marquadt.py
├── src/                            # Notebooks de experimentos
│   ├── tarefa4.ipynb               # Ajuste com Perceptron
│   ├── tarefa5.ipynb               # Ajuste com validação
│   ├── tarefa6.ipynb               # Validação cruzada
│   └── tarefa7_a.ipynb             # Ajuste com Wavelon
├── utils/                          # Utilitários
│   └── plots/
│       ├── plots_tarefa4.py
│       └── plots_tarefa5.py
└── README.md
```

## 🛠️ Instalação

### Pré-requisitos
- Python 3.8+
- pip

### Dependências

```bash
pip install numpy pandas matplotlib plotly scipy scikit-learn openpyxl
```

## 📖 Uso

### 1. Treinamento de Rede Perceptron

```python
from networks.perceptron import train_network, hidden_forward, unflatten_weights
import numpy as np

# Dados de exemplo
X1 = np.linspace(0, 1, 100)
X2 = np.linspace(0, 1, 100)
y = 0.5 + 0.4 * np.sin(2 * np.pi * X1) + 0.3 * np.cos(2 * np.pi * X2)

# Treinar rede
weights_flat, losses, n_iters = train_network(
    X1, X2,
    y=y,
    n_neurons=10,
    n_epochs=50,
    n_iterations_per_epoch=1000,
    tolerance=1e-6,
    alpha=1e-3
)

# Desempacotar pesos
neurons_weights = unflatten_weights(weights_flat, n_inputs=2, n_neurons=10)

# Predição
y_pred = hidden_forward(X1_test, X2_test, neurons_weights=neurons_weights, activation_fn=np.tanh)
```

### 2. Treinamento de Rede Wavelon

```python
from networks.wavelet import train_network_wavelet, hidden_forward, unflatten_weights
from activations.activations import mexican_hat_wavelet

# Treinar rede
weights_flat, losses, n_iters = train_network_wavelet(
    X1, X2,
    y=y,
    n_neurons=5,
    n_iterations_per_epoch=1000,
    tolerance=1e-9,
    alpha=1e-3,
    weights_init_method="heuristic"  # ou "rbs"
)

# Desempacotar pesos
neurons_params, output_weights = unflatten_weights(weights_flat, n_inputs=2, n_neurons=5)

# Predição
y_pred = hidden_forward(
    X1_test, X2_test,
    neurons_params=neurons_params,
    wavelet_fn=mexican_hat_wavelet,
    output_weights=output_weights
)
```

### 3. Normalização de Dados

```python
from normalizers.min_max_scaler import MinMaxScaler

# Criar scaler
scaler = MinMaxScaler(-1, 1)

# Ajustar e transformar
scaler.fit(X_train)
X_train_scaled = scaler.normalize(X_train)
X_test_scaled = scaler.normalize(X_test)

# Desnormalizar predições
y_pred = scaler.denormalize(y_pred_scaled)
```

## 📊 Métricas de Avaliação

```python
from metrics.metrics import MSE, RMSE, MAE, R2_score

mse = MSE(y_true, y_pred)
rmse = RMSE(y_true, y_pred)
mae = MAE(y_true, y_pred)
r2 = R2_score(y_true, y_pred)
```

## 🔬 Experimentos

### Tarefa 4: Ajuste de Superfície com Perceptron
- Dataset: `Trabalho4dados.xlsx`
- Objetivo: Ajustar superfície 3D usando MLP
- Técnicas: Normalização, otimização LM

### Tarefa 5: Interpolação com Validação
- Dataset: `Trabalho5dados.xlsx`
- Objetivo: Interpolar dados esparsos
- Técnicas: Train/Val/Test split, busca de hiperparâmetros

### Tarefa 6: Validação Cruzada
- Dataset: `Trabalho5dados.xlsx`
- Objetivo: Avaliação robusta do modelo
- Técnicas: K-Fold Cross-Validation (5 folds)

### Tarefa 7: Ajuste com Wavelon
- Dataset: Função senoidal com ruído
- Objetivo: Comparar Wavelon vs Perceptron
- Técnicas: Wavelets Mexican Hat, inicialização heurística

## 📈 Algoritmo de Otimização

### Levenberg-Marquardt

O algoritmo combina:
- **Gauss-Newton**: Convergência rápida perto do mínimo
- **Gradient Descent**: Estabilidade longe do mínimo

**Equação de atualização:**
```
w_{k+1} = w_k - (J^T J + λI)^{-1} J^T r
```

Onde:
- `J`: Jacobiana dos resíduos
- `r`: Vetor de resíduos (y_true - y_pred)
- `λ`: Parâmetro de regularização (damping)

**Critérios de parada:**
1. Norma do gradiente < tolerância
2. Mudança relativa nos pesos < tolerância
3. Mudança relativa na loss < tolerância

## 🎯 Funções de Ativação

### Tangente Hiperbólica
```python
φ(z) = tanh(z) = (e^z - e^{-z}) / (e^z + e^{-z})
φ'(z) = 1 - tanh²(z)
```

### Mexican Hat Wavelet
```python
ψ(z) = (1 - z²) * exp(-z²/2)
ψ'(z) = -z * (3 - z²) * exp(-z²/2)
```

## 🧪 Testes

Para executar os notebooks:

```bash
jupyter notebook src/tarefa4.ipynb
```

## 👨‍💻 Autor

Matheus - Mestrando em Modelagem Computacional

## 📄 Licença

Este projeto é para fins acadêmicos.

## 🤝 Contribuições

Sugestões e melhorias são bem-vindas!

---

**Status do Projeto:** 🚧 Em desenvolvimento ativo
