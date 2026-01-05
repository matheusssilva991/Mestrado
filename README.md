# Projetos de Mestrado

Repositório unificado contendo todos os projetos desenvolvidos durante o programa de mestrado, abrangendo disciplinas de Métodos Numéricos, Redes Neurais Artificiais e Tópicos Avançados em Tomografia.

## 📚 Projetos

### 1. [Métodos Numéricos](metodos_numericos_1/)

Biblioteca Python para implementação de métodos numéricos fundamentais.

**Tópicos abordados:**

- 📐 **Interpolação**: Lagrange, Taylor, Splines Cúbicos
- 🔍 **Busca de Raízes**: Bisseção, Posição Falsa, Newton-Raphson, Secante, Ponto Fixo
- 📊 **Regressão**: Linear, Polinomial, com métricas estatísticas
- 🔢 **Sistemas Lineares**: Métodos diretos e iterativos
- 🧮 **Representação Numérica**: Análise de erro e propagação

**Tecnologias:** Python, NumPy, Matplotlib, Jupyter

[📖 Documentação completa](metodos_numericos_1/README.md)

---

### 2. [Redes Neurais Artificiais - Tarefas](rna_tarefas/)

Implementações de redes neurais com algoritmo de Levenberg-Marquardt para tarefas de ajuste de curva e regressão.

**Arquiteturas implementadas:**

- 🧠 **Perceptron Multicamadas (MLP)**: Função tangente hiperbólica
- 🌊 **Rede Wavelon**: Wavelets Mexican Hat como ativação
- 🔧 **Sistema ANFIS**: Adaptive Neuro-Fuzzy Inference System

**Funcionalidades:**

- ✅ Otimização Levenberg-Marquardt
- ✅ Inicialização heurística e RBF/K-Means
- ✅ Normalização (MinMax, Standard Scaler)
- ✅ Validação cruzada (K-Fold)
- ✅ Métricas: MSE, RMSE, MAE, R²
- ✅ Visualizações interativas (Plotly)

**Tecnologias:** Python, NumPy, SciPy, Plotly, Jupyter

[📖 Documentação completa](rna_tarefas/README.md)

---

### 3. [Projeto Final RNA - Dataset ARCADE](rna_projeto/)

Projeto de análise de imagens médicas para detecção de doenças arteriais coronárias usando deep learning.

**Dataset ARCADE:**

- 🏥 3.000 frames de angiografia coronária por raios-X
- 👨‍⚕️ Anotado por cardiologistas experientes
- 🎯 Duas tarefas: segmentação de vasos e detecção de estenose
- 📏 Resolução: 512×512 pixels

**Funcionalidades:**

- ✅ Preprocessamento de imagens médicas
- ✅ Modelos de segmentação e classificação (PyTorch)
- ✅ Sistema ANFIS para análise
- ✅ Manipulação de bounding boxes
- ✅ Avaliação com métricas COCO

**Tecnologias:** Python, PyTorch, OpenCV, scikit-image, COCO Tools

[📖 Documentação completa](rna_projeto/README.md)

---

### 4. [Tópicos Avançados em Tomografia](topicos_avancado_tomografia/)

Projetos de reconstrução tomográfica computadorizada com dados reais e simulados.

#### 4.1. [Headset - Reconstrução Real](topicos_avancado_tomografia/headset/)

Reconstrução de objeto tipo "headset" a partir de projeções reais.

**Pipeline:**

- 📸 Correção flat/dark field
- 🔧 Alinhamento de projeções
- 📈 Transformação logarítmica (Beer-Lambert)
- 🖥️ Reconstrução FDK e OSSART (TIGRE)
- 🎨 Visualização 3D interativa (K3D)

**Tecnologias:** Python, TIGRE, OpenCV, SimpleITK, K3D

[📖 Documentação completa](topicos_avancado_tomografia/headset/README.md)

#### 4.2. [Parallelepiped - Simulação](topicos_avancado_tomografia/parallelepiped/)

Simulação completa de aquisição e reconstrução tomográfica.

**Funcionalidades:**

- 🎲 Geração de sinogramas sintéticos
- ⚡ Simulação de espectro de raios-X
- 🔄 Reconstrução FDK e iterativa
- 📊 Análise de qualidade (PSNR, SSIM)
- 📈 Comparação de algoritmos

**Tecnologias:** Python, TIGRE, Nibabel, Silx

[📖 Documentação completa](topicos_avancado_tomografia/parallelepiped/README.md)

---

## 🛠️ Tecnologias Utilizadas

### Linguagens

- 🐍 Python 3.13+

### Computação Científica

- NumPy, SciPy, Pandas
- Matplotlib, Seaborn, Plotly

### Machine Learning & Deep Learning

- PyTorch, TorchVision
- scikit-learn, scikit-image

### Processamento de Imagens

- OpenCV
- SimpleITK, Nibabel
- COCO Tools

### Reconstrução Tomográfica

- TIGRE (GPU-accelerated)
- Silx

### Visualização

- Jupyter Notebooks
- K3D (3D interativo)
- Plotly (gráficos interativos)

---

## 📦 Instalação Geral

Cada projeto possui suas próprias dependências. Instruções gerais:

```bash
# Clone o repositório completo
git clone https://github.com/matheusssilva991/mestrado.git
cd mestrado

# Entre no projeto desejado
cd <nome_do_projeto>

# Crie um ambiente virtual
python -m venv .venv

# Ative o ambiente
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Instale as dependências
pip install -e .
# ou
conda env create -f environment.yml  # Se disponível
```

---

## 📂 Estrutura do Workspace

```
Mestrado/
├── metodos_numericos_1/        # Métodos numéricos fundamentais
│   ├── methods/                # Implementações de algoritmos
│   ├── utils/                  # Utilitários (erro, parsing, etc)
│   ├── src/                    # Notebooks de exemplos
│   └── README.md
│
├── rna_tarefas/                # Tarefas de RNA
│   ├── libs/                   # Bibliotecas (networks, optimizers)
│   ├── utils/                  # Utilitários de visualização
│   ├── src/                    # Notebooks das tarefas
│   └── README.md
│
├── rna_projeto/                # Projeto final RNA (ARCADE)
│   ├── src/                    # Código principal e notebooks
│   ├── output/                 # Resultados e cache
│   └── README.md
│
├── topicos_avancado_tomografia/  # Tomografia
│   ├── headset/                # Projeto reconstrução real
│   ├── parallelepiped/         # Projeto simulação
│   └── README.md
│
└── README.md                   # Este arquivo
```

---

## 🎓 Disciplinas

| Projeto | Disciplina | Tópicos Principais |
|---------|-----------|-------------------|
| `metodos_numericos_1` | Métodos Numéricos | Interpolação, Raízes, Regressão, SELA |
| `rna_tarefas` | Redes Neurais Artificiais | MLP, Wavelon, ANFIS, Levenberg-Marquardt |
| `rna_projeto` | Projeto Final RNA | Deep Learning, Segmentação, Detecção |
| `topicos_avancado_tomografia` | Tópicos Avançados em Tomografia | Reconstrução CT, FDK, OSSART, Simulação |

---

## 📊 Resumo de Funcionalidades

### Métodos Numéricos

- Interpolação polinomial e splines
- Encontrar raízes de equações
- Ajuste de curvas por regressão
- Resolução de sistemas lineares
- Análise de erros numéricos

### Redes Neurais

- Perceptron com backpropagation
- Wavelon com funções wavelet
- Otimização Levenberg-Marquardt
- Validação cruzada K-Fold
- Visualizações de superfícies

### Projeto ARCADE

- Segmentação de vasos coronários
- Detecção de estenose
- Sistema ANFIS
- Métricas de avaliação médica
- Preprocessamento de imagens

### Tomografia

- Reconstrução analítica (FDK)
- Reconstrução iterativa (OSSART/SART)
- Simulação de espectro de raios-X
- Correção de artefatos
- Visualização 3D de volumes

---

## 🚀 Quickstart

### Métodos Numéricos

```bash
cd metodos_numericos_1
conda env create -f environment.yml
conda activate metodos_numericos
jupyter notebook src/
```

### RNA Tarefas

```bash
cd rna_tarefas
pip install -e .
jupyter notebook src/tarefa1.ipynb
```

### RNA Projeto (ARCADE)

```bash
cd rna_projeto
pip install -e .
jupyter notebook src/preprocessing.ipynb
```

### Tomografia

```bash
cd topicos_avancado_tomografia/headset
pip install -e .
# Instale TIGRE: pip install tigre
jupyter notebook src/
```

---

## 📈 Resultados e Publicações

Os notebooks de cada projeto contêm análises detalhadas, visualizações e resultados experimentais. Consulte a documentação específica de cada projeto para mais informações.

---

## 👤 Autor

**Matheus Silva**

- GitHub: [@matheusssilva991](https://github.com/matheusssilva991)
- Repositórios:
  - [metodos_numericos_final](https://github.com/matheusssilva991/metodos_numericos_final)
  - [rna_tarefas_mestrado](https://github.com/matheusssilva991/rna_tarefas_mestrado)
  - [trabalho_final_rna_mestrado](https://github.com/matheusssilva991/trabalho_final_rna_mestrado)
  - [TA_tomografia_mestrado](https://github.com/matheusssilva991/TA_tomografia_mestrado)

---

## 📄 Licença

Projetos acadêmicos desenvolvidos como parte do programa de mestrado. Para uso educacional e de pesquisa.

---

## 🙏 Agradecimentos

- Professores e orientadores das disciplinas
- Comunidades open-source (NumPy, PyTorch, TIGRE, etc.)
- Colaboradores e colegas do mestrado

---

## 📞 Contato

Para dúvidas ou colaborações relacionadas a estes projetos, entre em contato através do GitHub.

---

**Última atualização:** Janeiro 2026
