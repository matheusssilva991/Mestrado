# Projeto Final - Redes Neurais Artificiais (ARCADE Dataset)

## 📋 Descrição

Projeto de detecção e segmentação de doenças arteriais coronárias utilizando o dataset ARCADE (Automatic Region-based Coronary Artery Disease Diagnostics using X-ray Angiography). Este projeto implementa modelos de deep learning para análise de imagens de angiografia coronária por raios-X.

## 🎯 Objetivos

O dataset ARCADE fornece um recurso anotado em larga escala para desenvolvimento e benchmarking de modelos de IA no diagnóstico de doença arterial coronariana (CAD), incluindo:

- **3.000 frames** de angiografia coronária por raios-X anonimizados
- **Duas tarefas principais:**
  1. **Segmentação e classificação de vasos** (1.500 imagens) usando metodologia SYNTAX score
  2. **Detecção de estenose** (1.500 imagens) com placas ateroscleróticas anotadas
- **Anotações** criadas e validadas por cardiologistas experientes usando CVAT
- **Origem**: Pacientes reais (19-90 anos) do Research Institute of Cardiology and Internal Diseases (Almaty, Cazaquistão)
- **Equipamentos**: Philips Azurion 3 e Siemens Artis Zee
- **Resolução**: 512 × 512 pixels

## 🚀 Funcionalidades

- ✅ Carregamento e preprocessamento de dados ARCADE
- ✅ Transformações customizadas para imagens médicas
- ✅ Implementação de modelos de segmentação e classificação
- ✅ Sistema ANFIS (Adaptive Neuro-Fuzzy Inference System) para análise
- ✅ Utilidades para manipulação de bounding boxes
- ✅ Visualizações e métricas de avaliação
- ✅ Exportação de anotações em diferentes formatos

## 📂 Estrutura do Projeto

```
rna_projeto/
├── src/
│   ├── main.ipynb              # Notebook principal
│   ├── preprocessing.ipynb     # Preprocessamento de dados
│   └── packages/
│       ├── __init__.py
│       ├── bbox.py             # Manipulação de bounding boxes
│       ├── loader.py           # Carregamento de dados
│       ├── transformers.py     # Transformações de imagens
│       ├── utils.py            # Utilitários gerais
│       ├── anfis/              # Sistema ANFIS
│       └── plots/              # Visualizações
├── output/
│   ├── classification_annotations.csv
│   ├── classification_annotations_reshuffled.csv
│   ├── stenosis_annotations.csv
│   ├── syntax_annotations.csv
│   └── cache_data/             # Dados em cache (features, arrays)
├── docs/                       # Documentação
├── pyproject.toml              # Dependências do projeto
└── README.md
```

## 🛠️ Requisitos

- **Python**: >= 3.13
- **Principais dependências**:
  - `torch >= 2.9.1` - PyTorch para deep learning
  - `torchvision >= 0.24.1` - Visão computacional com PyTorch
  - `numpy >= 2.3.5` - Computação numérica
  - `pandas >= 2.3.3` - Manipulação de dados
  - `opencv-python >= 4.11.0.86` - Processamento de imagens
  - `scikit-image >= 0.25.2` - Análise de imagens
  - `scikit-learn >= 1.7.2` - Machine learning
  - `pycocotools >= 2.0.10` - Ferramentas COCO
  - `matplotlib >= 3.10.7` - Visualização
  - `seaborn >= 0.13.2` - Visualização estatística

## 📦 Instalação

```bash
# Clone o repositório
git clone https://github.com/matheusssilva991/trabalho_final_rna_mestrado.git

# Entre no diretório
cd trabalho_final_rna_mestrado/rna_projeto

# Crie um ambiente virtual
python -m venv .venv

# Ative o ambiente virtual
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Instale as dependências
pip install -e .
```

## 🎮 Como Usar

### Preprocessamento

```python
# Execute o notebook de preprocessamento
jupyter notebook src/preprocessing.ipynb
```

### Treinamento e Avaliação

```python
# Execute o notebook principal
jupyter notebook src/main.ipynb
```

### Usando as Ferramentas

```python
from packages.loader import load_arcade_dataset
from packages.transformers import MedicalImageTransform
from packages.bbox import BoundingBox

# Carregar dados
dataset = load_arcade_dataset()

# Aplicar transformações
transform = MedicalImageTransform()
transformed_data = transform(dataset)

# Manipular bounding boxes
bbox = BoundingBox(x, y, width, height)
```

## 📊 Dados de Saída

Os resultados são salvos na pasta `output/`:

- **Anotações**: CSVs com classificações e detecções
- **Cache**: Features extraídas e arrays numpy para treinamento rápido
- **Visualizações**: Gráficos e métricas de performance

## 📚 Referências

- **ARCADE Dataset**: Introduzido no ARCADE Challenge @ MICCAI 2023
- **SYNTAX Score**: Metodologia para avaliação de doença arterial coronariana
- Dataset disponível para pesquisa em análise de imagens médicas, deep learning e diagnóstico automatizado

## 👤 Autor

Matheus Silva - [GitHub](https://github.com/matheusssilva991)

## 📄 Licença

Este projeto é parte do trabalho de mestrado.
