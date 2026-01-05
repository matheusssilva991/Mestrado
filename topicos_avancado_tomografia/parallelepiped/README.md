# Parallelepiped - Simulação Tomográfica

## 📋 Descrição

Projeto de simulação completa de aquisição e reconstrução tomográfica de um paralelepípedo. Este projeto implementa desde a geração de sinogramas sintéticos até a reconstrução volumétrica, incluindo simulação de espectro de raios-X e análise de qualidade de imagens.

## 🎯 Objetivos

- Simular o processo completo de tomografia computadorizada
- Gerar sinogramas sintéticos de objetos geométricos simples
- Implementar simulação de espectro de raios-X contínuo
- Reconstruir volumes 3D usando algoritmos do TIGRE
- Analisar artefatos e qualidade de reconstrução
- Estudar o impacto de diferentes parâmetros de aquisição

## 🚀 Funcionalidades

- ✅ Geração de sinogramas 2D e 3D
- ✅ Simulação de espectro policromático de raios-X
- ✅ Reconstrução FDK (Feldkamp-Davis-Kress)
- ✅ Reconstrução iterativa (OSSART, SART)
- ✅ Visualização 3D interativa de volumes
- ✅ Análise de perfis de intensidade
- ✅ Métricas de qualidade de imagem
- ✅ Suporte para diferentes geometrias de aquisição

## 📂 Estrutura do Projeto

```
parallelepiped/
├── script/
│   ├── Tigre.ipynb              # Reconstrução com TIGRE
│   ├── generateSino.ipynb       # Geração de sinogramas
│   ├── runCT.py                 # Simulação CT completa
│   ├── spectrum.ipynb           # Análise espectral
│   ├── visualize_volume.ipynb   # Visualização 3D
│   └── utils/                   # Utilitários
├── output/                      # Resultados e visualizações
├── ContinuousSpectrumFile.txt   # Espectro contínuo
├── ContinuousSpectrumFileRaw.txt
├── Makefile.am
├── pyproject.toml
└── README.md
```

## 🛠️ Requisitos

- **Python**: >= 3.13
- **TIGRE**: Biblioteca de reconstrução tomográfica
  - Requer GPU com CUDA para melhor desempenho
  - Pode funcionar em CPU (mais lento)
- **Principais dependências**:
  - `numpy >= 2.3.4` - Computação numérica
  - `matplotlib >= 3.10.7` - Visualização 2D
  - `pandas >= 2.3.3` - Manipulação de dados
  - `k3d >= 2.17.0` - Visualização 3D interativa
  - `nibabel >= 5.3.2` - I/O de imagens médicas
  - `silx >= 2.2.2` - Análise de dados científicos

## 📦 Instalação

```bash
# Clone o repositório
git clone https://github.com/matheusssilva991/TA_tomografia_mestrado.git

# Entre no diretório do projeto
cd TA_tomografia_mestrado/parallelepiped

# Crie um ambiente virtual
python -m venv .venv

# Ative o ambiente virtual
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Instale as dependências
pip install -e .

# Instale o TIGRE
pip install tigre
```

### Instalação do TIGRE

Para melhor desempenho, instale o TIGRE com suporte CUDA:

```bash
# Verifique se você tem CUDA instalado
nvcc --version

# Instale o TIGRE
pip install tigre

# Ou compile do código-fonte (para otimização específica)
git clone https://github.com/CERN/TIGRE.git
cd TIGRE/Python
pip install .
```

## 🎮 Como Usar

### 1. Geração de Sinogramas

```bash
# Execute o notebook de geração de sinogramas
jupyter notebook script/generateSino.ipynb
```

O notebook permite:

- Definir geometria do objeto (dimensões do paralelepípedo)
- Configurar parâmetros de aquisição (número de projeções, ângulos)
- Gerar sinogramas 2D e 3D
- Visualizar projeções

### 2. Simulação de Espectro

```bash
# Análise do espectro de raios-X
jupyter notebook script/spectrum.ipynb
```

Funcionalidades:

- Carregar espectros pré-definidos
- Simular espectro policromático
- Análise de atenuação para diferentes materiais
- Visualização de perfis espectrais

### 3. Reconstrução com TIGRE

```bash
# Reconstrução volumétrica
jupyter notebook script/Tigre.ipynb
```

Algoritmos disponíveis:

- **FDK**: Reconstrução analítica rápida
- **SART**: Simultaneous Algebraic Reconstruction Technique
- **OSSART**: Ordered Subset SART (mais rápido)
- **CGLS**: Conjugate Gradient Least Squares

### 4. Simulação Completa

```python
# Execute a simulação completa via script
python script/runCT.py
```

### 5. Visualização de Resultados

```bash
# Visualização 3D interativa
jupyter notebook script/visualize_volume.ipynb
```

## 📊 Exemplo de Uso

```python
import numpy as np
import tigre
import tigre.algorithms as algs

# Definir geometria
geo = tigre.geometry()
geo.DSD = 1536  # Distance Source Detector (mm)
geo.DSO = 1000  # Distance Source Origin (mm)
geo.nDetector = np.array([512, 512])  # Pixels do detector
geo.dDetector = np.array([0.8, 0.8])  # Tamanho do pixel (mm)
geo.nVoxel = np.array([256, 256, 256])  # Voxels do volume
geo.dVoxel = np.array([1, 1, 1])  # Tamanho do voxel (mm)

# Definir ângulos de projeção
angles = np.linspace(0, 2*np.pi, 360)

# Gerar phantom (paralelepípedo)
phantom = np.zeros(geo.nVoxel)
phantom[64:192, 64:192, 64:192] = 1  # Cubo central

# Simular projeções
projections = tigre.Ax(phantom, geo, angles)

# Reconstruir com FDK
img_fdk = algs.fdk(projections, geo, angles)

# Reconstruir com OSSART (iterativo)
img_ossart = algs.ossart(projections, geo, angles,
                         niter=50, blocksize=20)
```

## 🔬 Parâmetros de Simulação

### Geometria do Detector

- **Tamanho**: 512x512 pixels
- **Resolução**: 0.8mm/pixel
- **Área ativa**: 409.6 x 409.6 mm²

### Geometria de Aquisição

- **DSD**: 1536mm (Source to Detector)
- **DSO**: 1000mm (Source to Object)
- **Magnificação**: ~1.54x
- **Ângulos**: 360 projeções (0-360°)

### Volume Reconstruído

- **Tamanho**: 256x256x256 voxels
- **Resolução**: 1mm³/voxel
- **FOV**: 256 x 256 x 256 mm³

## 📈 Análise de Qualidade

Métricas implementadas:

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **RMSE** (Root Mean Square Error)
- **Análise de perfis** de intensidade
- **Histogramas** de valores de voxels

## 🎨 Visualizações

O projeto inclui:

- Sinogramas 2D e 3D
- Projeções individuais
- Volumes 3D interativos (K3D)
- Slices axiais, coronais e sagitais
- Gráficos de perfis de intensidade
- Comparação lado-a-lado de algoritmos

## ⚙️ Configuração Avançada

### Otimização de Parâmetros OSSART

```python
# Ajustar parâmetros para melhor qualidade
niter = 50           # Número de iterações
blocksize = 20       # Tamanho dos blocos (ordered subsets)
lambda_param = 1.0   # Parâmetro de relaxação
```

### Simulação de Ruído

```python
# Adicionar ruído Poisson às projeções
import numpy as np

photon_count = 1e5  # Fótons por pixel
projections_noisy = np.random.poisson(
    projections * photon_count) / photon_count
```

## 📚 Referências Teóricas

1. **Feldkamp Algorithm (FDK)**
   - Feldkamp et al., "Practical cone-beam algorithm", JOSA A, 1984
   - Algoritmo padrão para reconstrução cone-beam

2. **SART (Simultaneous Algebraic Reconstruction Technique)**
   - Andersen & Kak, "Simultaneous Algebraic Reconstruction Technique", 1984
   - Método iterativo para reconstrução

3. **TIGRE Toolbox**
   - Biguri et al., "TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction"
   - <https://github.com/CERN/TIGRE>

4. **Espectro de Raios-X**
   - Boone & Seibert, "An accurate method for computer-generating tungsten anode x-ray spectra"
   - Modelos de espectro policromático

## 🐛 Troubleshooting

### TIGRE não encontrado

```bash
# Verifique a instalação
python -c "import tigre; print(tigre.__version__)"

# Se falhar, reinstale
pip install --upgrade tigre
```

### Erro de memória GPU

```python
# Reduza o tamanho do volume ou número de projeções
geo.nVoxel = np.array([128, 128, 128])  # Menor
angles = np.linspace(0, 2*np.pi, 180)   # Menos projeções
```

### Reconstrução lenta

```python
# Use FDK ao invés de iterativo para testes rápidos
img = algs.fdk(projections, geo, angles)  # Mais rápido

# Ou reduza iterações OSSART
img = algs.ossart(projections, geo, angles, niter=20)  # Menos iterações
```

## 👤 Autor

Matheus Silva - [GitHub](https://github.com/matheusssilva991)

## 📄 Licença

Este projeto é parte do trabalho de mestrado em Tópicos Avançados em Tomografia.

## 🤝 Contribuições

Este é um projeto acadêmico, mas sugestões e melhorias são bem-vindas através de issues no GitHub.
