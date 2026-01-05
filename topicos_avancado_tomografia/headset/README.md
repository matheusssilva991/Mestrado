# Reconstrução Tomográfica do Headset

Projeto de reconstrução tomográfica (CT) de um objeto tipo “headset”, utilizando preprocessamento de projeções (flat/dark, alinhamento), transformação logarítmica (Lei de Beer-Lambert) e algoritmos de reconstrução do pacote TIGRE (FDK e OSSART). Os fluxos principais estão organizados em notebooks em `src/`.

## Visão Geral

- **Dados brutos**: arquivos de projeções em `data/projections/` e imagens de flat/dark em `data/darkflat/`.
- **Processamento**: leitura, correção flat/dark, detecção/correção de desalinhamento, transformação log, definição de geometria e reconstrução (FDK/OSSART).
- **Resultados**: volumes reconstruídos e sinogramas corrigidos salvos em `output/`, além de visualizações K3D em `output/visualizations/` e exportação opcional em MHD/RAW.

## Estrutura do Repositório

```
pyproject.toml
README.md
data/
  darkflat/
    dark01.txt, dark02.txt, dark03.txt
    flat01.txt, flat02.txt, flat03.txt
  projections/
    img1.txt, img2.txt, ..., img218.txt, ...
imgs/
output/
  projections/
  visualizations/
src/
  load_reconstruction.ipynb
  results.ipynb
  utils/
```

## Requisitos

- **Python**: `>= 3.13` (ver `pyproject.toml`).
- **Dependências (pip)**:
  - `numpy`, `matplotlib`, `opencv-python`, `pandas`
  - `ipykernel`, `ipywidgets (<8)`, `k3d`
  - `simpleitk`
  - `pip` (atualizado)
- **TIGRE (Tomographic Iterative GPU-based Reconstruction)**:
  - Necessário para `FDK` e `OSSART`. Instalação via `pip install tigre` ou conforme a documentação oficial: <https://github.com/CERN/TIGRE>
  - Recomenda-se ambiente com CUDA/GPU para desempenho adequado (CPU funciona, porém mais lento).

## Instalação

Crie um ambiente virtual e instale as dependências do projeto:

```bash
# No diretório raiz do projeto
python -m venv .venv
source .venv/bin/activate

# Instalar dependências listadas no pyproject
pip install -e .

# Instalar TIGRE (se ainda não estiver instalado)
pip install tigre
# (Opcional) instalar CuPy compatível com sua versão de CUDA
# Ex.: pip install cupy-cuda12x
```

## Dados de Entrada

- **Projeções** em `data/projections/`: arquivos `imgN.txt` (N = 1..Nmax), um por ângulo.
- **Flat/Dark** em `data/darkflat/`: `flat01.txt`, `flat02.txt`, `flat03.txt` e `dark01.txt`, `dark02.txt`, `dark03.txt`.
- Ajuste os caminhos no notebook conforme necessário (`projections_path`, `darkflat_path`).

## Fluxo de Trabalho

1. **Reconstrução** — abra e execute o notebook principal:
   - [src/load_reconstruction.ipynb](src/load_reconstruction.ipynb)
   - Etapas principais:
     - Leitura de flats/darks e cálculo das médias.
     - Leitura das projeções com seleção de ângulos.
     - Correção flat/dark e recorte de ROI.
     - Detecção/correção de desalinhamento (0° vs 180°).
     - Transformação logarítmica (Beer–Lambert).
     - Definição de geometria (`tigre.geometry_default`) e ângulos.
     - Reconstrução: `FDK` e `OSSART`.
     - Salvamento: volumes (`*.npz`), sinograma corrigido e exportação `MHD/RAW`.

2. **Análise de Resultados** — abra:
   - [src/results.ipynb](src/results.ipynb)
   - Carrega volumes `.npz`, gera mosaicos, exporta visualizações K3D (`.html`) e calcula métricas (**SNR**, **PSNR**).

### Observação sobre caminhos de saída

- O notebook de reconstrução salva em `output/` por padrão.
- O notebook de resultados usa caminhos em `output/projections/`.
- Se necessário, ajuste os caminhos no início dos notebooks para mantê-los consistentes, ou mova/copied os arquivos `.npz` para `output/projections/`.

## Execução Rápida

```bash
# Ativar ambiente
source .venv/bin/activate

# Iniciar Jupyter Lab ou Notebook
jupyter lab
# ou
jupyter notebook

# Abra src/load_reconstruction.ipynb e execute as células
# Depois abra src/results.ipynb para análise e visualização
```

## Principais Arquivos

- Notebook de reconstrução: [src/load_reconstruction.ipynb](src/load_reconstruction.ipynb)
- Notebook de resultados: [src/results.ipynb](src/results.ipynb)
- Utilitários (I/O, visualização, métricas): `src/utils/`

## Exportação e Visualização

- O notebook de reconstrução pode exportar o volume normalizado como `MHD/RAW` (útil no ParaView/ITK-SNAP).
- O notebook de resultados salva visualizações interativas K3D em `output/visualizations/*.html`.

## Dicas e Solução de Problemas

- **`ModuleNotFoundError: tigre`**: instale TIGRE (`pip install tigre`) e verifique CUDA/GPU se desejar aceleração.
- **Arquivos ausentes**: confirme que as pastas `data/projections/` e `data/darkflat/` existem e contêm os arquivos esperados.
- **Desempenho**: OSSART é iterativo e pode ser mais lento; use GPU para acelerar.
- **Geometria**: parâmetros (`DSD`, `DSO`, `nDetector`, `dDetector`, `nVoxel`, `sVoxel`) devem refletir seu setup experimental.

## Referências

- TIGRE — Tomographic Iterative GPU-based Reconstruction Toolbox: <https://github.com/CERN/TIGRE>
