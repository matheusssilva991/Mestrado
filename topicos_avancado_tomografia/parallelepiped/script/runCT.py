"""
Script de Python para rodar uma simulação de TC usando o código XRMC

Para rodar o script: python runCT.py param1 param2 param3
- param1 (targetFolder), diretorio com os arquivos de simulação do XRMC, input.dat e outros
- param2 (folderOutput), diretorio onde serão armazenados os arquivos raw
- param3 (angleStep), passo de rotação numérico

Autor: Dany S. Dominguez
Data: 28/12/2023

Adaptado para Windows
"""

import os
import sys
import math
import shutil
import subprocess
from datetime import datetime
from script.utils.files import wait_for_file_stable, replace_rotate_angle


# Verificando se os parâmetros da entrada da CL são corretos
if len(sys.argv) != 4:
    print("Code need 3 arguments: targetFolder folderOutput angleStep")
    sys.exit(1)

# Inicialização de variáveis
currentAngle = 0.0
targetFolder = os.path.abspath(sys.argv[1])
folderOutput = os.path.abspath(sys.argv[2])
try:
    angleStep = float(sys.argv[3])
except ValueError:
    print("angleStep must be a number")
    sys.exit(1)

if not os.path.isdir(targetFolder):
    print(f"Target folder does not exist: {targetFolder}")
    sys.exit(1)

# Criando a pasta de saida das imagens de simulação
os.makedirs(folderOutput, exist_ok=True)
print(
    f"Init Angle: {currentAngle}\n Dir: {targetFolder}\n Output: {folderOutput}\n Angle Step: {angleStep}"
)
os.chdir(targetFolder)

# Iniciando a simulação, carimbo de tempo
print("Starting simulation...")
nowI = datetime.now()
print("Init =", nowI.strftime("%d/%m/%Y %H:%M:%S"))

# Nome do arquivo quadricas (ajuste se necessário)
quadric_file = "quadric.dat"
image_file = "image.dat"

# Loop principal de simulação
numProj = int(math.ceil(360.0 / angleStep))
for i in range(numProj):
    currAngle = i * angleStep
    nextAngle = currAngle + angleStep
    print(f"Processing {i + 1}/{numProj}. Angle = {currAngle:.1f}")

    # Executa xrmc; stdout/stderr descartados (cross-platform)
    try:
        subprocess.run(
            ["xrmc", "input.dat"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )
    except FileNotFoundError:
        print("xrmc executable not found in PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("xrmc failed:", e)
        sys.exit(1)

    # Aguarda o arquivo ser criado e estabilizar (escrito)
    if not wait_for_file_stable(image_file, timeout=30):
        print("Timeout waiting for image.dat to be written.")
        sys.exit(1)

    # Move e renomeia
    imgName = f"img_{i:04d}.dat"
    dest = os.path.join(folderOutput, imgName)
    try:
        shutil.move(image_file, dest)
    except Exception as e:
        print("Failed to move image:", e)
        sys.exit(1)

    # Atualiza o ângulo no arquivo de quadricas
    try:
        replace_rotate_angle(quadric_file, nextAngle)
    except Exception as e:
        print("Failed to update quadric file:", e)
        sys.exit(1)

# Reinicializa o arquivo de quadricas para a primeira projeção, angulo zero
try:
    replace_rotate_angle(quadric_file, 0.0)
except Exception as e:
    print("Failed to reset quadric file:", e)
    sys.exit(1)

# Encerra a simulação, carimbo de tempo
print("End simulation...")
nowF = datetime.now()
print("End =", nowF.strftime("%d/%m/%Y %H:%M:%S"))
deltaTime = nowF - nowI
print(f"Elapsed time in sec: {deltaTime.total_seconds()}")
