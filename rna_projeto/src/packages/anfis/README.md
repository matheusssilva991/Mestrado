# ANFIS em PyTorch (Resumo)

Implementação de um sistema neuro‑fuzzy ANFIS (Sugeno ordem 1) em PyTorch para classificação ou regressão sobre vetores de atributos (ex.: features GLCM).

Base de código derivada principalmente dos dois primeiros repositórios estudados:
- fraunhofer / my-anfis-pytorch
- jfpower / anfis-pytorch

Outras fontes (gregorLen / S-ANFIS-PyTorch, pyanfis) foram consultadas apenas para referência conceitual e comparação, não para extração direta de código.

Características:
- Funções de pertinência gaussianas parametrizadas.
- Geração automática de regras (produto cartesiano).
- Consequentes lineares TSK.
- Suporte a modo híbrido (MFs por gradiente + consequentes via mínimos quadrados) ou treino direto por backprop.
- Uso simples para classificação binária (1 logit + BCEWithLogitsLoss) ou multiclasse (n_out + CrossEntropyLoss).

Aplicação recomendada:
- Dados tabulares normalizados.
- Casos com necessidade de interpretabilidade por regras.
- Estudos acadêmicos envolvendo GLCM ou outros descritores.

Referência conceitual principal: J.-S. R. Jang (1993) ANFIS: Adaptive-Network-Based Fuzzy Inference System.

Uso acadêmico. Verifique licenças originais se redistribuir.
