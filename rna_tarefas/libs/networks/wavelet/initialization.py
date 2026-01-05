import numpy as np
from .wavelon import wavelon
from numpy.typing import NDArray
from activations.activations import mexican_hat_wavelet
import pandas as pd


def create_candidate_library(X, wavelet_fn=mexican_hat_wavelet):
    """
    Cria uma biblioteca de candidatos a wavelons com base nos dados de entrada.

    Parâmetros
    ----------
    X : arrays
        Features de entrada (x1, x2, ..., xn)
    wavelet_fn : callable
        Função wavelet-mãe

    Retorna
    -------
    H : ndarray
        Biblioteca de saídas dos wavelons (n_samples, n_candidates)
    potential_centers : ndarray
        Centros candidatos (n_candidates, n_features),
    global_candidate_scale : ndarray
        Escala candidata global (1, n_features)
    """

    # Preparação dos Dados de Entrada
    n_samples, n_features = X.shape

    # Definir Centros Candidatos (t)
    potential_centers = X  # (n_samples, n_features)

    # Definir Escala Candidata Global (s)
    M = X.max(axis=0)
    N = X.min(axis=0)
    global_candidate_scale = 0.2 * (M - N) + 1e-8 # (1, n_features)

    # Inicializar a Biblioteca de Saídas dos Wavelons (H)
    H = np.zeros((n_samples, n_samples))

    # Calcular a Saída de Cada Wavelon na Biblioteca
    for j in range(n_samples):
        t_j = potential_centers[j, :]
        s_j = global_candidate_scale
        h_j = wavelon(*[X[:, k] for k in range(n_features)],
                       translations=t_j,
                       scales=s_j,
                       wavelet_fn=wavelet_fn)
        H[:, j] = h_j

    # Retorna a biblioteca H e os parâmetros usados para criá-la
    return H, potential_centers, global_candidate_scale


def heuristic_initialization(X, n_neurons):
    """
    Inicialização heurística baseada nos limites dos dados.

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de features (n_samples, n_features)
    n_neurons : int
        Número de neurônios
    """
    M = X.max(axis=0)  # (n_features,)
    N = X.min(axis=0)  # (n_features,)

    translations_list = []
    scales_list = []

    # Criar neurônios com diversidade
    for _ in range(n_neurons):
        # Centro aleatório dentro do range dos dados
        t = 0.5 * (M + N)
        s = 0.2 * (M - N)
        translations_list.append(t)
        scales_list.append(s)

    output_weights = np.random.uniform(-1, 1, size=n_neurons + 1)

    return translations_list, scales_list, output_weights


def RBS_initialization(X, y: NDArray, n_neurons: int, wavelet_fn=mexican_hat_wavelet) -> tuple[list[NDArray], list[NDArray], NDArray]:
    """
    Inicialização usando Residual-Based Selection (RBS) - Versão Simples.
    """
    translations_list = []
    scales_list = []
    selected_indexes = []

    # Preparar dados de entrada
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
    y = np.asarray(y).flatten()
    n_samples = y.shape[0]

    H, centers_candidates, global_candidate_scale = create_candidate_library(X, wavelet_fn=wavelet_fn)

    residuals = y.copy()

    for k in range(n_neurons):
        best_quadratic_error = float('inf')
        best_index = -1
        best_w = 0.0

        # Selecionar o melhor wavelon com base nos resíduos
        for j in range(n_samples):
            if j in selected_indexes:
                continue

            # Calcular o peso w_j para o wavelon j
            h_j = H[:, j].flatten()
            denominator = np.dot(h_j, h_j)

            # Evitar divisão por zero
            if denominator < 1e-10:
                continue

            # Calcular o peso w_j
            w_j = np.dot(h_j, residuals) / denominator
            y_pred_j = w_j * h_j
            quadratic_error = np.sum((residuals - y_pred_j) ** 2)

            # Atualizar o melhor wavelon se o erro quadrático for menor
            if quadratic_error < best_quadratic_error:
                best_quadratic_error = quadratic_error
                best_index = j
                best_w = w_j

        # Verifica se um wavelon foi selecionado
        if best_index == -1:
            print(f"Aviso: Não foi possível selecionar mais wavelons (k={k})")
            break

        # Adiciona o melhor wavelon
        selected_indexes.append(best_index)

        # Pega a saída do melhor wavelon
        h_best = H[:, best_index].flatten()

        # Atualiza os resíduos
        residuals = residuals - (best_w * h_best)

    # Parâmetros finais (t e s)
    for i in selected_indexes:
        translations_list.append(centers_candidates[i])
        scales_list.append(global_candidate_scale)

    # Calcular os pesos de saída usando mínimos quadrados
    H_final = H[:, selected_indexes]
    H_bias = np.column_stack([H_final, np.ones(n_samples)])

    # Resolver o sistema linear H_bias * output_weights = y
    try:
        output_weights = np.linalg.lstsq(H_bias, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        print("Aviso: Usando pesos aleatórios devido a erro no sistema linear")
        output_weights = np.random.uniform(-1, 1, size=len(selected_indexes) + 1)

    return translations_list, scales_list, output_weights


def SSO_initialization(X, y: NDArray, n_neurons: int, wavelet_fn=mexican_hat_wavelet) -> tuple[list[NDArray], list[NDArray], NDArray]:
    """
    Inicialização usando Seleção por Ortogonalização (SSO / OLS).
    Usa o processo de Gram-Schmidt modificado para selecionar wavelons.

    Referência: Chen, Cowan, Grant (1991) - Orthogonal Least Squares Learning Algorithm
    """
    translations_list = []
    scales_list = []
    selected_indexes = []

    # Preparar dados de entrada
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
    y = np.asarray(y).flatten()
    n_samples = y.shape[0]

    # Criar a biblioteca de candidatos
    H, centers_candidates, global_candidate_scale = create_candidate_library(X, wavelet_fn=wavelet_fn)

    # Inicializar matriz de wavelons ortogonalizados
    W_ortho = H.copy()  # (n_samples, n_candidates)

    # Armazenar coeficientes de ortogonalização
    alphas = []  # Coeficientes g_k para reconstruir pesos finais

    # Loop de Seleção de Wavelons
    for k in range(n_neurons):
        best_err_reduction = -float('inf')
        best_index = -1
        best_g_k = 0.0  # Armazenar coeficiente

        # Encontra o melhor wavelon baseado na redução de erro
        for j in range(n_samples):  # n_samples = n_candidates
            if j in selected_indexes:
                continue

            # Pega o vetor ortogonalizado atual
            w_j = W_ortho[:, j].flatten()
            denominator = np.dot(w_j, w_j)

            if denominator < 1e-10:
                continue

            # Calcula o coeficiente e a redução de erro
            g_j = np.dot(w_j, y) / denominator
            err_reduction = (g_j ** 2) * denominator  # = (w_j^T y)^2 / (w_j^T w_j)

            if err_reduction > best_err_reduction:
                best_err_reduction = err_reduction
                best_index = j
                best_g_k = g_j

        if best_index == -1:
            print(f"Aviso: Não foi possível selecionar mais wavelons (k={k})")
            break

        # Adiciona o melhor wavelon e seu coeficiente
        selected_indexes.append(best_index)
        alphas.append(best_g_k)

        # Pega o vetor ortogonalizado do melhor wavelon
        w_k = W_ortho[:, best_index].flatten()
        w_k_norm_sq = np.dot(w_k, w_k)

        # Atualiza os vetores ortogonalizados restantes (Gram-Schmidt)
        for j in range(n_samples):
            if j in selected_indexes:
                continue

            w_j = W_ortho[:, j].flatten()

            # Projeção de w_j em w_k
            projection_coeff = np.dot(w_j, w_k) / w_k_norm_sq

            # Subtrai a componente paralela (ortogonalização)
            W_ortho[:, j] = w_j - projection_coeff * w_k


    # Calcular os pesos de saída usando mínimos quadrados
    H_final = H[:, selected_indexes]
    H_bias = np.column_stack([H_final, np.ones(n_samples)])

    try:
        output_weights = np.linalg.lstsq(H_bias, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        print("Aviso: Usando pesos aleatórios devido a erro no sistema linear")
        output_weights = np.random.uniform(-1, 1, size=len(selected_indexes) + 1)

    # Parâmetros finais (t e s)
    for i in selected_indexes:
        translations_list.append(centers_candidates[i])
        scales_list.append(global_candidate_scale)

    return translations_list, scales_list, output_weights



def weights_initialization_wavelet(*features, y, n_neurons, method="heuristic", wavelet_fn=None):
    """
    Inicializa pesos da rede Wavelon.
    """
    X = np.column_stack([np.atleast_1d(f) for f in features])
    n_samples, n_features = X.shape

    match method:
        case "heuristic":
            print("Usando inicialização heurística.")
            translations_list, scales_list, output_weights = heuristic_initialization(X, n_neurons)
        case "rbs":
            print("Usando inicialização RBS.")
            translations_list, scales_list, output_weights = RBS_initialization(
                X, y=y, n_neurons=n_neurons, wavelet_fn=wavelet_fn
            )
        case "sso":
            print("Usando inicialização SSO.")
            translations_list, scales_list, output_weights = SSO_initialization(
                X, y=y, n_neurons=n_neurons, wavelet_fn=wavelet_fn
            )
        case "be":
            print("Usando inicialização BE.")

        case _:
            print(f"Método '{method}' desconhecido. Usando inicialização aleatória.")
            translations_list = [np.random.uniform(X.min(axis=0), X.max(axis=0)) for _ in range(n_neurons)]
            scales_list = [np.random.uniform(0.1, 1.0, n_features) for _ in range(n_neurons)]
            output_weights = np.random.uniform(-1, 1, size=n_neurons + 1)


    initial_weights = []
    for t, s in zip(translations_list, scales_list):
        initial_weights.extend(t)  # n_features valores
        initial_weights.extend(s)  # n_features valores
    initial_weights.extend(output_weights)  # n_neurons + 1 valores

    return np.array(initial_weights)
