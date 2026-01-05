import sys
import os
from numpy.typing import NDArray
from typing import Tuple
import numpy as np

# Adicionar o diretório raiz do projeto ao Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.matrix import get_augmented_matrix


def partial_pivot_selector(augmented_matrix, idx_array, i):
    """
    Função de seleção de pivô para o método de eliminação de Gauss com pivoteamento parcial.
    :param augmented_matrix: Matriz aumentada do sistema
    :param idx_array: Vetor de índices das linhas
    :param i: Índice da coluna atual
    :param scale_array: Vetor de escalas (não utilizado neste caso)
    :return: Índice do pivô"""

    n = augmented_matrix.shape[0]
    p = i
    for k in range(i + 1, n):
        if abs(augmented_matrix[idx_array[p], i]) < abs(augmented_matrix[idx_array[k], i]):
            p = k
    return p


def scaled_partial_pivot_selector(augmented_matrix, idx_array, i, scale_array):
    """
    Função de seleção de pivô para o método de eliminação de Gauss com pivoteamento escalonado.
    :param augmented_matrix: Matriz aumentada do sistema
    :param idx_array: Vetor de índices das linhas
    :param i: Índice da coluna atual
    :param scale_array: Vetor de escalas
    :return: Índice do pivô
    """
    n = augmented_matrix.shape[0]
    p = i
    max_ratio = abs(augmented_matrix[idx_array[i], i]) / scale_array[idx_array[i]]
    for k in range(i + 1, n):
        ratio = abs(augmented_matrix[idx_array[k], i]) / scale_array[idx_array[k]]
        if ratio > max_ratio:
            max_ratio = ratio
            p = k
    return p


def elimination(augmented_matrix: NDArray) -> None:
    """
    Método de eliminação de Gauss.
        :param augmented_matrix: Matriz aumentada do sistema
        :return: None
    """
    n: int = augmented_matrix.shape[0]

    for i in range(n):
        if augmented_matrix[i, i] == 0:
            raise ZeroDivisionError("Método falhou! Matriz singular.")

        for j in range(i + 1, n):
            factor: np.float64 = augmented_matrix[j, i] / augmented_matrix[i, i]

            for k in range(i, n + 1):
                augmented_matrix[j, k] = (
                    augmented_matrix[j, k] - factor * augmented_matrix[i, k]
                )


def elimination_with_pivoting(
    augmented_matrix: NDArray,
    idx_array: list[int],
    pivot_selector,
) -> None:
    """
    Eliminação de Gauss genérica com pivoteamento customizável.
    :param augmented_matrix: Matriz aumentada do sistema
    :param idx_array: Vetor de índices das linhas
    :param pivot_selector: Função que recebe (augmented_matrix, idx_array, i, scale_array) e retorna o índice do pivô
    """
    n: int = augmented_matrix.shape[0]

    for i in range(n):
        p = pivot_selector(augmented_matrix, idx_array, i)
        if augmented_matrix[idx_array[p], i] == 0:
            raise ArithmeticError("Não existe solução única!")

        if idx_array[i] != idx_array[p]:
            idx_array[i], idx_array[p] = idx_array[p], idx_array[i]

        for j in range(i + 1, n):
            factor: np.float64 = (
                augmented_matrix[idx_array[j], i] / augmented_matrix[idx_array[i], i]
            )
            for k in range(i, n + 1):
                augmented_matrix[idx_array[j], k] -= factor * augmented_matrix[idx_array[i], k]


def elimination_partial_pivoting(augmented_matrix: NDArray, idx_array: list[int]) -> None:
    """
    Método de eliminação de Gauss com pivoteamento parcial.
        :param augmented_matrix: Matriz aumentada do sistema
        :param idx_array: Vetor de índices das linhas
        :return: None
    """
    elimination_with_pivoting(augmented_matrix, idx_array, partial_pivot_selector)


def elimination_scaled_pivoting(augmented_matrix: NDArray, idx_array: list[int], scale_array: NDArray) -> None:
    """
    Método de eliminação de Gauss com pivoteamento parcial e escalonamento.
        :param augmented_matrix: Matriz aumentada do sistema
        :param idx_array: Vetor de índices das linhas
        :param scale_array: Vetor de escalas
        :return: None
    """
    def selector(a, idx, i):
        return scaled_partial_pivot_selector(a, idx, i, scale_array)
    elimination_with_pivoting(augmented_matrix, idx_array, selector)


def elimination_complete_pivoting(augmented_matrix: NDArray, row_idx: list[int], col_idx: list[int]) -> None:
    """
    Método de eliminação de Gauss com pivoteamento completo.
        :param augmented_matrix: Matriz aumentada do sistema
        :param row_idx: Vetor de índices das linhas
        :param col_idx: Vetor de índices das colunas
        :return: None
    """
    n = augmented_matrix.shape[0]
    for i in range(n):
        max_val = 0
        p, q = i, i
        for r in range(i, n):
            for c in range(i, n):
                val = abs(augmented_matrix[row_idx[r], col_idx[c]])
                if val > max_val:
                    max_val = val
                    p, q = r, c

        if augmented_matrix[row_idx[p], col_idx[q]] == 0:
            raise ArithmeticError("Não existe solução única!")

        # Troca linhas
        if row_idx[i] != row_idx[p]:
            row_idx[i], row_idx[p] = row_idx[p], row_idx[i]

        # Troca colunas
        if col_idx[i] != col_idx[q]:
            col_idx[i], col_idx[q] = col_idx[q], col_idx[i]

        # Eliminação normal usando row_idx e col_idx
        for j in range(i + 1, n):
            factor = augmented_matrix[row_idx[j], col_idx[i]] / augmented_matrix[row_idx[i], col_idx[i]]
            for k in range(i, n + 1):
                augmented_matrix[row_idx[j], col_idx[k]] -= factor * augmented_matrix[row_idx[i], col_idx[k]]


def back_substitution(augmented_matrix: NDArray) -> NDArray:
    """
    Método de substituição para resolver um sistema de equações lineares
        :param augmented_matrix: Matriz aumentada do sistema
        :return: Vetor solução
    """
    n: int = augmented_matrix.shape[0]
    solution_array: NDArray = np.zeros(n, dtype=np.float64)

    solution_array[n - 1] = augmented_matrix[n - 1, n] / augmented_matrix[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        sum: np.float64 = np.float64(0.0)

        for j in range(i + 1, n):
            sum += augmented_matrix[i, j] * solution_array[j]

        solution_array[i] = (augmented_matrix[i, n] - sum) / augmented_matrix[i, i]
    return solution_array


def forward_substitution(augmented_matrix: NDArray) -> NDArray:
    """
    Método de substituição para resolver um sistema de equações lineares
        :param augmented_matrix: Matriz aumentada do sistema
        :return: Vetor solução
    """
    n: int = augmented_matrix.shape[0]
    solution_array: NDArray = np.zeros(n, dtype=np.float64)

    for i in range(n):
        sum: np.float64 = np.float64(0.0)

        for j in range(i):
            sum += augmented_matrix[i, j] * solution_array[j]

        solution_array[i] = (augmented_matrix[i, n] - sum) / augmented_matrix[i, i]

    return solution_array


def back_substitution_partial_pivoting(augmented_matrix: NDArray, idx_array: list[int]) -> NDArray:
    """
    Método de substituição para resolver um sistema de equações lineares
        :param augmented_matrix: Matriz aumentada do sistema
        :return: Vetor solução
    """
    n: int = augmented_matrix.shape[0]
    solution_array: NDArray = np.zeros(n, dtype=np.float64)

    solution_array[n - 1] = (
        augmented_matrix[idx_array[n - 1], n]
        / augmented_matrix[idx_array[n - 1], n - 1]
    )

    for i in range(n - 2, -1, -1):
        sum: np.float64 = np.float64(0.0)

        for j in range(i + 1, n):
            sum += augmented_matrix[idx_array[i], j] * solution_array[j]

        solution_array[i] = (
            augmented_matrix[idx_array[i], n] - sum
        ) / augmented_matrix[idx_array[i], i]

    return solution_array


def back_substitution_complete_pivoting(augmented_matrix: NDArray, row_idx: list[int], col_idx: list[int]) -> NDArray:
    """
    Substituição retroativa considerando pivoteamento completo (trocas de linhas e colunas).
    :param augmented_matrix: Matriz aumentada após eliminação
    :param row_idx: Índices das linhas (após trocas)
    :param col_idx: Índices das colunas (após trocas)
    :return: Vetor solução com variáveis na ordem correta
    """
    n = augmented_matrix.shape[0]
    x_temp = np.zeros(n)

    # Substituição retroativa considerando permutação de linhas e colunas
    for i in range(n - 1, -1, -1):
        sum_ = 0.0
        for j in range(i + 1, n):
            sum_ += augmented_matrix[row_idx[i], col_idx[j]] * x_temp[j]

        x_temp[i] = (augmented_matrix[row_idx[i], -1] - sum_) / augmented_matrix[row_idx[i], col_idx[i]]

    # Reorganiza as variáveis para a ordem original usando col_idx
    x = np.zeros(n)
    for i in range(n):
        x[col_idx[i]] = x_temp[i]

    return x


def naive_gauss(augmented_matrix: NDArray) -> NDArray:
    """
    Método de eliminação de Gauss sem pivoteamento.
        :param augmented_matrix: Matriz aumentada do sistema
        :return: Vetor solução
    """
    n: int = augmented_matrix.shape[0]

    elimination(augmented_matrix)

    if augmented_matrix[n - 1, n] == 0:
        raise ArithmeticError("Não existe solução única!")

    return back_substitution(augmented_matrix)


def gauss_partial_pivoting(augmented_matrix: NDArray) -> NDArray:
    """
    Método de eliminação de Gauss com pivoteamento parcial.
        :param augmented_matrix: Matriz aumentada do sistema
        :return: Vetor solução
    """
    n: int = augmented_matrix.shape[0]
    idx_array: list[int] = list(range(n))

    elimination_partial_pivoting(augmented_matrix, idx_array)

    if augmented_matrix[idx_array[n - 1], n - 1] == 0:
        raise ArithmeticError("Não existe solução única!")

    return back_substitution_partial_pivoting(augmented_matrix, idx_array)


def gauss_scaled_pivoting(augmented_matrix: NDArray) -> NDArray:
    """
    Método de eliminação de Gauss com pivoteamento parcial e escalonamento.
        :param augmented_matrix: Matriz aumentada do sistema
        :return: Vetor solução
    """
    n: int = augmented_matrix.shape[0]
    idx_array: list[int] = list(range(n))
    scale_array: NDArray = np.array([np.max(np.abs(augmented_matrix[i, :-1])) for i in range(n)])

    elimination_scaled_pivoting(augmented_matrix, idx_array, scale_array)

    if augmented_matrix[idx_array[n - 1], n - 1] == 0:
        raise ArithmeticError("Não existe solução única!")

    return back_substitution_partial_pivoting(augmented_matrix, idx_array)


def gauss_complete_pivoting(augmented_matrix: NDArray) -> NDArray:
    """
    Método de eliminação de Gauss com pivoteamento completo.
        :param augmented_matrix: Matriz aumentada do sistema
        :return: Vetor solução
    """
    n: int = augmented_matrix.shape[0]
    row_idx: list[int] = list(range(n))
    col_idx: list[int] = list(range(n+1))

    elimination_complete_pivoting(augmented_matrix, row_idx, col_idx)

    if augmented_matrix[row_idx[n - 1], col_idx[n - 1]] == 0:
        raise ArithmeticError("Não existe solução única!")
    return back_substitution_complete_pivoting(augmented_matrix, row_idx, col_idx)


def LU_factoring(a: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Fatoração LU de uma matriz quadrada.
        :param a: Matriz quadrada
        :return: Tupla (L, U) onde L é a matriz triangular inferior e U é a matriz triangular superior
    """

    n: int = a.shape[0]
    L: NDArray = np.zeros((n, n), dtype=np.float64)
    U: NDArray = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            sum: np.float64 = np.sum(L[i, :i] * U[:i, j])
            U[i, j] = a[i, j] - sum

        for j in range(i, n):
            if U[i, i] == 0:
                raise ArithmeticError(
                    "Fatoração LU não é possível!."
                )

            sum: np.float64 = np.sum(L[j, :i] * U[:i, i])
            L[j, i] = (a[j, i] - sum) / U[i, i]

    return (L, U)


def LU_solve(a: NDArray) -> NDArray:
    """
    Resolução de um sistema de equações lineares usando fatoração LU.
        :param a: Matriz aumentada do sistema
        :return: Vetor solução
    """
    L, U = LU_factoring(a[:, :-1])

    L_augmented: NDArray = np.hstack((L, a[:, -1].reshape(-1, 1)))
    y: NDArray = forward_substitution(L_augmented)

    U_augmented: NDArray = np.hstack((U, y.reshape(-1, 1)))
    x: NDArray = back_substitution(U_augmented)
    return x


if __name__ == "__main__":
    # Exemplo de uso
    A: NDArray = np.matrix([[0.003, 59.14], [5.291, -6.130]], dtype=np.float64)
    b: NDArray = np.array([59.17, 46.78], dtype=np.float64)

    augmented_matrix: NDArray = get_augmented_matrix(A, b)

    naive_gauss_solution = naive_gauss(augmented_matrix.copy())
    gauss_partial_pivoting_solution = gauss_partial_pivoting(augmented_matrix.copy())
    gauss_scaled_pivoting_solution = gauss_scaled_pivoting(augmented_matrix.copy())
    gauss_complete_pivoting_solution = gauss_complete_pivoting(augmented_matrix.copy())
    lu_solution = LU_solve(augmented_matrix.copy())

    print("Solução dos métodos:")
    print(f"{'Naive Gauss:':<35} {naive_gauss_solution}")
    print(f"{'Gauss com Pivoteamento Parcial:':<35} {gauss_partial_pivoting_solution}")
    print(f"{'Gauss com Pivoteamento Escalonado:':<35} {gauss_scaled_pivoting_solution}")
    print(f"{'Gauss com Pivoteamento Completo:':<35} {gauss_complete_pivoting_solution}")
    print(f"{'Fatoração LU:':<35} {lu_solution}")
    print(f"{'Solução correta:':<35} {np.linalg.solve(A, b)}")

