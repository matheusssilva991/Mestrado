import sys
import os
from numpy.typing import NDArray
import numpy as np

# Adicionar o diretório raiz do projeto ao Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.error import calculate_matrix_error as __calculate_error
from utils.matrix import get_augmented_matrix


def calculate_initial_solution(augmented_matrix: NDArray) -> NDArray:
    """
    Calcula a solução inicial para o método iterativo.
    :param augmented_matrix: Matriz aumentada do sistema
    :return: Solução inicial
    """
    n: int = augmented_matrix.shape[0]
    initial_guess: NDArray = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if augmented_matrix[i, i] == 0:
            raise ValueError("Elemento nulo encontrado na diagonal.")

        initial_guess[i] = augmented_matrix[i, -1] / augmented_matrix[i, i]

    return initial_guess


def jacobi(
    augmented_matrix: NDArray, tol: np.float64 | float, max_iter: int
) -> NDArray:
    """
    Método de Jacobi para resolver sistemas lineares.
    """

    def calculate_current_solution_jacobi(
        augmented_matrix: NDArray, current_solution: NDArray
    ) -> NDArray:
        """
        Método de Jacobi para calcular a próxima solução.
        :param augmented_matrix: Matriz aumentada do sistema
        :param current_solution: Solução atual
        :return: Próxima solução
        """
        n: int = augmented_matrix.shape[0]
        new_solution: NDArray = np.zeros(n, dtype=np.float64)

        for i in range(n):
            sum: np.float64 = np.float64(0.0)

            for j in range(i):
                sum += augmented_matrix[i, j] * new_solution[j]

            for j in range(i + 1, n):
                sum += augmented_matrix[i, j] * current_solution[j]

            new_solution[i] = (augmented_matrix[i, -1] - sum) / augmented_matrix[i, i]

        return new_solution

    return iterative_method(
        augmented_matrix,
        tol,
        max_iter,
        calculate_current_solution_jacobi,
        "O método de Jacobi",
    )


def gauss_seidel(
    augmented_matrix: NDArray, tol: np.float64 | float, max_iter: int
) -> NDArray:
    """
    Método de Gauss-Seidel para resolver sistemas lineares.
    """

    def calculate_solution_gs(matrix, prev_solution):
        # Implementação específica para Gauss-Seidel
        # (adaptação da sua implementação atual)
        n = matrix.shape[0]
        solution = np.copy(prev_solution)

        for i in range(n):
            sum_value = 0.0
            for j in range(n):
                if j != i:
                    sum_value += matrix[i, j] * solution[j]

            solution[i] = (matrix[i, -1] - sum_value) / matrix[i, i]

        return solution

    return iterative_method(
        augmented_matrix,
        tol,
        max_iter,
        calculate_solution_gs,
        "O método de Gauss-Seidel",
    )


def relaxing(
    augmented_matrix: NDArray,
    tol: np.float64 | float,
    max_iter: int,
    relax_factor: np.float64 | float,
) -> NDArray:
    """
    Método de Relaxação para resolver sistemas lineares.
    """

    def calculate_solution_relax(matrix, prev_solution):
        # Implementação específica para o método de relaxação
        # (adaptação da sua implementação atual)
        n = matrix.shape[0]
        solution = np.copy(prev_solution)

        for i in range(n):
            sum_value = 0.0
            for j in range(n):
                if j != i:
                    sum_value += matrix[i, j] * solution[j]

            gs_value = (matrix[i, -1] - sum_value) / matrix[i, i]
            solution[i] = prev_solution[i] + relax_factor * (
                gs_value - prev_solution[i]
            )

        return solution

    return iterative_method(
        augmented_matrix,
        tol,
        max_iter,
        calculate_solution_relax,
        f"O método de Relaxação (ω={relax_factor})",
    )


def iterative_method(
    augmented_matrix: NDArray,
    tol: np.float64 | float,
    max_iter: int,
    calculate_solution_func,
    method_name: str = "O método iterativo",
) -> NDArray:
    """
    Método iterativo genérico para resolver sistemas lineares.

    :param augmented_matrix: Matriz aumentada do sistema
    :param tol: Tolerância para convergência
    :param max_iter: Número máximo de iterações
    :param calculate_solution_func: Função que calcula a próxima solução
    :param method_name: Nome do método para mensagens de erro
    :return: Solução do sistema
    """
    n: int = augmented_matrix.shape[0]
    iter_count: int = 0
    previous_solution: NDArray = calculate_initial_solution(augmented_matrix)
    current_solution: NDArray = np.zeros(n, dtype=np.float64)

    while iter_count < max_iter:
        current_solution = calculate_solution_func(augmented_matrix, previous_solution)

        if np.any(np.isnan(current_solution)):
            raise ValueError("Solução inválida encontrada.")

        error = __calculate_error(current_solution, previous_solution)

        if error < tol:
            return current_solution

        iter_count += 1
        previous_solution = np.copy(current_solution)

    raise RuntimeError(f"{method_name} não convergiu após {max_iter} iterações.")


if __name__ == "__main__":
    # Exemplo de uso
    A: NDArray = np.matrix(
        [[6.0, -1.0, 3.0], [1.0, 3.0, 1.0], [3.0, -1.0, 5.0]], dtype=np.float64
    )
    b: NDArray = np.array([13.0, 10.0, 16.0], dtype=np.float64)
    tol = 1e-3
    max_iter = 1000
    relax_factor = 1.12  # Fator de relaxação para o método de relaxação

    augmented_matrix: NDArray = get_augmented_matrix(A, b)

    try:
        jacobi_solution = jacobi(augmented_matrix.copy(), tol, max_iter)
    except Exception as e:
        print(f"Erro no método de Jacobi: {e}")
        jacobi_solution = None

    try:
        gauss_seidel_solution = gauss_seidel(augmented_matrix.copy(), tol, max_iter)
    except Exception as e:
        print(f"Erro no método de Gauss-Seidel: {e}")
        gauss_seidel_solution = None

    try:
        relaxing_solution = relaxing(
            augmented_matrix.copy(), tol, max_iter, relax_factor
        )
    except Exception as e:
        print(f"Erro no método de Relaxação: {e}")
        relaxing_solution = None

    print("Solução dos métodos:")
    print(f"{'Jacobi:':<35} {jacobi_solution}")
    print(f"{'Gauss-Seidel:':<35} {gauss_seidel_solution}")
    print(f"{'Relaxação:':<35} {relaxing_solution}")
    print(f"{'Solução correta (numpy):':<35} {np.linalg.solve(A, b)}")
