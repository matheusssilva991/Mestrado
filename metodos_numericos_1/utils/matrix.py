import numpy as np
from numpy.typing import NDArray


def generate_hilbert_matrix(n: int) -> tuple[NDArray, NDArray]:
    """
    Generate an n x n Hilbert matrix.

    A Hilbert matrix is a square matrix with elements defined as:
    H(i, j) = 1 / (i + j - 1)

    Parameters:
    n (int): The size of the matrix.

    Returns:
    list: An n x n Hilbert matrix.
    """

    A: NDArray = np.zeros((n, n), dtype=np.float64)
    B: NDArray = np.zeros(n, dtype=np.float64)

    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / (i + j + 1)
            B[i] += A[i, j]

    return A, B


def get_augmented_matrix(
    A: NDArray, B: NDArray
) -> NDArray:
    """
    Generate an augmented matrix from matrix A and vector B.

    Parameters:
    A (NDArray): The coefficient matrix.
    B (NDArray): The constant terms vector.

    Returns:
    NDArray: The augmented matrix.
    """
    return np.hstack((A, B.reshape(-1, 1)))