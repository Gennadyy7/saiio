import numpy as np


def calculate_inv(A, invA, i, x):
    """
    Обновляет обратную матрицу при замене i-го столбца матрицы A на вектор x.
    Индексация i — с 1.
    """
    n = A.shape[0]
    l = invA.dot(x)
    if abs(l[i - 1]) < 1e-12:
        raise ValueError("Матрица с изменённым столбцом необратима")
    li = l[i - 1]
    l[i - 1] = -1
    l = -l / li
    Q = np.eye(n)
    Q[:, i - 1] = l
    invModA = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            invModA[row, col] = Q[row, row] * invA[row, col]
            if row != (i - 1):
                invModA[row, col] += Q[row, i - 1] * invA[i - 1, col]
    return invModA
