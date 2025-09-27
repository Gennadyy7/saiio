import numpy as np

from moiu.lab1 import calculate_inv


def main_phase(c: np.ndarray, A: np.ndarray, x: np.ndarray, B: np.ndarray):
    """
    Основная фаза симплекс-метода.
    B — массив базисных индексов (индексация с 1).
    Возвращает: (x_opt, B_opt)
    """
    A = np.array(A, dtype=float)
    c = np.array(c, dtype=float)
    x = np.array(x, dtype=float)
    B = np.array(B, dtype=int)

    first_iter = True
    while True:
        A_b = A[:, B - 1]
        if first_iter:
            inv_A_b = np.linalg.inv(A_b)
        else:
            # Обновляем обратную матрицу при замене одного столбца
            inv_A_b = calculate_inv(A_b, inv_A_b_prev, k + 1, A[:, j0])
        inv_A_b_prev = inv_A_b.copy()
        first_iter = False

        c_b = c[B - 1]
        u = c_b.dot(inv_A_b)
        delta = u.dot(A) - c

        if np.all(delta >= -1e-9):
            return x, B

        j0 = np.where(delta < -1e-9)[0][0]
        z = inv_A_b.dot(A[:, j0])

        teta = np.array([x[B[i] - 1] / z[i] if z[i] > 1e-12 else np.inf for i in range(len(B))])
        if np.all(teta == np.inf):
            raise Exception("Целевая функция не ограничена сверху на множестве допустимых планов")

        teta_0 = np.min(teta)
        k = np.argmin(teta)

        j_k = B[k]
        B[k] = j0 + 1
        x[j0] = teta_0
        for i in range(len(B)):
            if i != k:
                x[B[i] - 1] -= teta_0 * z[i]
        x[j_k - 1] = 0
