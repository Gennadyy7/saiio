import numpy as np

from moiu.lab2 import main_phase


def begin_phase(c: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
    Начальная фаза: находит допустимый базисный план.
    Возвращает (x, B) или строку с ошибкой.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    m, n = A.shape

    # Шаг 1: делаем b неотрицательным
    mask = b < 0
    b[mask] *= -1
    A[mask] *= -1

    # Шаг 2: вспомогательная задача
    c_aux = np.hstack([np.zeros(n), -np.ones(m)])
    A_aux = np.hstack([A, np.eye(m)])

    # Шаг 3: начальный план
    x_aux = np.hstack([np.zeros(n), b])
    B = np.arange(n + 1, n + m + 1)

    # Шаг 4: решаем вспомогательную задачу
    try:
        x_aux, B = main_phase(c_aux, A_aux, x_aux, B)
    except Exception as e:
        return str(e)

    # Шаг 5: проверка совместности
    if np.any(np.abs(x_aux[n:]) > 1e-8):
        return "Задача несовместна"

    # Удаляем искусственные переменные из базиса
    while np.any(B > n):
        # Находим первую искусственную переменную в базисе
        k = None
        for i, bi in enumerate(B):
            if bi > n:
                k = i
                break
        if k is None:
            break

        # Пытаемся заменить её на исходную переменную
        A_b = A_aux[:, B - 1]
        inv_A_b = np.linalg.inv(A_b)
        replaced = False
        for j in range(n):
            if (j + 1) not in B:
                l_vec = inv_A_b.dot(A_aux[:, j])
                if abs(l_vec[k]) > 1e-12:
                    B[k] = j + 1
                    replaced = True
                    break
        if not replaced:
            # Удаляем строку
            i_row = B[k] - n - 1
            A = np.delete(A, i_row, axis=0)
            b = np.delete(b, i_row)
            A_aux = np.delete(A_aux, i_row, axis=0)
            B = np.delete(B, k)
            m -= 1
            # Обновляем индексы
            B = np.array([bi if bi <= n else bi - 1 for bi in B])

    return x_aux[:n], B
