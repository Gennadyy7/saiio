import numpy as np

from moiu.lab2 import main_phase
from moiu.lab3 import begin_phase


class GomoryMethod:
    """
    Метод отсекающих плоскостей Гомори для задачи ЦЛП:
        c^T x → max
        A x = b
        x ≥ 0, x ∈ ℤ^n
    """

    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray, one_iteration: bool = True):
        self.c_orig = np.array(c, dtype=float)
        self.A_orig = np.array(A, dtype=float)
        self.b_orig = np.array(b, dtype=float)
        self.n_orig = self.c_orig.shape[0]

        self.one_iteration = True

    @staticmethod
    def _fractional_part(x: float) -> float:
        return x - np.floor(x)

    @staticmethod
    def _is_integer_vector(x: np.ndarray, tol: float = 1e-8) -> bool:
        return np.all(np.abs(x - np.round(x)) < tol)

    def _solve_relaxation(self, c, A, b):
        begin_res = begin_phase(c, A, b)
        if isinstance(begin_res, str):
            return begin_res
        x0, B0 = begin_res
        try:
            return main_phase(c, A, x0, B0)
        except Exception as e:
            return str(e)

    def _generate_cut(self, x, B, A):
        n = len(x)
        B0 = np.array(B) - 1  # в индексацию с 0
        k = None
        for idx, j in enumerate(B0):
            if not self._is_integer_vector(np.array([x[j]])):
                k = idx
                jk = j
                break
        if k is None:
            raise RuntimeError("Нет дробных компонент в базисе")

        AB = A[:, B0]
        all_idx = set(range(n))
        N0 = sorted(all_idx - set(B0))
        AN = A[:, N0] if N0 else np.empty((A.shape[0], 0))
        AB_inv = np.linalg.inv(AB)
        Q = AB_inv @ AN
        q_row = Q[k, :]

        rhs = self._fractional_part(x[jk])
        coeffs = np.zeros(n + 1)
        for idx, j in enumerate(N0):
            coeffs[j] = self._fractional_part(q_row[idx])
        coeffs[n] = -1.0
        return coeffs, rhs

    def solve(self):
        c = self.c_orig.copy()
        A = self.A_orig.copy()
        b = self.b_orig.copy()

        while True:
            res = self._solve_relaxation(c, A, b)
            if isinstance(res, str):
                return res
            x_opt, B_opt = res

            if self._is_integer_vector(x_opt):
                x_clean = np.round(x_opt[:self.n_orig]).astype(int)
                return x_clean

            try:
                cut, rhs = self._generate_cut(x_opt, B_opt, A)
            except Exception as e:
                return f"Ошибка генерации отсечения: {e}"

            if self.one_iteration:
                return cut, rhs


if __name__ == "__main__":
    # Пример 1
    c = np.array([0, 1, 0, 0], dtype=float)
    A = np.array([
        [3, 2, 1, 0],
        [-3, 2, 0, 1]
    ], dtype=float)
    b = np.array([6, 0], dtype=float)

    # # Пример 2
    # c = np.array([1, 1, 0, 0], dtype=float)
    # A = np.array([
    #     [9, 11, 1, 0],
    #     [11, 9, 0, 1]
    # ], dtype=float)
    # b = np.array([22, 22], dtype=float)

    solver = GomoryMethod(c, A, b)
    result = solver.solve()

    print("Результат выполнения лабораторной работы №2:")
    print("=" * 50)

    if isinstance(result, np.ndarray):
        print("✅ Найден оптимальный целочисленный план:")
        print("x =", result)
    elif isinstance(result, tuple):
        coeffs, rhs = result
        print("⚠️  Требуется добавить отсекающее ограничение Гомори:")
        print("Коэффициенты при переменных (включая новую s):", coeffs)
        print("Правая часть:", rhs)
        print("Новая переменная s соответствует последнему коэффициенту (-1).")
    else:
        print("❌", result)
