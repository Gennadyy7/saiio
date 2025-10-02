import numpy as np


class ResourceAllocationProblem:
    def __init__(self, profit_matrix: list[list[int]]):
        self.A = np.array(profit_matrix, dtype=int)
        self.P, cols = self.A.shape
        self.Q = cols - 1

        if self.Q < 0:
            raise ValueError("Матрица прибыли должна содержать хотя бы один столбец (для x=0).")

        self.B = np.zeros((self.P, self.Q + 1), dtype=int)
        self.C = np.zeros((self.P, self.Q + 1), dtype=int)

    def _forward_pass(self) -> None:
        for p in range(self.P):
            for q in range(self.Q + 1):
                if p == 0:
                    self.B[p, q] = self.A[p, q]
                    self.C[p, q] = q
                else:
                    best_value = -float('inf')
                    best_i = 0
                    for i in range(q + 1):
                        value = self.A[p, i] + self.B[p - 1, q - i]
                        if value > best_value:
                            best_value = value
                            best_i = i
                    self.B[p, q] = best_value
                    self.C[p, q] = best_i

    def _backward_pass(self) -> list[int]:
        allocn = [0] * self.P
        q = self.Q
        for p in range(self.P - 1, -1, -1):
            allocated = int(self.C[p, q])
            allocn[p] = allocated
            q -= allocated
        return allocn

    def solve(self) -> tuple[int, list[int]]:
        self._forward_pass()
        max_profit = int(self.B[self.P - 1, self.Q])
        allocn = self._backward_pass()
        return max_profit, allocn


if __name__ == "__main__":
    A = [
        [0, 1, 2, 3],
        [0, 0, 1, 2],
        [0, 2, 2, 3],
    ]

    problem = ResourceAllocationProblem(A)
    profit, allocation = problem.solve()

    print("Максимальная прибыль:", profit)
    print("Оптимальное распределение:", allocation)
