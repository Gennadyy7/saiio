from __future__ import annotations
from dataclasses import dataclass
import math

C = [
    [7, 2, 1, 9, 4],
    [9, 6, 9, 5, 5],
    [3, 8, 3, 1, 8],
    [7, 9, 4, 2, 2],
    [8, 4, 7, 4, 8],
]


@dataclass
class CostMatrix:
    matrix: list[list[float]]

    def __post_init__(self) -> None:
        if not self.matrix or not self.matrix[0]:
            raise ValueError("Матрица не должна быть пустой.")
        n = len(self.matrix)
        if any(len(row) != n for row in self.matrix):
            raise ValueError("Матрица должна быть квадратной (n x n).")

    @property
    def size(self) -> int:
        return len(self.matrix)

    def __getitem__(self, idx: tuple[int, int]) -> float:
        i, j = idx
        return self.matrix[i][j]


class HungarianSolver:
    def __init__(self, cost: CostMatrix):
        self.cost = cost
        self.n = cost.size

    def solve(self) -> tuple[list[int], float]:
        n = self.n
        a = [[0.0] * (n + 1)]
        for i in range(n):
            a.append([0.0] + [float(self.cost.matrix[i][j]) for j in range(n)])

        p = [0] * (n + 1)
        way = [0] * (n + 1)

        u = [0.0] * (n + 1)
        v = [0.0] * (n + 1)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [math.inf] * (n + 1)
            used = [False] * (n + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = math.inf
                j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = a[i0][j] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(0, n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = [-1] * n
        for j in range(1, n + 1):
            if p[j] != 0:
                assignment[p[j] - 1] = j - 1

        total_cost = 0.0
        for i in range(n):
            j = assignment[i]
            if j >= 0:
                total_cost += self.cost.matrix[i][j]

        return assignment, total_cost


def print_assignment(assignment: list[int], cost: CostMatrix) -> None:
    n = cost.size
    print("Результат назначения:")
    total = 0.0
    for i in range(n):
        j = assignment[i]
        if j >= 0:
            c = cost.matrix[i][j]
            print(f"  Строка {i} → Столбец {j}  (c = {c})")
            total += c
        else:
            print(f"  Строка {i} → не назначена")
    print(f"Суммарная стоимость: {total}")


if __name__ == "__main__":
    cost = CostMatrix(C)
    solver = HungarianSolver(cost)
    assignment, total = solver.solve()
    print("=== Входная матрица C ===")
    for row in C:
        print(" ", row)
    print()
    print_assignment(assignment, cost)
    print(f"\n(Проверка) Общее значение, возвращённое алгоритмом: {total}")
