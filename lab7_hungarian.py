from collections import deque
from fractions import Fraction

from lab6_bipartite_matching import BipartiteMatching

C = [
    [7, 2, 1, 9, 4],
    [9, 6, 9, 5, 5],
    [3, 8, 3, 1, 8],
    [7, 9, 4, 2, 2],
    [8, 4, 7, 4, 8],
]


class BipartiteMatchingConsumer(BipartiteMatching):
    def get_final_graph(self) -> dict[str, list[str]]:
        return {node: list(neighbors) for node, neighbors in self.graph.items()}

    def get_reachable_from_s(self) -> set[str]:
        visited = set()
        queue = deque([self.s])
        visited.add(self.s)
        while queue:
            current = queue.popleft()
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited


class HungarianAlgorithm:
    def __init__(self, C: list[list[int]], verbose: bool = True):
        if not C:
            raise ValueError("Матрица не должна быть пустой.")
        n = len(C)
        for i, row in enumerate(C):
            if len(row) != n:
                raise ValueError(f"Строка {i + 1} имеет длину {len(row)}, ожидалось {n} (квадратная матрица).")
        self.C: list[list[int]] = [row[:] for row in C]
        self.n: int = n
        self.alpha: list[Fraction] = [Fraction(0)] * n
        self.beta: list[Fraction] = [Fraction(0)] * n
        self.verbose: bool = verbose

    def _init_dual_plan(self) -> None:
        self.alpha = [Fraction(0)] * self.n
        for j in range(self.n):
            self.beta[j] = Fraction(min(self.C[i][j] for i in range(self.n)))
        if self.verbose:
            print("=== Инициализация двойственного плана (Шаг 1) ===")
            self._print_matrix_with_dual()

    def _print_matrix_with_dual(self) -> None:
        header = "     " + "".join(f"{str(b):>8}" for b in self.beta)
        print(header)
        for i in range(self.n):
            row_vals = "".join(f"{self.C[i][j]:>8}" for j in range(self.n))
            print(f"{str(self.alpha[i]):>5}{row_vals}")
        print()

    def _build_equality_graph(self) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        V1 = [f"u{i + 1}" for i in range(self.n)]
        V2 = [f"v{j + 1}" for j in range(self.n)]
        edges = []
        for i in range(self.n):
            for j in range(self.n):
                if self.alpha[i] + self.beta[j] == self.C[i][j]:
                    edges.append((V1[i], V2[j]))
        return V1, V2, edges

    def solve(self, verbose_bipartite_matching: bool = False) -> list[tuple[int, int]]:
        self._init_dual_plan()
        iteration = 0

        while True:
            iteration += 1
            V1, V2, edges = self._build_equality_graph()

            if self.verbose:
                print(f"=== Итерация {iteration} ===")
                print("Множество J== (рёбра равенственного графа G):")
                edge_pairs = sorted((int(u[1:]), int(v[1:])) for u, v in edges)
                print(f"  J== = {{{', '.join(map(str, edge_pairs))}}}")

            matcher = BipartiteMatchingConsumer(
                V1=V1, V2=V2, edges=edges, verbose=verbose_bipartite_matching
            )
            matching = matcher.solve()

            if self.verbose:
                matching_pairs = sorted((int(u[1:]), int(v[1:])) for u, v in matching)
                print(f"Результат лабораторной работы 6 (Шаг 4):")
                print(f"  Паросочетание M = {{{', '.join(map(str, matching_pairs))}}}")
                print(f"  |M| = {len(matching)}")

            if len(matching) == self.n:
                if self.verbose:
                    print("\n✅ |M| = n → найдено совершенное паросочетание. Алгоритм завершён (Шаг 5).")
                result = [(int(u[1:]), int(v[1:])) for u, v in matching]
                return sorted(result)

            reachable = matcher.get_reachable_from_s()
            I_star: set[int] = set()
            J_star: set[int] = set()
            for node in reachable:
                if node.startswith('u') and node != 's':
                    try:
                        i = int(node[1:]) - 1
                        I_star.add(i)
                    except (ValueError, IndexError):
                        pass
                elif node.startswith('v'):
                    try:
                        j = int(node[1:]) - 1
                        J_star.add(j)
                    except (ValueError, IndexError):
                        pass

            if self.verbose:
                I_star_1b = sorted(i + 1 for i in I_star)
                J_star_1b = sorted(j + 1 for j in J_star)
                print(f"\nШаги 6–7: вершины, достижимые из s в G*")
                print(f"  I* = {{{', '.join(map(str, I_star_1b)) if I_star_1b else '∅'}}}")
                print(f"  J* = {{{', '.join(map(str, J_star_1b)) if J_star_1b else '∅'}}}")

            alpha_tilde = [Fraction(1) if i in I_star else Fraction(-1) for i in range(self.n)]
            beta_tilde = [Fraction(-1) if j in J_star else Fraction(1) for j in range(self.n)]

            theta = None
            candidates = []
            for i in I_star:
                for j in range(self.n):
                    if j not in J_star:
                        diff = Fraction(self.C[i][j]) - self.alpha[i] - self.beta[j]
                        val = diff / 2
                        candidates.append((i, j, diff, val))
                        if theta is None or val < theta:
                            theta = val

            if theta is None:
                raise RuntimeError("Не удалось вычислить θ — алгоритм не может продолжаться.")

            if self.verbose:
                print(f"\nШаг 9: вычисление θ = min[(c_ij - α_i - β_j)/2] для i ∈ I*, j ∉ J*")
                for i, j, diff, val in candidates:
                    print(f"    ({i + 1},{j + 1}): "
                          f"({self.C[i][j]} - {self.alpha[i]} - {self.beta[j]}) / 2 = {diff}/2 = {val}")
                print(f"  ⇒ θ = {theta}")

            for i in range(self.n):
                self.alpha[i] += theta * alpha_tilde[i]
            for j in range(self.n):
                self.beta[j] += theta * beta_tilde[j]

            if self.verbose:
                print(f"\nШаги 10–11: обновлённый двойственный план")
                self._print_matrix_with_dual()

    @staticmethod
    def _bfs_reachable_from_s(graph: dict[str, list[str]]) -> set[str]:
        visited = set()
        queue = deque(['s'])
        visited.add('s')
        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited


def main():
    solver = HungarianAlgorithm(C, verbose=True)
    result = solver.solve(verbose_bipartite_matching=False)

    print("=== Итоговое решение ===")
    print("Оптимальное назначение (i, j):")
    total = 0
    for i, j in sorted(result):
        val = C[i - 1][j - 1]
        total += val
        print(f"  строка {i} → столбец {j} (значение = {val})")
    print(f"\nМинимальная сумма: {total}")


if __name__ == "__main__":
    main()
