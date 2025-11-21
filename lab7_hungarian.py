from collections import deque

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
    def __init__(self, C: list[list[int]]):
        if not C:
            raise ValueError("Матрица не должна быть пустой.")
        n = len(C)
        for i, row in enumerate(C):
            if len(row) != n:
                raise ValueError(f"Строка {i + 1} имеет длину {len(row)}, ожидалось {n} (квадратная матрица).")
        self.C: list[list[int]] = [row[:] for row in C]
        self.n: int = n
        self.alpha: list[float] = [0.0] * n
        self.beta: list[float] = [0.0] * n

    def _init_dual_plan(self) -> None:
        self.alpha = [0.0] * self.n
        for j in range(self.n):
            self.beta[j] = min(self.C[i][j] for i in range(self.n))

    def _build_equality_graph(self) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        V1 = [f"u{i + 1}" for i in range(self.n)]
        V2 = [f"v{j + 1}" for j in range(self.n)]
        edges = []
        for i in range(self.n):
            for j in range(self.n):
                if abs(self.alpha[i] + self.beta[j] - self.C[i][j]) < 1e-9:
                    edges.append((V1[i], V2[j]))
        return V1, V2, edges

    def _update_dual_plan(self, reachable: set[str]) -> bool:
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
        alpha_tilde = [1.0 if i in I_star else -1.0 for i in range(self.n)]
        beta_tilde = [-1.0 if j in J_star else 1.0 for j in range(self.n)]
        theta = float('inf')
        for i in I_star:
            for j in range(self.n):
                if j not in J_star:
                    diff = self.C[i][j] - self.alpha[i] - self.beta[j]
                    if diff < 0:
                        diff = 0.0
                    candidate = diff / 2.0
                    if candidate < theta:
                        theta = candidate
        if theta == float('inf'):
            return False
        for i in range(self.n):
            self.alpha[i] += theta * alpha_tilde[i]
        for j in range(self.n):
            self.beta[j] += theta * beta_tilde[j]
        return False

    def solve(self, verbose_bipartite_matching: bool = False) -> list[tuple[int, int]]:
        self._init_dual_plan()
        iteration = 0
        while True:
            iteration += 1
            V1, V2, edges = self._build_equality_graph()
            matcher = BipartiteMatchingConsumer(V1=V1, V2=V2, edges=edges, verbose=verbose_bipartite_matching)
            matching = matcher.solve()
            if len(matching) == self.n:
                result = []
                for u, v in matching:
                    i = int(u[1:])
                    j = int(v[1:])
                    result.append((i, j))
                return sorted(result)
            final_graph = matcher.get_final_graph()
            reachable = self._bfs_reachable_from_s(final_graph)
            self._update_dual_plan(reachable)

    @staticmethod
    def _bfs_reachable_from_s(graph: dict[str, list[str]]) -> set[str]:
        visited = set()
        queue = ['s']
        visited.add('s')
        while queue:
            current = queue.pop(0)
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def print_dual_plan(self) -> None:
        print("α =", [round(x, 3) for x in self.alpha])
        print("β =", [round(x, 3) for x in self.beta])


def main():
    solver = HungarianAlgorithm(C)
    result = solver.solve(verbose_bipartite_matching=False)
    print("Оптимальное назначение (i, j):")
    for i, j in result:
        print(f"  строка {i} → столбец {j} (значение = {C[i - 1][j - 1]})")
    total = sum(C[i - 1][j - 1] for i, j in result)
    print(f"\nМинимальная сумма: {total}")


if __name__ == "__main__":
    main()
