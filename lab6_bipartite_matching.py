from collections import defaultdict, deque

# Источник: условие
V1 = ['a', 'b', 'c']
V2 = ['x', 'y', 'z']
EDGES = [
    ('a', 'x'),
    ('b', 'x'),
    ('b', 'y'),
    ('c', 'x'),
    ('c', 'y'),
    ('c', 'z'),
]

# Источник: https://ocw.tudelft.nl/wp-content/uploads/Algoritmiek_Bipartite_Matching.pdf
V1 = ['1',  '2',  '3',  '4',  '5']
V2 = ["1'", "2'", "3'", "4'", "5'"]
EDGES = [
    ('1', "1'"),
    ('1', "2'"),
    ('2', "2'"),
    ('3', "1'"),
    ('3', "3'"),
    ('3', "4'"),
    ('4', "2'"),
    ('4', "5'"),
    ('5', "2'"),
    ('5', "5'"),
]


class BipartiteMatching:
    def __init__(self, V1: list[str], V2: list[str], edges: list[tuple[str, str]], verbose: bool = True):
        self.V1 = set(V1)
        self.V2 = set(V2)
        self.edges = edges
        self.verbose = verbose
        if not self.V1 or not self.V2:
            raise ValueError("Обе доли должны быть непустыми.")
        if self.V1 & self.V2:
            raise ValueError("Доли V1 и V2 должны быть непересекающимися.")
        for u, v in edges:
            if u not in self.V1:
                raise ValueError(f"Вершина '{u}' из ребра ({u}, {v}) не принадлежит V1.")
            if v not in self.V2:
                raise ValueError(f"Вершина '{v}' из ребра ({u}, {v}) не принадлежит V2.")
        self.s = 's'
        self.t = 't'
        self._build_initial_graph()

    def _build_initial_graph(self):
        self.graph = defaultdict(list)
        for u, v in self.edges:
            self.graph[u].append(v)
        for u in self.V1:
            self.graph[self.s].append(u)
        for v in self.V2:
            self.graph[v].append(self.t)

    def _find_path_bfs(self) -> list[str] | None:
        queue = deque([(self.s, [self.s])])
        visited = {self.s}
        while queue:
            current, path = queue.popleft()
            for neighbor in self.graph[current]:
                if neighbor == self.t:
                    return path + [self.t]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _invert_internal_edges(self, path: list[str]):
        internal_path = path[1:-1]
        for i in range(len(internal_path) - 1):
            u = internal_path[i]
            v = internal_path[i + 1]
            if v in self.graph[u]:
                self.graph[u].remove(v)
            if u not in self.graph[v]:
                self.graph[v].append(u)
        first_in_V1 = path[1]
        last_in_V2 = path[-2]
        if first_in_V1 in self.graph[self.s]:
            self.graph[self.s].remove(first_in_V1)
        if self.t in self.graph[last_in_V2]:
            self.graph[last_in_V2].remove(self.t)

    def _extract_matching(self) -> list[tuple[str, str]]:
        matching = []
        for v in self.V2:
            for u in self.graph[v]:
                if u in self.V1:
                    matching.append((u, v))
        return matching

    def solve(self) -> list[tuple[str, str]]:
        iteration = 0
        if self.verbose:
            print("=== Начальное состояние графа G* ===")
            self._print_graph()
            print()
        while True:
            iteration += 1
            path = self._find_path_bfs()
            if path is None:
                if self.verbose:
                    print("Вершина t недостижима из s. Алгоритм завершает работу.\n")
                break
            if self.verbose:
                print(f"Итерация {iteration}: найден путь s → t:")
                print("  " + " → ".join(path))
            self._invert_internal_edges(path)
            if self.verbose:
                print("  После удаления крайних дуг и инвертирования внутренних:")
                self._print_graph()
                print()
        matching = self._extract_matching()
        return matching

    def _print_graph(self):
        all_nodes = {self.s, self.t} | self.V1 | self.V2
        for node in sorted(all_nodes,
                           key=lambda x: (0 if x == self.s else 1 if x == self.t else 2 if x in self.V1 else 3, x)):
            neighbors = sorted(self.graph[node])
            if neighbors:
                print(f"    {node} → {', '.join(neighbors)}")

    def print_input_and_result(self):
        print("=== Входные данные ===")
        print(f"Доля V1: {sorted(self.V1)}")
        print(f"Доля V2: {sorted(self.V2)}")
        print(f"Рёбра E: {sorted(self.edges)}")
        matching = self.solve()
        print("=== Результат ===")
        print(f"Максимальное паросочетание содержит {len(matching)} рёбер:")
        if matching:
            for u, v in sorted(matching):
                print(f"  {{{u}, {v}}}")
        else:
            print("  Паросочетание пусто.")


def main():
    solver = BipartiteMatching(V1=V1, V2=V2, edges=EDGES, verbose=True)
    solver.print_input_and_result()


if __name__ == "__main__":
    main()
