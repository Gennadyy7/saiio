from collections import defaultdict, deque

# Источник: https://moccasin-main-whippet-16.mypinata.cloud/ipfs/bafkreigopuosjjb7khqphs3pjtbfi5j2yolp4hulg337oqbrtmd5nje2xe
VERTICES = ['1', '2', '3', '4', '5', '6']
ARCS = [
    ('1', '2', 5),
    ('1', '3', 3),
    ('2', '4', 2),
    ('2', '5', 6),
    ('3', '5', 1),
    ('4', '6', 1),
    ('5', '6', 4),
]
S = '1'
T = '6'

# Источник: https://moccasin-main-whippet-16.mypinata.cloud/ipfs/bafkreierqko6ryqmfoqabztsy4nfii4q3kosvos6e4g5crs6cv5l7mxfie
VERTICES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
ARCS = [
    ('A', 'B', 3),
    ('A', 'C', 6),
    ('B', 'C', 4),
    ('C', 'D', 8),
    ('B', 'D', 4),
    ('B', 'E', 11),
    ('C', 'G', 11),
    ('D', 'E', 2),
    ('D', 'G', 2),
    ('D', 'F', 5),
    ('E', 'H', 9),
    ('F', 'H', 1),
    ('G', 'H', 2),
]
S = 'A'
T = 'H'


class DAGLongestPath:
    def __init__(self, vertices: list[str], arcs: list[tuple[str, str, int]], s: str, t: str):
        self.vertices = set(vertices)
        self.s = s
        self.t = t

        if s not in self.vertices or t not in self.vertices:
            raise ValueError("Вершины s и/или t отсутствуют в списке вершин.")

        self.graph: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.in_degree: dict[str, int] = {v: 0 for v in self.vertices}
        self.rev_graph: dict[str, list[str]] = defaultdict(list)

        for u, v, length in arcs:
            if u not in self.vertices or v not in self.vertices:
                raise ValueError(f"Дуга ({u}, {v}) содержит неизвестную вершину.")
            if length < 0:
                raise ValueError(f"Длина дуги ({u}, {v}) должна быть неотрицательной.")
            self.graph[u].append((v, length))
            self.rev_graph[v].append(u)
            self.in_degree[v] += 1

        for v in self.vertices:
            if v not in self.in_degree:
                self.in_degree[v] = 0

    def topological_sort(self) -> list[str] | None:
        in_deg = self.in_degree.copy()
        queue = deque([v for v in self.vertices if in_deg[v] == 0])
        topo_order = []

        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v, _ in self.graph[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)

        if len(topo_order) != len(self.vertices):
            return None
        return topo_order

    def find_longest_path(self) -> tuple[bool, int | None, list[str] | None]:
        topo = self.topological_sort()
        if topo is None:
            print("Ошибка: граф содержит контур (нарушено условие задачи).")
            return False, None, None

        try:
            k = topo.index(self.s)
            l = topo.index(self.t)  # noqa: E741
        except ValueError:
            return False, None, None

        if k > l:
            print(f"В топологическом порядке s='{self.s}' идёт после t='{self.t}'. Путь невозможен.")
            return False, None, None

        relevant_vertices = topo[k:l+1]
        vertex_to_index = {v: i for i, v in enumerate(relevant_vertices)}

        OPT = {}
        predecessor = {}

        for v in relevant_vertices:
            OPT[v] = float('-inf')
            predecessor[v] = None

        OPT[self.s] = 0

        for v in relevant_vertices:
            if OPT[v] == float('-inf'):
                continue
            for w, length in self.graph[v]:
                if w not in vertex_to_index:
                    continue
                new_length = OPT[v] + length
                if new_length > OPT[w]:
                    OPT[w] = new_length
                    predecessor[w] = v

        if OPT[self.t] == float('-inf'):
            return False, None, None

        path = []
        cur = self.t
        while cur is not None:
            path.append(cur)
            cur = predecessor[cur]
        path.reverse()

        return True, OPT[self.t], path

    def solve_and_print(self):
        print(f"Вершины: {sorted(self.vertices)}")
        print(f"Дуги (начало → конец : длина):")
        for u in sorted(self.graph.keys()):
            for v, length in self.graph[u]:
                print(f"  {u} → {v} : {length}")
        print(f"Начальная вершина s = '{self.s}'")
        print(f"Конечная вершина t = '{self.t}'")
        print()

        reachable, length, path = self.find_longest_path()

        if not reachable:
            print("Результат: вершина t НЕДОСТИЖИМА из вершины s.")
        else:
            print("Результат: вершина t ДОСТИЖИМА из вершины s.")
            print(f"Максимальная длина (s, t)-пути: {length}")
            print(f"Наидлиннейший путь: {' → '.join(path)}")


def main():
    solver = DAGLongestPath(VERTICES, ARCS, S, T)
    solver.solve_and_print()


if __name__ == "__main__":
    main()
