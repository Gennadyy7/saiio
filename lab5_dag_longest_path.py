from collections import defaultdict, deque

VERTICES_1 = ['A', 'B', 'C', 'D', 'E']
ARCS_1 = [
    ('A', 'B', 3),
    ('A', 'C', 2),
    ('B', 'D', 4),
    ('C', 'D', 1),
    ('D', 'E', 2)
]
S_1 = 'A'
T_1 = 'E'

VERTICES_2 = ['X', 'Y', 'Z']
ARCS_2 = [
    ('Y', 'X', 5),
    ('Z', 'Y', 2)
]
S_2 = 'X'
T_2 = 'Z'

VERTICES_3 = ['P', 'Q', 'R']
ARCS_3 = [
    ('Q', 'R', 10),
    ('P', 'Q', 1)
]
S_3 = 'R'
T_3 = 'P'

VERTICES = VERTICES_1
ARCS = ARCS_1
S = S_1
T = T_1


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
