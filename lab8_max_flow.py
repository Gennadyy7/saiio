from collections import deque
from typing import NamedTuple, Self

# Источник: условие
CAPACITIES: dict[tuple[str, str], int] = {
    ('s', 'a'): 3,
    ('s', 'b'): 2,
    ('a', 't'): 1,
    ('b', 't'): 2,
    ('a', 'b'): 1,
    ('a', 's'): 0,
    ('b', 's'): 0,
    ('t', 'a'): 0,
    ('t', 'b'): 0,
    ('b', 'a'): 0,
}

SOURCE: str = 's'
TARGET: str = 't'

# Источник: https://www.geeksforgeeks.org/dsa/ford-fulkerson-algorithm-for-maximum-flow-problem/
# CAPACITIES: dict[tuple[str, str], int] = {
#     ('0', '1'): 16,
#     ('1', '0'): 0,
#     ('0', '2'): 13,
#     ('2', '0'): 0,
#     ('1', '3'): 12,
#     ('3', '1'): 0,
#     ('1', '2'): 10,
#     ('2', '1'): 4,
#     ('2', '4'): 14,
#     ('4', '2'): 0,
#     ('3', '2'): 9,
#     ('2', '3'): 0,
#     ('3', '5'): 20,
#     ('5', '3'): 0,
#     ('4', '3'): 7,
#     ('3', '4'): 0,
#     ('4', '5'): 4,
#     ('5', '4'): 0,
# }
#
# SOURCE: str = '0'
# TARGET: str = '5'


class Arc(NamedTuple):
    u: str
    v: str

    def __repr__(self) -> str:
        return f"({self.u},{self.v})"

    def reverse(self) -> Self:
        return Arc(self.v, self.u)


class Network:
    def __init__(self, capacities: dict[tuple[str, str], int], source: str, target: str):
        self.capacities: dict[Arc, int] = {
            Arc(u, v): c for (u, v), c in capacities.items()
        }
        missing_reverse = []
        for arc in list(self.capacities.keys()):
            rev = arc.reverse()
            if rev not in self.capacities:
                missing_reverse.append(rev)
        if missing_reverse:
            raise ValueError(f"Отсутствуют обратные дуги: {missing_reverse}")

        self.vertices: set[str] = set()
        for arc in self.capacities:
            self.vertices.add(arc.u)
            self.vertices.add(arc.v)

        if source not in self.vertices:
            raise ValueError(f"Исток '{source}' отсутствует в сети")
        if target not in self.vertices:
            raise ValueError(f"Сток '{target}' отсутствует в сети")

        self.source = source
        self.target = target

    def get_arcs_from(self, u: str) -> list[Arc]:
        return [arc for arc in self.capacities if arc.u == u]

    def __repr__(self) -> str:
        lines = [f"Сеть с вершинами: {sorted(self.vertices)}", "Дуги и пропускные способности:"]
        for arc in sorted(self.capacities):
            lines.append(f"  {arc} → {self.capacities[arc]}")
        return "\n".join(lines)


class FordFulkersonSolver:
    def __init__(self, network: Network, verbose: bool = False):
        self.network = network
        self.verbose = verbose

        self.flow: dict[Arc, int] = {arc: 0 for arc in self.network.capacities}

        self.residual_capacities: dict[Arc, int] = {}

    def _compute_residual(self) -> None:
        self.residual_capacities = {}
        for arc in self.network.capacities:
            rev = arc.reverse()
            c = self.network.capacities[arc]
            f = self.flow[arc]
            f_rev = self.flow.get(rev, 0)
            self.residual_capacities[arc] = c - f + f_rev

    def _find_path_labeling(self) -> list[Arc] | None:
        parent: dict[str, Arc | None] = {v: None for v in self.network.vertices}
        queue = deque([self.network.source])
        visited = {self.network.source}

        while queue:
            u = queue.popleft()
            if u == self.network.target:
                break
            for arc in self.network.get_arcs_from(u):
                v = arc.v
                if v not in visited and self.residual_capacities[arc] > 0:
                    visited.add(v)
                    parent[v] = arc
                    queue.append(v)

        if parent[self.network.target] is None:
            return None

        path: list[Arc] = []
        curr = self.network.target
        while curr != self.network.source:
            arc = parent[curr]
            if arc is None:
                raise RuntimeError("Некорректное восстановление пути")
            path.append(arc)
            curr = arc.u
        path.reverse()
        return path

    def _update_flow_along_path(self, path: list[Arc], theta: int) -> None:
        f_P: dict[Arc, int] = {arc: 0 for arc in self.flow}
        for arc in path:
            f_P[arc] = theta

        new_flow = {}
        for arc in self.flow:
            rev = arc.reverse()
            term = (self.flow[arc] - self.flow.get(rev, 0) + f_P.get(arc, 0) - f_P.get(rev, 0))
            new_flow[arc] = max(0, term)
        self.flow = new_flow

    def _update_residual_along_path(self, path: list[Arc], theta: int) -> None:
        for arc in path:
            rev = arc.reverse()
            self.residual_capacities[arc] -= theta
            self.residual_capacities[rev] += theta

    def _get_reachable_from_s(self) -> set[str]:
        visited = set()
        queue = deque([self.network.source])
        visited.add(self.network.source)

        while queue:
            u = queue.popleft()
            for arc in self.network.get_arcs_from(u):
                v = arc.v
                if v not in visited and self.residual_capacities.get(arc, 0) > 0:
                    visited.add(v)
                    queue.append(v)
        return visited

    def _compute_flow_value(self) -> int:
        total = 0
        for arc in self.network.get_arcs_from(self.network.source):
            rev = arc.reverse()
            total += self.flow[arc] - self.flow.get(rev, 0)
        return total

    def solve(self) -> dict:
        iteration = 0

        if self.verbose:
            print("=== Начальная сеть ===")
            print(self.network)
            print("\nНачальный поток: все нули.\n")

        while True:
            iteration += 1
            self._compute_residual()

            if self.verbose:
                print(f"=== Итерация {iteration} ===")
                print("Остаточные пропускные способности c_f:")
                for arc in sorted(self.residual_capacities):
                    cf = self.residual_capacities[arc]
                    if cf != 0 or self.verbose:
                        print(f"  {arc}: {cf}")
                print()

            path = self._find_path_labeling()
            if path is None:
                if self.verbose:
                    print("(s,t)-путь не найден → поток максимален.")
                break

            theta = min(self.residual_capacities[arc] for arc in path)

            if self.verbose:
                print(f"Найден путь: {[str(a) for a in path]}")
                print(f"θ = min({[self.residual_capacities[a] for a in path]}) = {theta}")

            self._update_flow_along_path(path, theta)
            self._update_residual_along_path(path, theta)

            if self.verbose:
                print("Поток после обновления:")
                nonzero = {a: f for a, f in self.flow.items() if f != 0}
                for arc in sorted(nonzero):
                    print(f"  {arc} → {nonzero[arc]}")
                print()

        flow_value = self._compute_flow_value()
        reachable = self._get_reachable_from_s()
        min_cut_S = reachable
        min_cut_T = self.network.vertices - reachable

        result = {
            "flow": dict(self.flow),
            "flow_value": flow_value,
            "min_cut_S": min_cut_S,
            "min_cut_T": min_cut_T,
            "iterations": iteration,
        }

        return result


def main():
    net = Network(CAPACITIES, SOURCE, TARGET)
    solver = FordFulkersonSolver(net, verbose=True)

    result = solver.solve()

    print("\nФинальный результат:")

    print("\n1. Максимальный поток (значения на дугах):")
    flow_items = sorted(result["flow"].items(), key=lambda x: (x[0].u, x[0].v))
    for arc, val in flow_items:
        cap = net.capacities[arc]
        print(f"   {arc}: {val} / {cap}")

    print(f"\n2. Мощность максимального потока |f| = {result['flow_value']}")

    print(f"\n3. Минимальный разрез (S, T):")
    S = sorted(result["min_cut_S"])
    T = sorted(result["min_cut_T"])
    print(f"   S = {{ {', '.join(S)} }}")
    print(f"   T = {{ {', '.join(T)} }}")

    cut_capacity = 0
    for u in result["min_cut_S"]:
        for v in result["min_cut_T"]:
            arc = Arc(u, v)
            if arc in net.capacities:
                cut_capacity += net.capacities[arc]
    print(f"\n4. Пропускная способность разреза = {cut_capacity}")
    print(f"\nАлгоритм завершил работу за {result['iterations']} итераций.")


if __name__ == "__main__":
    main()
