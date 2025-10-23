volumes = [2, 1, 3, 2]
values = [12, 10, 20, 15]
capacity = 5

# volumes = [5, 4, 3, 2, 1]
# values = [10, 40, 30, 20, 10]
# capacity = 6

# volumes = [10, 20, 30]
# values = [5, 10, 15]
# capacity = 5


class KnapsackSolver:
    def __init__(self, volumes: list[int], values: list[int], capacity: int):
        if len(volumes) != len(values):
            raise ValueError("Длины volumes и values должны совпадать.")
        if any(v < 0 for v in volumes) or any(c < 0 for c in values):
            raise ValueError("Объёмы и ценности должны быть неотрицательными.")
        if capacity < 0:
            raise ValueError("Вместимость рюкзака должна быть неотрицательной.")

        self.volumes = volumes
        self.values = values
        self.capacity = capacity
        self.n = len(volumes)

        self.OPT = [[0] * (capacity + 1) for _ in range(self.n + 1)]
        self.x = [[0] * (capacity + 1) for _ in range(self.n + 1)]

        self.selected_items = []
        self.max_value = 0

    def _forward_pass(self):
        for b in range(self.capacity + 1):
            if self.volumes[0] <= b:
                self.OPT[1][b] = self.values[0]
                self.x[1][b] = 1
            else:
                self.OPT[1][b] = 0
                self.x[1][b] = 0

        for k in range(2, self.n + 1):
            v_k = self.volumes[k - 1]
            c_k = self.values[k - 1]
            for b in range(self.capacity + 1):
                if v_k <= b:
                    opt_without = self.OPT[k - 1][b]
                    opt_with = self.OPT[k - 1][b - v_k] + c_k
                    if opt_with > opt_without:
                        self.OPT[k][b] = opt_with
                        self.x[k][b] = 1
                    else:
                        self.OPT[k][b] = opt_without
                        self.x[k][b] = 0
                else:
                    self.OPT[k][b] = self.OPT[k - 1][b]
                    self.x[k][b] = 0

    def _backward_pass(self):
        b_remaining = self.capacity
        self.selected_items = []
        for k in range(self.n, 0, -1):
            if self.x[k][b_remaining] == 1:
                self.selected_items.append(k - 1)
                b_remaining -= self.volumes[k - 1]
        self.selected_items.reverse()
        self.max_value = self.OPT[self.n][self.capacity]

    def solve(self):
        if self.n == 0:
            self.max_value = 0
            self.selected_items = []
            return

        self._forward_pass()
        self._backward_pass()

    def print_tables(self):
        print("\n=== Таблица OPT(k, b) ===")
        print("k \\ b", end="")
        for b in range(self.capacity + 1):
            print(f"{b:>4}", end="")
        print()
        for k in range(self.n + 1):
            print(f"{k:>3}  ", end="")
            for b in range(self.capacity + 1):
                print(f"{self.OPT[k][b]:>4}", end="")
            print()

        print("\n=== Таблица x(k, b) (1 = предмет k взят) ===")
        print("k \\ b", end="")
        for b in range(self.capacity + 1):
            print(f"{b:>4}", end="")
        print()
        for k in range(self.n + 1):
            print(f"{k:>3}  ", end="")
            for b in range(self.capacity + 1):
                print(f"{self.x[k][b]:>4}", end="")
            print()

    def print_solution(self):
        print(f"Количество предметов: {self.n}")
        print(f"Вместимость рюкзака: {self.capacity}")
        print(f"Предметы (индекс: объём, ценность):")
        for i in range(self.n):
            print(f"  {i + 1}: v={self.volumes[i]}, c={self.values[i]}")

        self.print_tables()

        print(f"\nМаксимальная суммарная ценность: {self.max_value}")
        if self.selected_items:
            print("Выбранные предметы (по индексам):", [i + 1 for i in self.selected_items])
            total_volume = sum(self.volumes[i] for i in self.selected_items)
            print(f"Суммарный объём выбранных предметов: {total_volume}")
            print("Характеристики выбранных предметов:")
            for i in self.selected_items:
                print(f"  Предмет {i + 1}: v={self.volumes[i]}, c={self.values[i]}")
        else:
            print("Ни один предмет не был выбран (рюкзак пуст).")


def main():
    solver = KnapsackSolver(volumes=volumes, values=values, capacity=capacity)
    solver.solve()
    solver.print_solution()


if __name__ == "__main__":
    main()
