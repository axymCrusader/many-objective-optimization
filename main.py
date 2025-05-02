import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Callable, Optional


def f(x: list[float]) -> float:
    """Функция, которую нужно минимизировать."""
    return 2 * (x[0] - 3) ** 2 + (x[1] - 4) ** 2


def gradient_f(x: list[float]) -> list[float]:
    """Градиент функции f."""
    return [4 * (x[0] - 3), 2 * (x[1] - 4)]


def monte_carlo_method(
    func: Callable[[list[float]], float],
    bounds: list[tuple[float, float]],
    num_iterations: int = 1000,
) -> tuple[Optional[list[float]], float]:
    """Метод Монте-Карло для минимизации функции."""
    random.seed(42)

    dimensions = len(bounds)
    best_point = None
    min_value = float("inf")

    for _ in range(num_iterations):
        point = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)]
        value = func(point)

        if value < min_value:
            min_value = value
            best_point = point.copy()

    return best_point, min_value


def gradient_descent_method(
    func: Callable[[list[float]], float],
    gradient_func: Callable[[list[float]], list[float]],
    x0: list[float],
    B: float,
    epsilon: float,
    bounds: Optional[list[tuple[float, float]]] = None,
    max_iterations: int = 1000,
) -> tuple[list[float], float, int]:
    """Метод градиентного спуска с возможным ограничением границ."""
    x_current = x0.copy()
    iterations = 0

    while iterations < max_iterations:
        grad = gradient_func(x_current)

        x_next = [x_current[i] - B * grad[i] for i in range(len(x_current))]

        if bounds:
            x_next = [
                min(max(x_next[i], bounds[i][0]), bounds[i][1])
                for i in range(len(x_next))
            ]

        distance = (
            sum((x_next[i] - x_current[i]) ** 2 for i in range(len(x_current))) ** 0.5
        )
        if distance < epsilon:
            break

        x_current = x_next
        iterations += 1

    return x_current, func(x_current), iterations


def visualize_function(
    func: Callable[[list[float]], float],
    bounds: list[tuple[float, float]] = [(0, 5), (0, 5)],
    title: str = "График функции f(x, y) = 2*(x-3)² + (y-4)²",
    point: Optional[list[float]] = None,
    point_label: Optional[str] = None,
) -> None:
    """Визуализирует 3D график функции."""
    x_bounds, y_bounds = bounds

    x = np.linspace(x_bounds[0], x_bounds[1], 100)
    y = np.linspace(y_bounds[0], y_bounds[1], 100)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)

    if point is not None:
        x_point, y_point = point
        z_point = func([x_point, y_point])
        ax.scatter(
            x_point, y_point, z_point, color="red", s=100, label=point_label or "Точка"
        )
        ax.legend()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(x, y)")
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()


def main() -> None:
    epsilon = 0.1
    x0 = [1.0, 2.0]
    bounds = [(0.0, 5.0), (0.0, 5.0)]
    B = 0.1

    best_point, min_value = monte_carlo_method(f, bounds)
    print("\n=== Результаты метода Монте-Карло ===")
    print(f"Найденная оптимальная точка: {best_point}")
    print(f"Минимальное значение функции: {min_value}")

    gd_point, gd_value, iterations = gradient_descent_method(
        f, gradient_f, x0, B, epsilon, bounds
    )
    print("\n=== Результаты градиентного спуска ===")
    print(f"Найденная оптимальная точка: {gd_point}")
    print(f"Минимальное значение функции: {gd_value}")
    print(f"Количество итераций: {iterations}")

    visualize_function(f, bounds, point=gd_point, point_label="Градиентный спуск")


if __name__ == "__main__":
    main()
