import matplotlib.pyplot as plt
import numpy as np

from func.vanilla_gradient_descent import run_vanilla_gd_2d


def f(x, y):
    return (0.5 * (x**2)) + y**2


# Define the learning rate
learning_rate = 0.05
grid_locs = np.linspace(-100, 100, 200)
x_grid, y_grid = np.meshgrid(grid_locs, grid_locs)

init_x = x_grid[np.random.choice(len(x_grid)), np.random.choice(len(x_grid))]
init_y = y_grid[np.random.choice(len(y_grid)), np.random.choice(len(y_grid))]

log, grad = run_vanilla_gd_2d(
    f, init_x=init_x, init_y=init_y, learning_rate=learning_rate, n_max_iter=500
)


# Plot the surface.
plt.imshow(
    f(x_grid, y_grid),
    cmap="hot",
    extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
)
plt.plot(log[:, 0], log[:, 1], "g-", marker="o", linewidth=2)
plt.colorbar()
plt.show()


# Plot the examples
plt.plot(grad)
plt.legend(["grad(x)", "grad(y)"])
plt.show()
