import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return (0.5 * x**2) + y**2


def run_vanilla_gd(func, init_value, learning_rate=0.01, n_max_iter=100):
    # Initialize x with a starting value
    x = torch.tensor(init_value, requires_grad=True)
    x_log = np.full((n_max_iter, len(init_value)), np.nan)
    x_grad = np.full((n_max_iter, len(init_value)), np.nan)

    # Run gradient descent for a certain number of iterations
    for i in range(n_max_iter):
        # Calculate the value of f(x)
        y = func(*x)

        # Compute the gradients of y with respect to x
        y.backward()

        # Update x using the gradients
        x.data -= learning_rate * x.grad

        x_log[i, :] = x.tolist()
        x_grad[i, :] = x.grad
        # Reset the gradients for the next iteration
        x.grad.zero_()

    return x_log, x_grad


# Define the learning rate
learning_rate = 0.2
grid_locs = np.linspace(-100, 100, 200)
x_grid, y_grid = np.meshgrid(grid_locs, grid_locs)

init_value = np.asarray(
    [
        x_grid[np.random.choice(len(x_grid)), np.random.choice(len(x_grid))],
        y_grid[np.random.choice(len(y_grid)), np.random.choice(len(y_grid))],
    ]
)

x_log, x_grad = run_vanilla_gd(f, init_value=init_value, learning_rate=learning_rate)


sns.heatmap(f(x_grid, y_grid))
sns.lineplot(x=x_log, y=f(x_log), marker="o")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

sns.lineplot(x=np.arange(0, len(x_grad)), y=x_grad, marker="o")
plt.show()
