import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from func.vanilla_gradient_descent import run_vanilla_gd


def f1(x):
    return x**2 + 2 * x + 1


def f2(x):
    return (x**4) - 2 * (x**3) + 2


def run_gradient_descent(x_values, func, n_max_iter=100, learning_rate=0.01):
    # Sample random initial value
    init_value = x_values[np.random.choice(len(x_values))]

    # Run gradient descent
    x_log, x_grad = run_vanilla_gd(
        func, init_value=init_value, learning_rate=learning_rate, n_max_iter=n_max_iter
    )

    # Plot gradient descent output
    sns.lineplot(x=x_values, y=func(x_values))
    sns.lineplot(x=x_log, y=func(x_log), marker="o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Plot gradient values
    sns.lineplot(x=np.arange(len(x_grad)), y=x_grad, marker="o")
    plt.xlabel("update")
    plt.ylabel("grad(x)")
    plt.show()


run_gradient_descent(np.arange(-10, 10, 0.05), f1, 500)
run_gradient_descent(np.arange(-0.5, 2, 0.01), f2, 100)
