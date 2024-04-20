import torch
import numpy as np


def run_vanilla_gd(func, init_value, learning_rate=0.01, n_max_iter=100):
    # Initialize x with a random starting value
    x = torch.tensor(init_value, requires_grad=True)

    # logging arrays
    x_log = np.full(n_max_iter, np.nan)
    x_grad = np.full(n_max_iter, np.nan)

    # Run gradient descent for a certain number of iterations
    for i in range(n_max_iter):
        # Calculate the value of f(x)
        y = func(x)

        # Compute the examples of y with respect to x
        y.backward()

        # Update x using the examples
        x.data -= learning_rate * x.grad

        x_log[i] = x.item()
        x_grad[i] = x.grad

        # Reset the examples for the next iteration
        x.grad.zero_()

    return x_log, x_grad


def run_vanilla_gd_2d(func, init_x, init_y, learning_rate=0.01, n_max_iter=100):
    # Initialize x with a starting value
    x = torch.tensor(init_x, requires_grad=True)
    y = torch.tensor(init_y, requires_grad=True)

    log = np.full((n_max_iter, 2), np.nan)
    grad = np.full((n_max_iter, 2), np.nan)

    # Run gradient descent for a certain number of iterations
    for i in range(n_max_iter):
        # Calculate the value of f(x)
        z = func(x, y)

        # Compute the examples of z with respect to x and y
        z.backward()

        # Update x and y using the examples
        x.data -= learning_rate * x.grad
        y.data -= learning_rate * y.grad

        log[i, :] = [x.item(), y.item()]
        grad[i, :] = [x.grad, y.grad]

        # Reset the examples for the next iteration
        x.grad.zero_(), y.grad.zero_()

    return log, grad
