import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Define the function f(x) = x^2 + 2x + 1
def f(x):
    return x**2 + 2 * x + 1


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

        # Compute the gradients of y with respect to x
        y.backward()

        # Update x using the gradients
        x.data -= learning_rate * x.grad

        x_log[i] = x.item()
        x_grad[i] = x.grad

        # Reset the gradients for the next iteration
        x.grad.zero_()

    return x_log, x_grad


# Define the learning rate
learning_rate = 0.1

# Define the range of values of x
x_values = np.arange(-100, 100, 0.5)

# Sample random initial value
init_value = x_values[np.random.choice(len(x_values))]

# Run gradient descent
x_log, x_grad = run_vanilla_gd(f, init_value=init_value, learning_rate=learning_rate)


# Plot gradient descent output
sns.lineplot(x=x_values, y=f(x_values))
sns.lineplot(x=x_log, y=f(x_log), marker="o")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot gradient values
sns.lineplot(x=np.arange(len(x_grad)), y=x_grad, marker="o")
plt.xlabel("update")
plt.ylabel("grad(x)")
plt.show()
