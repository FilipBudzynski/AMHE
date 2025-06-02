import numpy as np
from des import DES


def sphere_function(x):
    return np.sum(x**2)


if __name__ == "__main__":
    dim = 10
    bounds = [(-5.0, 5.0)] * dim
    max_evals = 20000

    des = DES(
        objective_func=sphere_function,
        dim=dim,
        bounds=bounds,
        max_evals=max_evals,
    )
    best_solution, best_value = des.run()

    print(f"Objective value DES: {best_value}")
    print("Best solution found:", best_solution)
