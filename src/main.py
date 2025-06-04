import numpy as np
import matplotlib.pyplot as plt
from numpy._core.fromnumeric import std
from des import DES
from surrogate import GussianProcessSurrogate
from time import time
import functools
import random


def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0
    return wrapper


@count_calls
def q1(x):
    return np.sum(x**2)


@count_calls
def q2(x):
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)


@count_calls
def q3(x):
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)


@count_calls
def q4(x):
    n = len(x)
    exponents = np.linspace(0, 6, n)  # 10^0 to 10^6
    weights = 10**exponents
    return np.sum(weights * x**2)


def q6(x):
    return x[0] ** 2 + 100 * np.sqrt(np.sum(x[1:] ** 2))


@count_calls
def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


if __name__ == "__main__":
    SEED = 42  
    np.random.seed(SEED)
    random.seed(SEED)

    dim = 3
    max_evals = 20000
    bounds = [(-5, 5)] * dim

    functions = [q1, q2, q3]
    names = ["Sphere (q1)", "Cigar (q2)", "Discus (q3)"]

    all_histories = {}

    for func, name in zip(functions, names):
        print(f"Running DES2 on {name}...")
        des = DES(
            func,
            dim,
            bounds,
            max_evals,
            surrogate_model=GussianProcessSurrogate(std_treshold=0.1),
        )
        start_time = time()
        best_sol, best_val = des.run()
        end_time = time()
        evals, fitness_hist = des.logger.dump()
        print(f"Best value found: {best_val:.6f} in {end_time - start_time}")
        print(f"Real evaluations: {func.call_count}")
        all_histories[name] = (evals, fitness_hist)

    plt.figure(figsize=(12, 8))
    for name, (evals, fitness) in all_histories.items():
        plt.plot(evals, fitness, label=name)

    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Fitness Value Found")
    plt.title(
        f"DES2 Optimization Performance on Benchmark Functions (n={dim}, {max_evals} evals)"
    )
    plt.yscale("log")
    plt.ylim(1e-10, 1e6)
    plt.legend()
    plt.grid(True)
    plt.show()
