import numpy as np
import matplotlib.pyplot as plt
from des import DES


def q1(x):
    return np.sum(x**2)


def q2(x):
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)


def q3(x):
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)


def q4(x):
    n = len(x)
    exponents = np.linspace(0, 6, n)  # 10^0 to 10^6
    weights = 10**exponents
    return np.sum(weights * x**2)


def q6(x):
    return x[0] ** 2 + 100 * np.sqrt(np.sum(x[1:] ** 2))


if __name__ == "__main__":
    dim = 3
    max_evals = 40000
    bounds = [(-5, 5)] * dim

    functions = [q1, q2, q3]
    names = ["Sphere (q1)", "Cigar (q2)", "Discus (q3)"]

    all_histories = {}

    for func, name in zip(functions, names):
        print(f"Running DES2 on {name}...")
        des = DES(func, dim, bounds, max_evals)
        best_sol, best_val = des.run()
        evals, fitness_hist = des.logger.dump()
        print(f"Best value found: {best_val:.6f}")
        all_histories[name] = (evals, fitness_hist)

    plt.figure(figsize=(12, 8))
    for name, (evals, fitness) in all_histories.items():
        plt.plot(evals, fitness, label=name)

    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Fitness Value Found")
    plt.title("DES2 Optimization Performance on Benchmark Functions (n=3, 20000 evals)")
    plt.yscale("log")
    plt.ylim(1e-10, 1e6)
    plt.legend()
    plt.grid(True)
    plt.show()
