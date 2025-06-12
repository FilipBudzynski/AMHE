import numpy as np
import matplotlib.pyplot as plt
from numpy._core.fromnumeric import std
from algorithm.des import DES
from surrogate import GaussianProcessSurrogate
from time import time
import random
from functions.functions import sphere, cigar, discus, ackley
from functions.count_calls import count_calls


import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    dim = 10 
    max_evals = 40000
    bounds = [(-5, 5)] * dim

    benchmarks = {
        "Sphere": sphere,
        "Cigar": cigar,
        "Discus": discus,
        "Ackley": ackley,
    }

    all_histories = {}

    for name, func in benchmarks.items():
        wrapped_func = count_calls(func)
        print(f"Running DES on {name}...")
        wrapped_func.call_count = 0
        des = DES(
            wrapped_func,
            dim,
            bounds,
            max_evals,
            # surrogate_model=GaussianProcessSurrogate(
            #     std_treshold=0.5, min_data_to_train=200, train_window_size=500
            # ),
        )
        start_time = time()
        best_sol, best_val = des.run()
        end_time = time()
        evals, fitness_hist = des.logger.dump()
        print(f"Best value found: {best_val:.6f} in {end_time - start_time}")
        print(f"Real evaluations: {wrapped_func.call_count}")
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
    plt.savefig("surrogate-des-30dim", dpi=300, bbox_inches="tight")
    plt.show()
