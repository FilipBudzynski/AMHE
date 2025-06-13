import numpy as np
import matplotlib.pyplot as plt
from algorithm.des import DES
from surrogate import GaussianProcessSurrogate
from functions.functions import sphere
from functions.count_calls import count_calls
from time import time
import random
import warnings

warnings.filterwarnings("ignore")

def run_experiment(param_name, param_values, dim=3, max_evals=20000, num_runs=10):
    SEED = 42
    bounds = [(-5, 5)] * dim
    fixed_params = {
        "std_treshold": 0.005,
        "min_data_to_train": dim * 8,
        "train_window_size": dim * 50,
        "how_often_to_train": 10,
    }

    all_results = {}

    print("\nRunning reference DES (no surrogate)...")
    des_only_histories = []
    for run in range(num_runs):
        np.random.seed(SEED + run)
        random.seed(SEED + run)

        wrapped_func = count_calls(sphere)
        des_plain = DES(
            wrapped_func,
            dim,
            bounds,
            max_evals,
            surrogate_model=None,
        )
        des_plain.run()
        evals, fitness = des_plain.logger.dump()

        padded_fitness = np.interp(
            np.arange(max_evals),
            evals,
            fitness,
            left=fitness[0],
            right=fitness[-1],
        )
        des_only_histories.append(padded_fitness)

    avg_plain = np.mean(des_only_histories, axis=0)
    all_results["DES only"] = avg_plain

    for value in param_values:
        print(f"\nTesting {param_name} = {value}")
        histories = []

        for run in range(num_runs):
            np.random.seed(SEED + run)
            random.seed(SEED + run)

            surrogate_args = fixed_params.copy()
            surrogate_args[param_name] = value

            wrapped_func = count_calls(sphere)
            des = DES(
                wrapped_func,
                dim,
                bounds,
                max_evals,
                surrogate_model=GaussianProcessSurrogate(**surrogate_args),
            )
            des.run()
            evals, fitness = des.logger.dump()

            padded_fitness = np.interp(
                np.arange(max_evals),
                evals,
                fitness,
                left=fitness[0],
                right=fitness[-1],
            )
            histories.append(padded_fitness)

        avg_fitness = np.mean(histories, axis=0)
        all_results[f"{param_name}={value}"] = avg_fitness

    plt.figure(figsize=(10, 6))
    for label, curve in all_results.items():
        lw = 2 if "DES only" in label else 1.8
        plt.plot(curve, label=label, linewidth=lw)

    plt.xlabel("Function Evaluations")
    plt.ylabel("Average Best Fitness")
    plt.yscale("log")
    plt.ylim(1e-10, 1e6)
    plt.title(f"Sphere - Effect of {param_name} (avg of {num_runs} runs)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"sphere_paramtest_{param_name}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_experiment("std_treshold", [0.001, 0.005, 0.01, 0.1])
    dim=3
    run_experiment("min_data_to_train", [dim*8, dim*10, dim*20])  # for dim=3
    run_experiment("train_window_size", [dim*80, dim*100, dim*200])  # for dim=3
    run_experiment("how_often_to_train", [2, 5, 10, 20])
