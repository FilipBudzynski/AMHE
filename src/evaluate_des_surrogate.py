import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
import csv
from functions.functions import sphere, cigar, discus, ackley
from functions.count_calls import count_calls

from algorithm.des import DES
from surrogate import (
    GaussianProcessSurrogate,
)
import warnings

warnings.filterwarnings("ignore")


def save_results_to_csv(func_name, results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{func_name.lower()}_results.csv")

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Variant", "Run", "BestValue", "RealEvaluations", "TimeSec"])
        for variant in ["classic", "surrogate"]:
            for i, res in enumerate(results[variant]):
                writer.writerow(
                    [
                        variant,
                        i + 1,
                        f"{res['best_val']:.6e}",
                        res["evals"],
                        f"{res['time']:.4f}",
                    ]
                )

    print(f"results saved to : {csv_path}")


benchmarks = {
    "Sphere": sphere,
    "Cigar": cigar,
    "Discus": discus,
    "Ackley": ackley,
}


def run_experiment(func_name, func, dim=10, max_evals=5000, runs=3, seed=42):
    results = {"classic": [], "surrogate": []}
    surrogate = None

    for run in range(runs):
        print(f"\n[RUN {run+1}/{runs}] {func_name}")

        wrapped_func = count_calls(func)

        bounds = [(-5, 5)] * dim

        # DES
        np.random.seed(seed + run)
        random.seed(seed + run)
        wrapped_func.call_count = 0
        des = DES(wrapped_func, dim, bounds, max_evals)
        t0 = time.time()
        best_x1, best_val1 = des.run()
        t1 = time.time()
        classic_result = {
            "best_val": best_val1,
            "evals": wrapped_func.call_count,
            "time": t1 - t0,
        }

        # DES + surrogate
        np.random.seed(seed + run)
        random.seed(seed + run)
        wrapped_func.call_count = 0
        surrogate = GaussianProcessSurrogate(std_treshold=0.05, min_data_to_train=50)
        des_sur = DES(wrapped_func, dim, bounds, max_evals, surrogate_model=surrogate)
        t2 = time.time()
        best_x2, best_val2 = des_sur.run()
        t3 = time.time()
        surrogate_result = {
            "best_val": best_val2,
            "evals": wrapped_func.call_count,
            "time": t3 - t2,
        }

        results["classic"].append(classic_result)
        results["surrogate"].append(surrogate_result)

    return results, surrogate


if __name__ == "__main__":
    DIM = 30
    MAX_EVALS = 40000
    RUNS = 5
    SEED = 123

    for name, func in benchmarks.items():
        print(f"\n\n=== {name.upper()} ===")
        results, surrogate = run_experiment(
            name, func, dim=DIM, max_evals=MAX_EVALS, runs=RUNS, seed=SEED
        )

        save_results_to_csv(
            f"{name}_dim-{DIM}_evals-{MAX_EVALS}_min-data-to-train-{50}", results
        )

        print(f"\n{name} – ŚREDNIE WYNIKI (z {RUNS} uruchomień):")
        for variant in ["classic", "surrogate"]:
            avg_val = np.mean([r["best_val"] for r in results[variant]])
            avg_evals = np.mean([r["evals"] for r in results[variant]])
            avg_time = np.mean([r["time"] for r in results[variant]])
            print(
                f"  {variant.upper():<10} | val: {avg_val:.2e} | evals: {avg_evals:.0f} | time: {avg_time:.2f}s"
            )
