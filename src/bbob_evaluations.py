import cocoex
import numpy as np
import random
import time
import os
import csv
from algorithm.des import DES
from surrogate import GaussianProcessSurrogate
import warnings

warnings.filterwarnings("ignore")

DIMENSIONS = [2, 5, 10, 20, 40]
FUN_IDS = range(1, 25)
INSTANCES = [1]
TARGET = 1e-6
RUNS = 3
MAX_EVALS = 100000
SEED = 123

RESULTS_DIR = "bbob_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_and_track(algo, f, dim, budget):
    t0 = time.time()
    xbest, fbest = algo.run()
    evals, fitness_hist = algo.logger.dump()
    t1 = time.time()
    evals = f.evaluations
    return {
        "evals": evals,
        "time": t1 - t0,
        "best_so_far": fitness_hist,  
        "best_val": fbest,
    }


def run_experiment():
    for dim in DIMENSIONS:
        for fun_id in FUN_IDS:
            for instance in INSTANCES:
                suite = cocoex.Suite(
                    "bbob",
                    f"dimensions:{dim}",
                    f"function_indices:{fun_id},instances:{instance}",
                )

                for f in suite:
                    func_name = f"f{fun_id}_d{dim}_i{instance}"
                    print(f"Running: {func_name}")

                    for variant in ["classic", "surrogate"]:
                        rows = []
                        for run in range(RUNS):
                            print(f"  Run {run+1}/{RUNS}")
                            suite = cocoex.Suite(
                                "bbob",
                                "",
                                f"function_indices:{fun_id},instances:{instance}",
                            )
                            f = suite.next_problem()

                            np.random.seed(SEED + run)
                            random.seed(SEED + run)

                            if variant == "classic":
                                bounds = list(zip(f.lower_bounds, f.upper_bounds))
                                des = DES(f, dim, bounds, MAX_EVALS)
                            else:
                                surrogate = GaussianProcessSurrogate(
                                    std_treshold=0.05, min_data_to_train=50
                                )
                                bounds = list(zip(f.lower_bounds, f.upper_bounds))
                                des = DES(
                                    f,
                                    dim,
                                    bounds,
                                    MAX_EVALS,
                                    surrogate_model=surrogate,
                                )

                            result = run_and_track(des, f, dim, MAX_EVALS)
                            rows.append(result)

                        save_results(func_name, variant, rows)


def save_results(func_name, variant, runs_data):
    csv_path = os.path.join(RESULTS_DIR, f"{func_name}_{variant}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "EvalCount", "Time", "BestVal", "BestHistory..."])
        for i, res in enumerate(runs_data):
            writer.writerow(
                [
                    i + 1,
                    res["evals"],
                    f"{res['time']:.4f}",
                    f"{res['best_val']:.6e}",
                    ",".join(f"{v:.6e}" for v in res["best_so_far"]),
                ]
            )
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    run_experiment()
