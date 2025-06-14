import numpy as np
import matplotlib.pyplot as plt
from algorithm.des import DES
from surrogate import GaussianProcessSurrogate
from functions.functions import elipsoid, sphere, cigar, discus, ackley, rosenbrock
from functions.count_calls import count_calls
import random
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    SEED = 42
    dim = 10
    max_evals = 10000
    bounds = np.array([[-5] * dim, [5] * dim])
    num_runs = 50

    benchmarks = {
        "Sphere": sphere,
        "Discus": discus,
        "Cigar": cigar,
        "Ackley": ackley,
        "Rosenbrock": rosenbrock,
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (name, func) in enumerate(benchmarks.items()):
        print(f"\nRunning 50x evaluations on {name}...")

        all_plain_histories = []
        all_surr_histories = []

        for run in range(num_runs):
            np.random.seed(SEED + run)
            random.seed(SEED + run)

            wrapped_func = count_calls(func)

            # DES
            wrapped_func.call_count = 0
            des_plain = DES(
                wrapped_func,
                dim,
                bounds,
                max_evals,
                surrogate_model=None,
            )
            des_plain.run()
            evals_plain, fitness_hist_plain = des_plain.logger.dump()
            interp_plain = np.interp(
                np.arange(max_evals),
                np.array(evals_plain),
                np.array(fitness_hist_plain),
            )
            all_plain_histories.append(interp_plain)

            # DES + Surrogate
            wrapped_func.call_count = 0
            des_surr = DES(
                wrapped_func,
                dim,
                bounds,
                max_evals,
                surrogate_model=GaussianProcessSurrogate(
                    std_treshold=0.005,
                    min_data_to_train=dim * 8,
                    train_window_size=dim * 50,
                ),
            )
            des_surr.run()
            evals_surr, fitness_hist_surr = des_surr.logger.dump()
            interp_surr = np.interp(
                np.arange(max_evals),
                np.array(evals_surr),
                np.array(fitness_hist_surr),
            )
            all_surr_histories.append(interp_surr)

            print(f"Run {run+1}/{num_runs} complete")

        avg_plain = np.mean(all_plain_histories, axis=0)
        avg_surr = np.mean(all_surr_histories, axis=0)

        ax = axes[idx]
        ax.plot(avg_plain, label="DES only", linewidth=2)
        ax.plot(avg_surr, label="DES + Surrogate", linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel("Average Best Fitness")
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1e6)
        ax.grid(True)
        ax.legend()

    if len(benchmarks) < len(axes):
        for i in range(len(benchmarks), len(axes)):
            fig.delaxes(axes[i])

    fig.suptitle(
        f"DES vs DES + Surrogate (Average of {num_runs} runs, dim={dim})", fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("des_vs_surrogate_all_functions.png", dpi=300)
    plt.show()
