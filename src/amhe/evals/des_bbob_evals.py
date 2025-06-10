import os
import numpy as np
import matplotlib.pyplot as plt
from cocoex import Suite, Observer
import cocopp
from amhe.algorithm.des import DES
from amhe.surrogate import GaussianProcessSurrogate

# Set experiment details
# dimensions = [5, 10, 20, 40]
dimensions = [5]
problem_groups = [
    list(range(1, 6)),     # Group 1
    # list(range(6, 10)),    # Group 2
    # list(range(10, 15)),   # Group 3
    # list(range(15, 20)),   # Group 4
    # list(range(20, 25)),   # Group 5
]
max_runs = 10
max_evals_factor = 10000  # as per BBOB standard

# Main loop
for dim in dimensions:
    for group_idx, problem_ids in enumerate(problem_groups):
        exp_name = f'DES_dim{dim}_group{group_idx+1}'
        result_folder = f'./results/{exp_name}'
        os.makedirs(result_folder, exist_ok=True)

        # Setup suite and observer
        suite = Suite("bbob", f"function_indices:{','.join(map(str, problem_ids))}", f"dimensions:{dim}")
        observer = Observer("bbob", f"result_folder:{result_folder}")

        for problem in suite:
            problem.observe_with(observer)
            for run in range(max_runs):
                np.random.seed(run)
                des = DES(
                    lambda x: problem(x),
                    dim,
                    bounds=[problem.lower_bounds, problem.upper_bounds],
                    max_evals=dim * max_evals_factor,
                    surrogate_model=GaussianProcessSurrogate(
                        std_treshold=0.5, min_data_to_train=50
                    ),
                )
                des.run()
                # problem.finalize()  # Required by COCO
