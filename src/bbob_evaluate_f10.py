import cocoex  # experimentation module
import cocopp  # post-processing module (not strictly necessary)
from algorithm.des import DES
from surrogate import GaussianProcessSurrogate

suite_name = "bbob"
budget_multiplier = 10000 # x dimension
max_evals_factor = budget_multiplier  # used in DES as dim * max_evals_factor

target_fun_id = 10
target_dimensions = [5, 10, 20]

output_folder = "bbob_f10"
observer = cocoex.Observer(suite_name, f"result_folder: {output_folder}")
suite = cocoex.Suite(
    suite_name,
    "",
    f"function_indices:{target_fun_id} dimensions:{','.join(map(str, target_dimensions))}"
)

### go
for problem in suite:
    dim = problem.dimension
    print(f"\nRunning function f{target_fun_id} in dimension {dim}")

    problem.observe_with(observer)
    problem(dim * [0])  

    des = DES(
        lambda x: problem(x),
        dim,
        bounds=[problem.lower_bounds, problem.upper_bounds],
        max_evals=dim * max_evals_factor,
        surrogate_model=GaussianProcessSurrogate(
            std_treshold=0.005,
            min_data_to_train=dim*3,
            train_window_size=dim*30,
        ),
    )
    des.run()

### post-process data
cocopp.main(observer.result_folder)

