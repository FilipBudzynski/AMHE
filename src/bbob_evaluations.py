import cocoex  # experimentation module
import cocopp  # post-processing module (not strictly necessary)
from src.algorithm.des import DES
from src.surrogate import GaussianProcessSurrogate

### input
suite_name = "bbob"
budget_multiplier = 1  # x dimension
max_evals_factor = budget_multiplier  # used in DES as dim * max_evals_factor

### prepare
suite = cocoex.Suite(suite_name, "", "")
output_folder = 'DES_on_{}_{}D'.format(suite_name, int(budget_multiplier))
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
repeater = cocoex.ExperimentRepeater(budget_multiplier)
minimal_print = cocoex.utilities.MiniPrint()

### go
while not repeater.done():
    for problem in suite:
        if repeater.done(problem):
            continue
        problem.observe_with(observer)
        problem(problem.dimension * [0])  # for comparability

        dim = problem.dimension
        if problem.dimension != 2:
            print("xd")
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

        repeater.track(problem)
        minimal_print(problem)

### post-process data
cocopp.main(observer.result_folder)
