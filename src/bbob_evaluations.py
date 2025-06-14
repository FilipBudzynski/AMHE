import cocoex 
import cocopp  
from algorithm.des import DES
from surrogate import GaussianProcessSurrogate
import warnings
warnings.filterwarnings("ignore")

suite_name = "bbob"
budget_multiplier = 10000  
max_evals_factor = budget_multiplier  

suite = cocoex.Suite(suite_name, "", "")
output_folder = 'DES_on_{}_{}D'.format(suite_name, int(budget_multiplier))
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
repeater = cocoex.ExperimentRepeater(budget_multiplier)
minimal_print = cocoex.utilities.MiniPrint()

while not repeater.done():
    for problem in suite:
        if repeater.done(problem):
            continue
        problem.observe_with(observer)
        problem(problem.dimension * [0])  

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

cocopp.main(observer.result_folder)
