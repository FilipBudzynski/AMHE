from abc import ABC, abstractmethod


class Evaluator(ABC):

    @abstractmethod
    def evaluate(self, population) -> list[float]:
        pass


class DESEvaluator(Evaluator):
    def __init__(self, bounds, dim, obj_func):
        self.bounds = bounds
        self.dim = dim
        self.obj_func = obj_func
        self.qmax = float("-inf")

    def evaluate(self, population) -> list[float]:
        fitness = []
        for x in population:
            penalty = 0.0
            for i in range(self.dim):
                if x[i] < self.bounds[0, i]:
                    penalty += (self.bounds[0, i] - x[i]) ** 2
                elif x[i] > self.bounds[1, i]:
                    penalty += (x[i] - self.bounds[1, i]) ** 2

            f_val = self.obj_func(x)

            if penalty > 0:
                f_val = max(f_val, self.qmax) + penalty  # infeasible
            else:
                self.qmax = max(self.qmax, f_val)  # feasible

            fitness.append(f_val)
        return fitness
