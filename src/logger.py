from abc import ABC, abstractmethod


class Logger(ABC):
    pass


class SimpleLogger(Logger):
    def __init__(self) -> None:
        self.fitness_history = []
        self.evals_history = []
        self.best_so_far = float("inf")
        self.eval_count = 0

    def log_evaluation(self, fitness_value):
        self.eval_count += 1
        if fitness_value < self.best_so_far:
            self.best_so_far = fitness_value
        self.fitness_history.append(self.best_so_far)
        self.evals_history.append(self.eval_count)

    def dump(self):
        return self.evals_history, self.fitness_history
