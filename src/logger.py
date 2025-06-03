from abc import ABC, abstractmethod


class Logger(ABC):

    @abstractmethod
    def log(self, best_value, eval_count):
        pass


class SimpleLogger(Logger):
    def __init__(self) -> None:
        self.fitness_history = []
        self.evals_history = []

    def log(self, best_value, eval_count):
        self.fitness_history.append(best_value)
        self.evals_history.append(eval_count)

    def dump(self):
        return self.evals_history, self.fitness_history
