import numpy as np


class DES:
    def __init__(
        self,
        objective_func,
        dim,
        bounds,
        max_evals,
        population_size=None,
        mu=None,
        archive_horizon=None,
        F=None,
        c=None,
        epsilon=None,
    ):
        self.obj_func = objective_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.max_evals = max_evals

        self.lambda_ = population_size or 4 * dim
        self.mu = mu or self.lambda_ // 2
        self.F = F or 1.0 / np.sqrt(2)
        self.c = c or (4.0 / (dim + 4))
        self.H = archive_horizon or int(6 + 3 * np.sqrt(dim))
        self.epsilon = epsilon or 1e-6 / np.sqrt(self.dim)

        self.archive = []
        self.qmax = float("-inf")

    def initialize_population(self):
        return np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(self.lambda_, self.dim)
        )

    def evaluate(self, pop):
        fitness = []
        for x in pop:
            penalty = 0.0
            for i in range(self.dim):
                if x[i] < self.bounds[i, 0]:
                    penalty += (self.bounds[i, 0] - x[i]) ** 2
                elif x[i] > self.bounds[i, 1]:
                    penalty += (x[i] - self.bounds[i, 1]) ** 2
            f_val = self.obj_func(x)
            if penalty > 0:
                f_val = max(f_val, self.qmax) + penalty
            else:
                self.qmax = max(self.qmax, f_val)
            fitness.append(f_val)
        return np.array(fitness)

    def select_mu_best(self, pop, fitness):
        indices = np.argsort(fitness)
        return pop[indices[: self.mu]]

    def update_midpoint_shift(self, delta, m, s):
        return (1 - self.c) * delta + self.c * (s - m)

    def generate_offspring(self, s, delta):
        new_pop = []
        for _ in range(self.lambda_):
            h = np.random.randint(min(len(self.archive), self.H))
            archive_pop = self.archive[-(h + 1)]

            j, k = np.random.choice(self.mu, 2, replace=False)
            diff = archive_pop[j] - archive_pop[k]

            direction = (
                np.random.normal()
                * delta
                * np.linalg.norm(np.random.normal(size=self.dim))
            )
            noise = self.epsilon * np.random.normal(size=self.dim)

            x_new = s + self.F * diff + direction + noise
            new_pop.append(x_new)
        return np.array(new_pop)

    def run(self):
        eval_count = 0
        pop = self.initialize_population()
        delta = np.zeros(self.dim)
        m = pop.mean(axis=0)

        best_solution = None
        best_value = float("inf")

        while eval_count < self.max_evals:
            f_pop = self.evaluate(pop)
            eval_count += len(f_pop)

            min_idx = np.argmin(f_pop)
            if f_pop[min_idx] < best_value:
                best_value = f_pop[min_idx]
                best_solution = pop[min_idx]

            s = self.select_mu_best(pop, f_pop).mean(axis=0)
            delta = self.update_midpoint_shift(delta, m, s)

            self.archive.append(pop.copy())
            if len(self.archive) > self.H:
                self.archive.pop(0)

            pop = self.generate_offspring(s, delta)
            m = s

            if np.mean(np.std(pop, axis=0)) < self.epsilon:
                break

        return best_solution, best_value
