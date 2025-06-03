import numpy as np
import math
from logger import SimpleLogger


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
        cc=None,
        cd=None,
        ce=None,
        epsilon=None,
        logger=SimpleLogger,
    ):
        self.obj_func = objective_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.max_evals = max_evals

        self.lambda_ = population_size or 4 + math.floor(3 * np.log(dim))
        self.mu = mu or self.lambda_ // 2
        self.cc = cc or (1.0 / np.sqrt(dim))
        self.cd = cd or (self.mu / (self.mu + 2))
        self.ce = ce or (2.0 / dim**2)

        self.H = archive_horizon or int(6 + 3 * np.sqrt(dim))
        self.epsilon = epsilon or 1e-6

        self.archive = []
        self.p_archive = []
        self.qmax = float("-inf")
        self.logger = logger()

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
                f_val = max(f_val, self.qmax) + penalty  # infeasible
            else:
                self.qmax = max(self.qmax, f_val)  # feasible

            fitness.append(f_val)
        return np.array(fitness)

    def select_mu_best(self, pop, fitness):
        indices = np.argsort(fitness)
        return pop[indices[: self.mu]]

    def run(self):
        eval_count = 0
        t = 1

        # Step 1–3
        pop = self.initialize_population()
        m_t = pop.mean(axis=0)
        fitness = self.evaluate(pop)
        eval_count += self.lambda_

        self.archive.append((pop.copy(), fitness.copy()))

        delta_t = np.zeros(self.dim)
        p_t = None

        best_solution = None
        best_value = float("inf")

        while eval_count < self.max_evals:
            # Step 6: mean of best mu individuals
            mu_best = self.select_mu_best(pop, fitness)
            m_tp1 = mu_best.mean(axis=0)

            # Step 7: delta
            delta_t = m_tp1 - m_t

            # Step 8–11: evolution path
            if t == 1:
                p_t = delta_t
            else:
                p_t = (1 - self.cc) * p_t + np.sqrt(
                    self.mu * self.cc * (2 - self.cc)
                ) * delta_t

            self.p_archive.append(p_t.copy())
            if len(self.p_archive) > self.H:
                self.p_archive.pop(0)

            # Step 13–21: generate new population
            new_pop = []
            for _ in range(self.lambda_):
                # Step 14: random tau1, tau2, tau3 in {1, ..., H}
                tau1 = np.random.randint(1, min(len(self.archive), self.H) + 1)
                tau2 = np.random.randint(1, min(len(self.archive), self.H) + 1)
                tau3 = np.random.randint(1, min(len(self.archive), self.H) + 1)

                # Step 15: random j, k from {1, ..., mu}
                j, k = np.random.choice(self.mu, 2, replace=False)

                # Step 16–19: build direction vector d_i^(t)
                pop_t, fittness_t = self.archive[-tau1]
                x_archive_tau1 = self.select_mu_best(pop_t, fittness_t)
                xj = x_archive_tau1[j]
                xk = x_archive_tau1[k]
                diff_term = np.sqrt(self.cd / 2) * (xj - xk)

                # Δ^(t−tau2)
                if len(self.archive) > tau2:
                    pop_t, fittness_t = self.archive[-tau2]
                    m_prev = self.select_mu_best(pop_t, fittness_t).mean(axis=0)

                    pop_t, fittness_t = self.archive[-tau2 - 1]
                    m_prev_prev = self.select_mu_best(pop_t, fittness_t).mean(axis=0)

                    delta_past = m_prev - m_prev_prev
                else:
                    delta_past = np.zeros(self.dim)

                delta_term = np.sqrt(self.cd) * delta_past * np.random.randn(self.dim)

                # p^(t−tau3)
                if len(self.p_archive) >= tau3:
                    p_past = self.p_archive[-tau3]
                else:
                    p_past = np.zeros(self.dim)
                p_term = np.sqrt(1 - self.cd) * p_past * np.random.randn(self.dim)

                # ε term
                epsilon_term = (
                    self.epsilon
                    * ((1 - self.ce) ** (t / 2))
                    * np.random.randn(self.dim)
                )

                d_i = diff_term + delta_term + p_term + epsilon_term

                # Step 20
                x_new = m_tp1 + d_i
                new_pop.append(x_new)

            pop = np.array(new_pop)
            fitness = self.evaluate(pop)
            eval_count += self.lambda_

            m_t = m_tp1
            self.archive.append((pop.copy(), fitness.copy()))
            if len(self.archive) > self.H:
                self.archive.pop(0)

            # Step 22: update best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_value:
                best_value = fitness[min_idx]
                best_solution = pop[min_idx]

            if self.logger:
                self.logger.log(best_value, eval_count)

            t += 1

            # optional stop
            if np.mean(np.std(pop, axis=0)) < self.epsilon:
                break

        return best_solution, best_value
