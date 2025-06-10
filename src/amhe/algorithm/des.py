import numpy as np
import math
from typing import Optional
from amhe.evaluator import DESEvaluator
from amhe.logger import SimpleLogger
from amhe.surrogate import Surrogate


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
        evaluator=None,
        surrogate_model: Surrogate | None = None,
        logger=SimpleLogger,
    ):
        self.obj_func = objective_func
        self.dim = dim
        self.bounds = np.array(bounds)
        if self.bounds.shape != (self.dim, 2):
            self.bounds = np.array([self.bounds[0], self.bounds[1]]).T
        self.max_evals = max_evals

        self._lambda = population_size or 4 + math.floor(3 * np.log(dim))
        self.mu = mu or self._lambda // 2
        self.cc = cc or (1 / np.sqrt(dim))
        self.cd = cd or (self.mu / (self.mu + 2))
        self.ce = ce or (2 / dim**2)

        self.H = archive_horizon or int(6 + 3 * np.sqrt(dim))
        self.epsilon = epsilon or 1e-6

        self.archive = []
        self.p_archive = []
        self.logger = logger()

        self.evaluator = evaluator or DESEvaluator(
            self.bounds, self.dim, objective_func
        )
        self.surrogate_model = surrogate_model or None

    def initialize_population(self):
        return np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(self._lambda, self.dim)
        )

    def evaluate(self, pop):
        fitnesses = [None] * len(pop)
        num_real_evals = 0

        indices_to_evaluate = []
        predictions = []
        if self.surrogate_model:
            indices_to_evaluate, predictions = self.surrogate_model.predict(pop)

        if len(indices_to_evaluate) > 0 and len(predictions) > 0:

            true_fitnesses = self.evaluator.evaluate(
                [pop[i] for i in indices_to_evaluate]
            )
            num_real_evals += len(indices_to_evaluate)

            for idx, fit in zip(indices_to_evaluate, true_fitnesses):
                fitnesses[idx] = fit
                self.surrogate_model.append(pop[idx], fit)

            for i in range(len(pop)):
                if fitnesses[i] is None:
                    fitnesses[i] = predictions[i]

            if self.logger:
                for fit in fitnesses:
                    self.logger.log_evaluation(fit)

        else:
            fitnesses = self.evaluator.evaluate(pop)
            num_real_evals += len(pop)

            if self.surrogate_model:
                self.surrogate_model.append(pop, fitnesses)

            if self.logger:
                for fit in fitnesses:
                    self.logger.log_evaluation(fit)

        return fitnesses, num_real_evals

    def select_mu_best(self, pop, fitness):
        indices = np.argsort(fitness)
        return pop[indices[: self.mu]]

    def run(self):
        eval_count = 0
        t = 1

        # Step 1–3
        pop = self.initialize_population()
        m_t = pop.mean(axis=0)
        fitness, evals = self.evaluate(pop)
        eval_count += evals
        # eval_count += self._lambda

        self.archive.append((pop.copy(), fitness.copy()))

        delta_t = np.zeros(self.dim)
        p_t = None

        best_solution = None
        best_value = float("inf")

        while eval_count < self.max_evals:
            if self.surrogate_model:
                self.surrogate_model.train(t)

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
            for _ in range(self._lambda):
                # Step 14: random tau1, tau2, tau3 in {1, ..., H}
                tau1 = np.random.randint(0, min(len(self.archive), self.H))
                tau2 = np.random.randint(0, min(len(self.archive), self.H))
                tau3 = np.random.randint(0, min(len(self.archive), self.H))

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
                    p_past = p_t
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
            fitness, evals = self.evaluate(pop)
            eval_count += evals
            # eval_count += self._lambda

            m_t = m_tp1
            self.archive.append((pop.copy(), np.array(fitness).copy()))
            if len(self.archive) > self.H:
                self.archive.pop(0)

            # Step 22: update best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_value:
                best_value = fitness[min_idx]
                best_solution = pop[min_idx]

            t += 1

            if np.mean(np.std(pop, axis=0)) < self.epsilon:
                break

        return best_solution, best_value
