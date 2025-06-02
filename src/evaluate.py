import numpy as np
import matplotlib.pyplot as plt
from des import DES


# Define benchmark functions
def q1(x):
    return np.sum(x**2)


def q2(x):
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)


def q3(x):
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)


# Wrapper function to run DES2 and track progress
def run_des2_with_tracking(des2_instance):
    eval_count = 0
    t = 1

    pop = des2_instance.initialize_population()
    m_t = pop.mean(axis=0)
    fitness = des2_instance.evaluate(pop)
    eval_count += des2_instance.lambda_

    des2_instance.archive.append(pop.copy())

    delta_t = np.zeros(des2_instance.dim)
    p_t = None

    best_solution = None
    best_value = float("inf")

    fitness_history = []
    evals_history = []

    while eval_count < des2_instance.max_evals:
        mu_best = des2_instance.select_mu_best(pop, fitness)
        m_tp1 = mu_best.mean(axis=0)
        delta_tp1 = m_tp1 - m_t

        if t == 1:
            p_t = delta_tp1
        else:
            p_t = (1 - des2_instance.cc) * p_t + np.sqrt(
                des2_instance.mu * des2_instance.cc * (2 - des2_instance.cc)
            ) * delta_tp1

        new_pop = []
        for _ in range(des2_instance.lambda_):
            tau1 = np.random.randint(
                1, min(len(des2_instance.archive), des2_instance.H) + 1
            )
            tau2 = np.random.randint(
                1, min(len(des2_instance.archive), des2_instance.H) + 1
            )
            tau3 = np.random.randint(
                1, min(len(des2_instance.archive), des2_instance.H) + 1
            )

            j, k = np.random.choice(des2_instance.mu, 2, replace=False)

            x_archive_tau1 = des2_instance.select_mu_best(
                des2_instance.archive[-tau1],
                des2_instance.evaluate(des2_instance.archive[-tau1]),
            )
            xj = x_archive_tau1[j]
            xk = x_archive_tau1[k]
            diff_term = np.sqrt(des2_instance.cd / 2) * (xj - xk)

            if len(des2_instance.archive) > tau2:
                m_prev = des2_instance.select_mu_best(
                    des2_instance.archive[-tau2],
                    des2_instance.evaluate(des2_instance.archive[-tau2]),
                ).mean(axis=0)
                m_prev_prev = des2_instance.select_mu_best(
                    des2_instance.archive[-tau2 - 1],
                    des2_instance.evaluate(des2_instance.archive[-tau2 - 1]),
                ).mean(axis=0)
                delta_past = m_prev - m_prev_prev
            else:
                delta_past = np.zeros(des2_instance.dim)

            delta_term = (
                np.sqrt(des2_instance.cd)
                * delta_past
                * np.random.randn(des2_instance.dim)
            )
            p_term = (
                np.sqrt(1 - des2_instance.cd) * p_t * np.random.randn(des2_instance.dim)
            )
            epsilon_term = (
                des2_instance.epsilon
                * ((1 - des2_instance.ce) ** (t / 2))
                * np.random.randn(des2_instance.dim)
            )

            d_i = diff_term + delta_term + p_term + epsilon_term
            x_new = m_tp1 + d_i
            new_pop.append(x_new)

        pop = np.array(new_pop)
        fitness = des2_instance.evaluate(pop)
        eval_count += des2_instance.lambda_

        m_t = m_tp1
        delta_t = delta_tp1
        des2_instance.archive.append(pop.copy())
        if len(des2_instance.archive) > des2_instance.H:
            des2_instance.archive.pop(0)

        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_value:
            best_value = fitness[min_idx]
            best_solution = pop[min_idx]

        fitness_history.append(best_value)
        evals_history.append(eval_count)

        t += 1

        if np.mean(np.std(pop, axis=0)) < des2_instance.epsilon:
            break

    return best_solution, best_value, evals_history, fitness_history


# Setup for testing
dim = 3
max_evals = 20000
bounds = [(-5, 5)] * dim

functions = [q1, q2, q3]
names = ["Well-conditioned Sphere (q1)", "Cigar (q2)", "Discus (q3)"]


all_histories = {}

for func, name in zip(functions, names):
    print(f"Running DES2 on {name}...")
    des2 = DES(func, dim, bounds, max_evals)  # Use your actual DES2 class here
    best_sol, best_val, evals, fitness_hist = run_des2_with_tracking(des2)
    print(f"Best value found: {best_val:.6f}")
    all_histories[name] = (evals, fitness_hist)

# Plot results
plt.figure(figsize=(12, 8))
for name, (evals, fitness) in all_histories.items():
    plt.plot(evals, fitness, label=name)

plt.xlabel("Function Evaluations")
plt.ylabel("Best Fitness Value Found")
plt.title("DES2 Optimization Performance on Benchmark Functions (n=3, 20000 evals)")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()
