import matplotlib.pyplot as plt
import cocopp
import os

dimensions = [5, 10, 20, 40]
problem_groups = ["group1", "group2", "group3", "group4", "group5"]

fig, axs = plt.subplots(5, 4, figsize=(20, 25), sharex=True, sharey=True)

for i, group in enumerate(problem_groups):
    for j, dim in enumerate(dimensions):
        exp_name = f'DES_dim{dim}_{group}'
        result_path = os.path.join("results", exp_name)
        if not os.path.exists(result_path):
            continue

        # Load and plot ECDF using cocopp
        data = cocopp.load(result_path)
        cocopp.bbobplot.ecdf(data)
        axs[i, j].set_title(f'Group {i+1}, n={dim}')

        # Hide global plot from showing
        if hasattr(plt, 'clf'): plt.clf()

# Final layout and show
plt.tight_layout()
plt.show()
