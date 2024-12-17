import numpy as np

# CPO Algorithm Implementation with Three Stages
def CPO_exploration_phase(Npop, Max_it, lb, ub, nD, fobj):
    """ Exploration Phase of CPO Algorithm """
    X = np.random.uniform(lb, ub, (Npop, nD))  # Initialize population
    fitness = np.apply_along_axis(fobj, 1, X)  # Fitness evaluation

    g_best = X[np.argmin(fitness)]  # Global best solution
    g_best_score = np.min(fitness)
    curve = []

    for t in range(Max_it):
        for i in range(Npop):
            r1, r2 = np.random.uniform(-1, 1, 2)
            X_random = X[np.random.randint(0, Npop)]

            # Exploration: Expand search space
            X_new = X[i] + r1 * (g_best - X[i]) + r2 * (X_random - X[i])
            X_new = np.clip(X_new, lb, ub)

            new_fitness = fobj(X_new)
            if new_fitness < fitness[i]:
                X[i] = X_new
                fitness[i] = new_fitness

        # Update global best
        if np.min(fitness) < g_best_score:
            g_best = X[np.argmin(fitness)]
            g_best_score = np.min(fitness)

        curve.append(g_best_score)

    return g_best, g_best_score, curve

