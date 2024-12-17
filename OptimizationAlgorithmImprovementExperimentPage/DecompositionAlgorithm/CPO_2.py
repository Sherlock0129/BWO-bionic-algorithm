import random
import numpy as np

def CPO_exploitation_phase(Npop, Max_it, lb, ub, nD, fobj, g_best):
    """ Exploitation Phase of CPO Algorithm """
    X = np.random.uniform(lb, ub, (Npop, nD))  # Initialize population
    fitness = np.apply_along_axis(fobj, 1, X)

    g_best_score = fobj(g_best)
    curve = []

    for t in range(Max_it):
        for i in range(Npop):
            r3 = np.random.uniform(-1, 1)

            # Exploitation: Move closer to global best solution
            X_new = g_best + r3 * (g_best - X[i]) * np.exp(-t / Max_it)
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