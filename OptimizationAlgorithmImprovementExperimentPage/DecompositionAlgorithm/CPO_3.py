import random
import math
import numpy as np

def CPO_convergence_phase(Npop, Max_it, lb, ub, nD, fobj, g_best):
    """ Convergence Phase of CPO Algorithm """
    X = np.random.uniform(lb, ub, (Npop, nD))  # Initialize population
    fitness = np.apply_along_axis(fobj, 1, X)

    g_best_score = fobj(g_best)
    curve = []

    for t in range(Max_it):
        for i in range(Npop):
            # Convergence: Reduce perturbation further
            r4 = np.random.uniform(0, 1)
            X_new = g_best + r4 * (g_best - X[i]) * np.exp(-2 * t / Max_it)
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

