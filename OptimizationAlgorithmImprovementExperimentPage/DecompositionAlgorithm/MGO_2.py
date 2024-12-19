import numpy as np
import random

def mgo_phase_2(Npop, Max_it, lb, ub, nD, fobj, g_best):
    population = np.random.uniform(lb, ub, (Npop, nD))
    fitness = np.array([fobj(ind) for ind in population])
    xposbest = g_best
    fvalbest = fobj(g_best)
    Curve = [fvalbest]

    for _ in range(Max_it):
        for i in range(Npop):
            population[i] += np.random.uniform(-0.5, 0.5, nD) * (xposbest - population[i])
            population[i] = np.clip(population[i], lb, ub)
            fitness[i] = fobj(population[i])

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < fvalbest:
            xposbest = population[best_idx]
            fvalbest = fitness[best_idx]

        Curve.append(fvalbest)

    return xposbest, fvalbest, Curve