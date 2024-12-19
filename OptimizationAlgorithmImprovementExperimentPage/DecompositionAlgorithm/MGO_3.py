import random

import numpy as np
import math


def mgo_phase_3(Npop, Max_it, lb, ub, nD, fobj, g_best):
    population = np.random.uniform(lb, ub, (Npop, nD))
    fitness = np.array([fobj(ind) for ind in population])
    xposbest = g_best
    fvalbest = fobj(g_best)
    Curve = [fvalbest]

    for _ in range(Max_it):
        for i in range(Npop):
            step_size = 0.1 * np.random.randn(nD)  # 更小的探索步长
            population[i] += step_size * (xposbest - population[i])
            population[i] = np.clip(population[i], lb, ub)
            fitness[i] = fobj(population[i])

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < fvalbest:
            xposbest = population[best_idx]
            fvalbest = fitness[best_idx]

        Curve.append(fvalbest)

    return xposbest, fvalbest, Curve