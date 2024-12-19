import numpy as np


def mgo_phase_1(Npop, Max_it, lb, ub, nD, fobj):
    # 初始化种群
    population = np.random.uniform(lb, ub, (Npop, nD))
    fitness = np.array([fobj(ind) for ind in population])
    best_idx = np.argmin(fitness)
    xposbest = population[best_idx]
    fvalbest = fitness[best_idx]
    Curve = [fvalbest]

    for _ in range(Max_it):
        # 移动策略（简单示例）
        for i in range(Npop):
            population[i] += np.random.uniform(-1, 1, nD) * (xposbest - population[i])
            population[i] = np.clip(population[i], lb, ub)
            fitness[i] = fobj(population[i])

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < fvalbest:
            xposbest = population[best_idx]
            fvalbest = fitness[best_idx]

        Curve.append(fvalbest)

    return xposbest, fvalbest, Curve

