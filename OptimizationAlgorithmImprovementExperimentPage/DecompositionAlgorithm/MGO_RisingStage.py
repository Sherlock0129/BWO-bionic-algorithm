import numpy as np
import random

def coefficient_vector(dim, Iter, Max_it):
    a2 = -1 + Iter * (-1 / Max_it)
    u = np.random.randn(1, dim)
    v = np.random.randn(1, dim)

    cofi = np.zeros((4, dim))
    cofi[0, :] = np.random.rand(1, dim)
    cofi[1, :] = (a2 + 1) + np.random.rand(1, dim)
    cofi[2, :] = a2 * np.random.randn(1, dim)
    cofi[3, :] = u * v ** 2 * np.cos((np.random.rand(1, dim) * 2) * u)

    return cofi

def rising_phase(Npop, Max_it, lb, ub, nD, fobj, Positions, fitness, Best_position, Best_fitness, t):
    RandomSolution = np.random.choice(Npop, size=int(np.ceil(Npop / 3)), replace=False)
    M = Positions[np.random.randint(np.ceil(Npop / 3), Npop), :] * np.floor(np.random.rand()) + np.mean(Positions[RandomSolution, :], axis=0) * np.ceil(np.random.rand())
    cofi = coefficient_vector(nD, t, Max_it)
    A = np.random.randn(1, nD) * np.exp(2 - t * (2 / Max_it))
    D = (np.abs(Positions) + np.abs(Best_position)) * (2 * np.random.rand(1, nD) - 1)
    NewX = np.zeros((4, nD))
    for i in range(Npop):
        NewX[0, :] = (ub - lb) * np.random.rand(1, nD) + lb
        NewX[1, :] = Best_position - np.abs((np.random.randint(2) * M - np.random.randint(2) * Positions[i, :]) * A) * cofi[0, :]
        NewX[2, :] = (M + cofi[0, :]) + (np.random.randint(2) * Best_position - np.random.randint(2) * Positions[np.random.randint(Npop), :]) * cofi[0, :]
        NewX[3, :] = (Positions[i, :] - D) + (np.random.randint(2) * Best_position - np.random.randint(2) * M) * cofi[0, :]
        NewX = np.clip(NewX, lb, ub)
        fitness_new = np.array([fobj(NewX[j, :]) for j in range(NewX.shape[0])])
        if fitness_new.min() < fitness[i]:
            Positions[i, :] = NewX[fitness_new.argmin(), :]
            fitness[i] = fitness_new.min()
            if fitness[i] < Best_fitness:
                Best_position = Positions[i, :]
                Best_fitness = fitness[i]
    return Positions, fitness, Best_position, Best_fitness