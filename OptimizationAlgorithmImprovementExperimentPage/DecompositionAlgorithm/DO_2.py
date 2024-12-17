import numpy as np
import random

def rising_phase(Npop, Max_it, lb, ub, nD, fobj, Positions, fitness, Best_position, Best_fitness, t):
    beta = np.random.randn(Npop, nD)
    alpha = random.random() * ((1 / Max_it ** 2) * t ** 2 - 2 / Max_it * t + 1)
    a = 1 / (Max_it ** 2 - 2 * Max_it + 1)
    b = -2 * a
    c = 1 - a - b
    k = 1 - random.random() * (c + a * t ** 2 + b * t)

    if np.random.randn() < 1.5:
        dandelions_1 = np.zeros_like(Positions)
        for i in range(Npop):
            lamb = np.abs(np.random.randn(1, nD))
            theta = (2 * np.random.rand() - 1) * np.pi
            row = 1 / np.exp(theta)
            vx = row * np.cos(theta)
            vy = row * np.sin(theta)
            NEW = np.random.rand(1, nD) * (ub - lb) + lb
            dandelions_1[i, :] = Positions[i, :] + alpha * vx * vy * np.log(np.random.normal(0, 1)) * (NEW[0, :] - Positions[i, :])
    else:
        dandelions_1 = Positions * k

    Positions = np.clip(dandelions_1, lb, ub)
    return Positions, fitness, Best_position, Best_fitnessest_fitness