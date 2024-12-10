import numpy as np
import math

def levy(n, m, beta):
    num = math.gamma(1 + beta) * math.sin(np.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    return u / (np.abs(v) ** (1 / beta))

def decline_landing_phase(Npop, Max_it, lb, ub, nD, fobj, Positions, fitness, Best_position, Best_fitness, t):
    alpha = random.random() * ((1 / Max_it ** 2) * t ** 2 - 2 / Max_it * t + 1)
    beta = np.random.randn(Npop, nD)

    # Decline stage
    dandelions_mean = np.mean(Positions, axis=0)
    dandelions_2 = np.zeros_like(Positions)
    for i in range(Npop):
        for j in range(nD):
            dandelions_2[i, j] = Positions[i, j] - beta[i, j] * alpha * (dandelions_mean[j] - beta[i, j] * alpha * Positions[i, j])

    Positions = np.clip(dandelions_2, lb, ub)

    # Landing stage
    Step_length = levy(Npop, nD, 1.5)
    Elite = np.tile(Best_position, (Npop, 1))
    dandelions_3 = np.zeros_like(Positions)
    for i in range(Npop):
        for j in range(nD):
            dandelions_3[i, j] = Elite[i, j] + Step_length[i, j] * alpha * (Elite[i, j] - Positions[i, j] * (2 * t / Max_it))

    Positions = np.clip(dandelions_3, lb, ub)

    # Calculate fitness values
    for i in range(Npop):
        fitness[i] = fobj(Positions[i, :])

    # Sort dandelions based on fitness
    sorted_indexes = np.argsort(fitness)
    Positions = Positions[sorted_indexes[:Npop], :]
    SortfitbestN = fitness[sorted_indexes[:Npop]]

    # Update the best position and fitness
    if SortfitbestN[0] < Best_fitness:
        Best_position = Positions[0, :]
        Best_fitness = SortfitbestN[0]

    return Positions, fitness, Best_position, Best_fitness