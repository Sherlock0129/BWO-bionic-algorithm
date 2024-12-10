import numpy as np
import math
import random
import matplotlib.pyplot as plt


def levy(n, m, beta):
    num = math.gamma(1 + beta) * math.sin(np.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    return u / (np.abs(v) ** (1 / beta))


def initialization(Npop, nD, ub, lb):
    Boundary_no = len(ub)

    if Boundary_no == 1:
        # All variables share the same boundary
        return np.random.rand(Npop, nD) * (ub - lb) + lb
    else:
        # Different boundaries for each dimension
        Positions = np.zeros((Npop, nD))
        for i in range(nD):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(Npop) * (ub_i - lb_i) + lb_i
        return Positions


def DO(Npop, Max_it, lb, ub, nD, fobj):
    Best_fitness = float('inf')
    Best_position = np.zeros(nD)
    Convergence_curve = np.zeros(Max_it)

    dandelions = initialization(Npop, nD, ub, lb)
    dandelionsFitness = np.zeros(Npop)

    for i in range(Npop):
        dandelionsFitness[i] = fobj(dandelions[i, :])

    # Calculate the fitness values of initial dandelions
    sorted_indexes = np.argsort(dandelionsFitness)
    Best_position = dandelions[sorted_indexes[0], :]
    Best_fitness = dandelionsFitness[sorted_indexes[0]]
    Convergence_curve[0] = Best_fitness
    t = 1

    while t < Max_it:
        # Rising stage
        beta = np.random.randn(Npop, nD)
        alpha = random.random() * ((1 / Max_it ** 2) * t ** 2 - 2 / Max_it * t + 1)
        a = 1 / (Max_it ** 2 - 2 * Max_it + 1)
        b = -2 * a
        c = 1 - a - b
        k = 1 - random.random() * (c + a * t ** 2 + b * t)

        if np.random.randn() < 1.5:
            dandelions_1 = np.zeros_like(dandelions)
            for i in range(Npop):
                lamb = np.abs(np.random.randn(1, nD))
                theta = (2 * np.random.rand() - 1) * np.pi
                row = 1 / np.exp(theta)
                vx = row * np.cos(theta)
                vy = row * np.sin(theta)
                NEW = np.random.rand(1, nD) * (ub - lb) + lb
                dandelions_1[i, :] = dandelions[i, :] + alpha * vx * vy * np.log(np.random.normal(0, 1)) * (
                            NEW[0, :] - dandelions[i, :])
        else:
            dandelions_1 = dandelions * k

        dandelions = np.clip(dandelions_1, lb, ub)

        # Decline stage
        dandelions_mean = np.mean(dandelions, axis=0)
        dandelions_2 = np.zeros_like(dandelions)
        for i in range(Npop):
            for j in range(nD):
                dandelions_2[i, j] = dandelions[i, j] - beta[i, j] * alpha * (
                            dandelions_mean[j] - beta[i, j] * alpha * dandelions[i, j])

        dandelions = np.clip(dandelions_2, lb, ub)

        # Landing stage
        Step_length = levy(Npop, nD, 1.5)
        Elite = np.tile(Best_position, (Npop, 1))
        dandelions_3 = np.zeros_like(dandelions)
        for i in range(Npop):
            for j in range(nD):
                dandelions_3[i, j] = Elite[i, j] + Step_length[i, j] * alpha * (
                            Elite[i, j] - dandelions[i, j] * (2 * t / Max_it))

        dandelions = np.clip(dandelions_3, lb, ub)

        # Calculate fitness values
        for i in range(Npop):
            dandelionsFitness[i] = fobj(dandelions[i, :])

        # Sort dandelions based on fitness
        sorted_indexes = np.argsort(dandelionsFitness)
        dandelions = dandelions[sorted_indexes[:Npop], :]
        SortfitbestN = dandelionsFitness[sorted_indexes[:Npop]]

        # Update the best position and fitness
        if SortfitbestN[0] < Best_fitness:
            Best_position = dandelions[0, :]
            Best_fitness = SortfitbestN[0]

        Convergence_curve[t] = Best_fitness
        t += 1

    return Best_fitness, Best_position, Convergence_curve

