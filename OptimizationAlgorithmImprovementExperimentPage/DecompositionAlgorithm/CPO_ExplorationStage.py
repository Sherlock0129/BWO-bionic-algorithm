import random
import numpy as np

def exploration_phase(Npop, Max_it, lb, ub, nD, fobj, Positions, fitness, Gb_Sol, Gb_Fit):
    Xp = Positions.copy()
    for i in range(Npop):
        U1 = np.random.rand(nD) > random.random()
        if random.random() < random.random():  # First defense mechanism
            y = (Positions[i, :] + Positions[random.randint(0, Npop - 1), :]) / 2
            Positions[i, :] = Positions[i, :] + np.random.randn() * np.abs(2 * random.random() * Gb_Sol - y)
        else:  # Second defense mechanism
            y = (Positions[i, :] + Positions[random.randint(0, Npop - 1), :]) / 2
            Positions[i, :] = U1 * Positions[i, :] + (1 - U1) * (y + random.random() * (Positions[random.randint(0, Npop - 1), :] - Positions[random.randint(0, Npop - 1), :]))

        # Boundary handling
        for j in range(nD):
            if Positions[i, j] > ub[j]:
                Positions[i, j] = lb[j] + random.random() * (ub[j] - lb[j])
            elif Positions[i, j] < lb[j]:
                Positions[i, j] = lb[j] + random.random() * (ub[j] - lb[j])

        # Fitness calculation and update
        nF = fobj(Positions[i, :])
        if fitness[i] < nF:
            Positions[i, :] = Xp[i, :]
        else:
            Xp[i, :] = Positions[i, :]
            fitness[i] = nF
            if fitness[i] <= Gb_Fit:
                Gb_Sol = Positions[i, :]
                Gb_Fit = fitness[i]

    return Positions, fitness, Gb_Sol, Gb_Fit