import random
import math
import numpy as np

def cpo(Npop, Max_it, lb, ub, nD, fobj):
    def initialization(Npop, nD, ub, lb):
        Boundary_no = len(ub)
        if Boundary_no == 1:
            return np.random.rand(Npop, nD) * (ub - lb) + lb
        else:
            Positions = np.zeros((Npop, nD))
            for i in range(nD):
                ub_i = ub[i]
                lb_i = lb[i]
                Positions[:, i] = np.random.rand(Npop) * (ub_i - lb_i) + lb_i
            return Positions

    Positions = initialization(Npop, nD, ub, lb)
    fitness = np.zeros(Npop)
    Gb_Fit = np.inf
    Gb_Sol = np.zeros(nD)
    Conv_curve = np.zeros(Max_it)
    Xp = Positions.copy()
    opt = fobj * 100

    for i in range(Npop):
        fitness[i] = fobj(Positions[i, :])
    Gb_Fit, index = fitness.min(0), fitness.argmin(0)
    Gb_Sol = Positions[index, :]

    t = 0
    while t <= Max_it and not np.array_equal(Gb_Fit, opt):
        r2 = random.random()
        for i in range(Npop):
            U1 = np.random.rand(nD) > random.random()
            if random.random() < random.random():  # Exploration phase
                if random.random() < random.random():  # First defense mechanism
                    y = (Positions[i, :] + Positions[random.randint(0, Npop - 1), :]) / 2
                    Positions[i, :] = Positions[i, :] + np.random.randn() * np.abs(2 * random.random() * Gb_Sol - y)
                else:  # Second defense mechanism
                    y = (Positions[i, :] + Positions[random.randint(0, Npop - 1), :]) / 2
                    Positions[i, :] = U1 * Positions[i, :] + (1 - U1) * (y + random.random() * (Positions[random.randint(0, Npop - 1), :] - Positions[random.randint(0, Npop - 1), :]))
            else:
                Yt = 2 * random.random() * (1 - t / Max_it) ** (t / Max_it)
                U2 = np.random.rand(nD) < 0.5 * 2 - 1
                S = random.random() * U2
                if random.random() < 0.8:  # Third defense mechanism
                    St = math.exp(fitness[i] / (sum(fitness) + np.finfo(float).eps))
                    S = S * Yt * St
                    Positions[i, :] = (1 - U1) * Positions[i, :] + U1 * (Positions[random.randint(0, Npop - 1), :] + St * (Positions[random.randint(0, Npop - 1), :] - Positions[random.randint(0, Npop - 1), :]) - S)
                else:  # Fourth defense mechanism
                    Mt = math.exp(fitness[i] / (sum(fitness) + np.finfo(float).eps))
                    vt = Positions[i, :]
                    Vtp = Positions[random.randint(0, Npop - 1), :]
                    Ft = np.random.rand(nD) * (Mt * (-vt + Vtp))
                    S = S * Yt * Ft
                    Positions[i, :] = (Gb_Sol + (0.2 * (1 - r2) + r2) * (U2 * Gb_Sol - Positions[i, :])) - S

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

        t += 1
        if t > Max_it:
            break
        Conv_curve[t - 1] = Gb_Fit

        # Update population size
        Npop = int(120 + (Npop - 120) * (1 - (t % (Max_it / 2)) / (Max_it / 2)))

    return Gb_Fit, Gb_Sol, Conv_curve