import numpy as np
from scipy.special import gamma


def exploitation_phase(pos, fit, Npop, nD, lb, ub, fobj, xposbest, T, Max_it):
    newpos = pos.copy()
    alpha = 3 / 2
    sigma = (gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
             (gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)

    for i in range(Npop):
        r3 = np.random.rand()
        r4 = np.random.rand()
        C1 = 2 * r4 * (1 - T / Max_it)
        RJ = np.random.randint(0, Npop)
        while RJ == i:
            RJ = np.random.randint(0, Npop)

        u = np.random.randn(nD) * sigma
        v = np.random.randn(nD)
        S = u / np.abs(v) ** (1 / alpha)
        KD = 0.05
        LevyFlight = KD * S

        newpos[i] = r3 * xposbest - r4 * pos[i] + C1 * LevyFlight * (pos[RJ] - pos[i])
        newpos[i] = np.clip(newpos[i], lb, ub)
        fit[i] = fobj(newpos[i])

    return newpos, fit