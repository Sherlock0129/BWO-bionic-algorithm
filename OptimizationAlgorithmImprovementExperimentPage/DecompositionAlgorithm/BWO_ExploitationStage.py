import numpy as np
from scipy.special import gamma

def exploitation_phase(Npop, Max_it, lb, ub, nD, fobj):
    pos = np.random.uniform(lb, ub, (Npop, nD))
    fit = np.array([fobj(ind) for ind in pos])
    xposbest = pos[np.argmin(fit)]
    fvalbest = np.min(fit)
    Curve = []

    alpha = 3 / 2
    sigma = (gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
             (gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)

    for T in range(Max_it):
        newpos = pos.copy()
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

        current_best_idx = np.argmin(fit)
        if fit[current_best_idx] < fvalbest:
            xposbest = newpos[current_best_idx]
            fvalbest = fit[current_best_idx]

        pos = newpos
        Curve.append(fvalbest)

    return xposbest, fvalbest, Curve