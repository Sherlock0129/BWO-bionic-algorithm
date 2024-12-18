import numpy as np

def whale_fall_phase(Npop, Max_it, lb, ub, nD, fobj, g_best):
    pos = g_best + np.random.uniform(-1, 1, (Npop, nD)) * (ub - lb) * 0.1
    fit = np.array([fobj(ind) for ind in pos])
    xposbest = pos[np.argmin(fit)]
    fvalbest = np.min(fit)
    Curve = []

    for T in range(Max_it):
        newpos = pos.copy()
        for i in range(Npop):
            RJ = np.random.randint(0, Npop)
            r5 = np.random.rand()
            r6 = np.random.rand()
            r7 = np.random.rand()
            C2 = 2 * Npop
            stepsize2 = r7 * (ub - lb) * np.exp(-C2 * T / Max_it)
            newpos[i] = (r5 * pos[i] - r6 * pos[RJ]) + stepsize2

            newpos[i] = np.clip(newpos[i], lb, ub)
            fit[i] = fobj(newpos[i])

        current_best_idx = np.argmin(fit)
        if fit[current_best_idx] < fvalbest:
            xposbest = newpos[current_best_idx]
            fvalbest = fit[current_best_idx]

        pos = newpos
        Curve.append(fvalbest)

    return xposbest, fvalbest, Curve