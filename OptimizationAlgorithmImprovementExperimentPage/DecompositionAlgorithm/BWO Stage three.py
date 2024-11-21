import numpy as np

def whale_fall_phase(pos, fit, Npop, nD, lb, ub, fobj, T, Max_it, WF):
    newpos = pos.copy()
    for i in range(Npop):
        if np.random.rand() <= WF:
            RJ = np.random.randint(0, Npop)
            r5 = np.random.rand()
            r6 = np.random.rand()
            r7 = np.random.rand()
            C2 = 2 * Npop * WF
            stepsize2 = r7 * (ub - lb) * np.exp(-C2 * T / Max_it)
            newpos[i] = (r5 * pos[i] - r6 * pos[RJ]) + stepsize2

            newpos[i] = np.clip(newpos[i], lb, ub)
            fit[i] = fobj(newpos[i])

    return newpos, fit
