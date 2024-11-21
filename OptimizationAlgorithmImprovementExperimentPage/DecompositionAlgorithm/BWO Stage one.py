# Exploration Phase
import numpy as np

def exploration_phase(pos, fit, Npop, nD, lb, ub, fobj, xposbest, T, Max_it):
    newpos = pos.copy()
    for i in range(Npop):
        r1 = np.random.rand()
        r2 = np.random.rand()
        RJ = np.random.randint(0, Npop)
        while RJ == i:
            RJ = np.random.randint(0, Npop)

        if nD <= Npop / 5:
            params = np.random.permutation(nD)[:2]
            if len(params) > 1:
                newpos[i, params[0]] = pos[i, params[0]] + (pos[RJ, params[0]] - pos[i, params[1]]) * (r1 + 1) * np.sin(r2 * 2 * np.pi)
                newpos[i, params[1]] = pos[i, params[1]] + (pos[RJ, params[0]] - pos[i, params[1]]) * (r1 + 1) * np.cos(r2 * 2 * np.pi)
        else:
            params = np.random.permutation(nD)
            for j in range(nD // 2):
                newpos[i, 2 * j] = pos[i, params[2 * j]] + (pos[RJ, params[0]] - pos[i, params[2 * j]]) * (r1 + 1) * np.sin(r2 * 2 * np.pi)
                newpos[i, 2 * j + 1] = pos[i, params[2 * j + 1]] + (pos[RJ, params[0]] - pos[i, params[2 * j + 1]]) * (r1 + 1) * np.cos(r2 * 2 * np.pi)

        newpos[i] = np.clip(newpos[i], lb, ub)
        fit[i] = fobj(newpos[i])

    return newpos, fit
