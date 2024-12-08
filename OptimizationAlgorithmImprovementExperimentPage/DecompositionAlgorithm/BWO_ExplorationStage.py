import numpy as np

def exploration_phase(Npop, Max_it, lb, ub, nD, fobj):
    pos = np.random.uniform(lb, ub, (Npop, nD))
    fit = np.array([fobj(ind) for ind in pos])
    xposbest = pos[np.argmin(fit)]
    fvalbest = np.min(fit)
    Curve = []

    for T in range(Max_it):
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

        current_best_idx = np.argmin(fit)
        if fit[current_best_idx] < fvalbest:
            xposbest = newpos[current_best_idx]
            fvalbest = fit[current_best_idx]

        pos = newpos
        Curve.append(fvalbest)

    return xposbest, fvalbest, Curve