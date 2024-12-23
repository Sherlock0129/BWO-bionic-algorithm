import numpy as np
from scipy.special import gamma

# 适应度函数（可以根据问题调整）
def fitness(position, fobj):
    return fobj(position)

def mgo_phase_1(Npop, Max_it, lb, ub, nD, fobj):
    population = np.random.uniform(lb, ub, (Npop, nD))
    fitness = np.array([fobj(ind) for ind in population])
    best_idx = np.argmin(fitness)
    xposbest = population[best_idx]
    fvalbest = fitness[best_idx]
    Curve = [fvalbest]

    for _ in range(Max_it):
        # 移动策略（简单示例）
        for i in range(Npop):
            population[i] += np.random.uniform(-1, 1, nD) * (xposbest - population[i])
            population[i] = np.clip(population[i], lb, ub)
            fitness[i] = fobj(population[i])

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < fvalbest:
            xposbest = population[best_idx]
            fvalbest = fitness[best_idx]

        Curve.append(fvalbest)

    return xposbest


# BWO 开发阶段（Exploration Phase）
def bwo(Npop, Max_it, lb, ub, nD, fobj):
    # 二维优化问题的鱼群算法
    # 初始化
    fit = np.inf * np.ones(Npop)
    newfit = fit.copy()
    Curve = np.inf * np.ones(Max_it)
    Counts_run = 0

    # 如果上界和下界是标量，扩展为向量
    if np.ndim(ub) == 0:
        lb = lb * np.ones(nD)
        ub = ub * np.ones(nD)

    # 初始化位置
    pos = mgo_phase_1(Npop, Max_it, lb, ub, nD, fobj)

    # 计算初始适应度
    for i in range(Npop):
        fit[i] = fobj(pos[i])
        Counts_run += 1

    fvalbest = np.min(fit)
    index = np.argmin(fit)
    xposbest = pos[index]

    T = 1
    while T <= Max_it:
        newpos = pos.copy()
        WF = 0.1 - 0.05 * (T / Max_it)  # 鱼群掉落的概率
        kk = (1 - 0.5 * T / Max_it) * np.random.rand(Npop)  # 探索或开发的概率

        for i in range(Npop):
            if kk[i] > 0.5:  # 探索阶段
                r1 = np.random.rand()
                r2 = np.random.rand()
                RJ = np.random.randint(0, Npop)
                while RJ == i:
                    RJ = np.random.randint(0, Npop)

                if nD <= Npop / 5:
                    params = np.random.permutation(nD)[:2]
                    # 检查是否有足够的参数
                    if len(params) > 1:
                        newpos[i, params[0]] = pos[i, params[0]] + (pos[RJ, params[0]] - pos[i, params[1]]) * (
                                    r1 + 1) * np.sin(r2 * 2 * np.pi)
                        newpos[i, params[1]] = pos[i, params[1]] + (pos[RJ, params[0]] - pos[i, params[1]]) * (
                                    r1 + 1) * np.cos(r2 * 2 * np.pi)
                else:
                    params = np.random.permutation(nD)
                    for j in range(nD // 2):
                        newpos[i, 2 * j] = pos[i, params[2 * j]] + (pos[RJ, params[0]] - pos[i, params[2 * j]]) * (
                                    r1 + 1) * np.sin(r2 * 2 * np.pi)
                        newpos[i, 2 * j + 1] = pos[i, params[2 * j + 1]] + (
                                    pos[RJ, params[0]] - pos[i, params[2 * j + 1]]) * (r1 + 1) * np.cos(r2 * 2 * np.pi)

            else:  # 开发阶段
                r3 = np.random.rand()
                r4 = np.random.rand()
                C1 = 2 * r4 * (1 - T / Max_it)
                RJ = np.random.randint(0, Npop)
                while RJ == i:
                    RJ = np.random.randint(0, Npop)

                alpha = 3 / 2
                sigma = (gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
                         (gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
                u = np.random.randn(nD) * sigma
                v = np.random.randn(nD)
                S = u / np.abs(v) ** (1 / alpha)
                KD = 0.05
                LevyFlight = KD * S

                newpos[i] = r3 * xposbest - r4 * pos[i] + C1 * LevyFlight * (pos[RJ] - pos[i])

            # 边界处理
            newpos[i] = np.clip(newpos[i], lb, ub)
            newfit[i] = fobj(newpos[i])  # 适应度计算
            Counts_run += 1

            if newfit[i] < fit[i]:
                pos[i] = newpos[i]
                fit[i] = newfit[i]

        # 鱼群掉落
        for i in range(Npop):
            if kk[i] <= WF:
                RJ = np.random.randint(0, Npop)
                r5 = np.random.rand()
                r6 = np.random.rand()
                r7 = np.random.rand()
                C2 = 2 * Npop * WF
                stepsize2 = r7 * (ub - lb) * np.exp(-C2 * T / Max_it)
                newpos[i] = (r5 * pos[i] - r6 * pos[RJ]) + stepsize2

                # 边界处理
                newpos[i] = np.clip(newpos[i], lb, ub)
                newfit[i] = fobj(newpos[i])  # 适应度计算
                Counts_run += 1

                if newfit[i] < fit[i]:
                    pos[i] = newpos[i]
                    fit[i] = newfit[i]

        fval = np.min(fit)
        if fval < fvalbest:
            fvalbest = fval
            xposbest = pos[np.argmin(fit)]

        Curve[T - 1] = fvalbest
        T += 1

    # display(['The function call is ', num2str(Counts_run)]);
    return xposbest, fvalbest, Curve


def mgo_phase_3(Npop, Max_it, lb, ub, nD, fobj, g_best):
    population = np.random.uniform(lb, ub, (Npop, nD))
    fitness = np.array([fobj(ind) for ind in population])
    xposbest = g_best
    fvalbest = fobj(g_best)
    Curve = [fvalbest]

    for _ in range(Max_it):
        for i in range(Npop):
            step_size = 0.1 * np.random.randn(nD)  # 更小的探索步长
            population[i] += step_size * (xposbest - population[i])
            population[i] = np.clip(population[i], lb, ub)
            fitness[i] = fobj(population[i])

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < fvalbest:
            xposbest = population[best_idx]
            fvalbest = fitness[best_idx]

        Curve.append(fvalbest)

    return xposbest, fvalbest, Curve

def optimize(Npop, Max_it, lb, ub, nD, fobj):
    xposbest, fvalbest, Curve = bwo(Npop, Max_it, lb, ub, nD, fobj)
    xposbest, fvalbest, Curve = mgo_phase_3(Npop, Max_it, lb, ub, nD, fobj, fvalbest)
    return xposbest, fvalbest, Curve