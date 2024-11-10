import numpy as np


def de(Npop, Max_it, lb, ub, nD, fobj):
    # 差分进化算法
    # 初始化
    pop = np.random.rand(Npop, nD) * (ub - lb) + lb  # 初始化种群
    fit = np.array([fobj(ind) for ind in pop])  # 计算初始种群的适应度
    best_idx = np.argmin(fit)  # 最佳个体的索引
    best_ind = pop[best_idx]  # 最佳个体的位置
    best_fval = fit[best_idx]  # 最佳个体的适应度值
    Curve = np.zeros(Max_it)  # 记录每次迭代的最佳适应度值

    # DE参数
    F = 0.5  # 差分权重
    CR = 0.9  # 交叉概率

    for t in range(Max_it):
        for i in range(Npop):
            # 选择三个不同的个体
            indices = list(range(Npop))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]

            # 生成变异向量
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, lb, ub)  # 边界处理

            # 生成试验向量
            trial = np.copy(pop[i])
            crossover = np.random.rand(nD) < CR
            trial[crossover] = mutant[crossover]

            # 计算试验向量的适应度
            trial_fval = fobj(trial)

            # 如果试验向量比当前个体好，则替换当前个体
            if trial_fval < fit[i]:
                pop[i] = trial
                fit[i] = trial_fval

                # 更新全局最佳
                if trial_fval < best_fval:
                    best_ind = trial
                    best_fval = trial_fval

        # 记录当前代的最佳适应度值
        Curve[t] = best_fval

    return best_ind, best_fval, Curve
