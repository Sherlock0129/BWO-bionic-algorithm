import numpy as np

def aoa(Npop, Max_it, lb, ub, nD, fobj):
    # 算术优化算法
    # 初始化
    pos = np.random.rand(Npop, nD) * (ub - lb) + lb  # 初始化种群位置
    fit = np.array([fobj(p) for p in pos])  # 计算适应度
    best_idx = np.argmin(fit)
    best_pos = pos[best_idx]  # 当前全局最佳位置
    best_fval = fit[best_idx]  # 当前全局最佳适应度值
    Curve = np.zeros(Max_it)  # 记录每次迭代的全局最佳适应度值

    # AOA 参数
    min_acc = 0.1  # 最小加速率
    max_acc = 1.0  # 最大加速率
    alpha = 5      # 控制探索和开发平衡的参数
    mu = 0.5       # 控制变异的系数

    for t in range(Max_it):
        # 动态调整加速率和探索-开发比率
        acc = min_acc + (max_acc - min_acc) * (t / Max_it)
        p = 1 - (t / Max_it) ** (alpha / 2)

        for i in range(Npop):
            if np.random.rand() < p:  # 探索阶段
                r1, r2 = np.random.rand(), np.random.rand()
                new_pos = pos[i] + r1 * acc * ((ub - lb) * r2 + lb - pos[i])
            else:  # 开发阶段
                r3, r4 = np.random.rand(), np.random.rand()
                new_pos = best_pos + mu * acc * (r3 * (ub - lb) + lb - r4 * pos[i])

            # 边界处理
            new_pos = np.clip(new_pos, lb, ub)
            new_fval = fobj(new_pos)

            # 更新个体位置和适应度
            if new_fval < fit[i]:
                pos[i] = new_pos
                fit[i] = new_fval

                # 更新全局最佳
                if new_fval < best_fval:
                    best_pos = new_pos
                    best_fval = new_fval

        # 记录当前迭代的全局最佳适应度值
        Curve[t] = best_fval

    return best_pos, best_fval, Curve
