import numpy as np

def hho(Npop, Max_it, lb, ub, nD, fobj):
    # 鹰群优化算法
    # 初始化种群
    pos = np.random.rand(Npop, nD) * (ub - lb) + lb
    fit = np.array([fobj(p) for p in pos])
    best_idx = np.argmin(fit)
    best_pos = pos[best_idx]  # 全局最佳位置
    best_fval = fit[best_idx]  # 全局最佳适应度值
    Curve = np.zeros(Max_it)  # 记录每次迭代的全局最佳适应度值

    for t in range(Max_it):
        E0 = 2 * np.random.rand() - 1  # 鹰群能量的初始值
        E = E0 * (1 - t / Max_it)  # 鹰的动态能量

        for i in range(Npop):
            if np.abs(E) >= 1:  # 鹰进行探索模式
                # 选择一个随机个体作为目标
                rand_idx = np.random.randint(Npop)
                random_pos = pos[rand_idx]

                # 生成新位置
                new_pos = random_pos - np.random.rand() * np.abs(random_pos - 2 * np.random.rand() * pos[i])
            else:  # 鹰进行开发模式
                q = np.random.rand()
                if q >= 0.5:  # “突袭”模式
                    new_pos = best_pos - E * np.abs(best_pos - pos[i])
                else:  # “围捕”模式
                    jump_strength = 2 * (1 - np.random.rand())  # 跳跃强度
                    new_pos = best_pos - E * np.abs(jump_strength * best_pos - pos[i])

            # 边界处理
            new_pos = np.clip(new_pos, lb, ub)
            new_fval = fobj(new_pos)

            # 更新个体位置
            if new_fval < fit[i]:
                pos[i] = new_pos
                fit[i] = new_fval

                # 更新全局最佳
                if new_fval < best_fval:
                    best_pos = new_pos
                    best_fval = new_fval

        # 记录当前代的最佳适应度值
        Curve[t] = best_fval

    return best_pos, best_fval, Curve
