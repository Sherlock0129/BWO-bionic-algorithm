import numpy as np

def csa(Npop, Max_it, lb, ub, nD, fobj):
    # 乌鸦搜索算法
    # 初始化
    pos = np.random.rand(Npop, nD) * (ub - lb) + lb  # 初始化种群位置
    memory = pos.copy()  # 每只乌鸦的记忆位置（初始为自身位置）
    fit = np.array([fobj(p) for p in pos])  # 计算初始适应度
    best_idx = np.argmin(fit)
    best_pos = pos[best_idx]  # 当前全局最佳位置
    best_fval = fit[best_idx]  # 当前全局最佳适应度值
    Curve = np.zeros(Max_it)  # 记录每次迭代的全局最佳适应度值

    # CSA 参数
    AP = 0.1  # 觉察概率 (Awareness Probability)
    FL = 2.0  # 飞行长度 (Flight Length)

    for t in range(Max_it):
        for i in range(Npop):
            # 随机选择一只乌鸦的记忆位置
            j = np.random.randint(Npop)
            if np.random.rand() > AP:
                # 乌鸦前往记忆位置
                new_pos = pos[i] + FL * np.random.rand(nD) * (memory[j] - pos[i])
            else:
                # 乌鸦随机探索
                new_pos = np.random.rand(nD) * (ub - lb) + lb

            # 边界处理
            new_pos = np.clip(new_pos, lb, ub)
            new_fval = fobj(new_pos)

            # 更新位置和记忆
            if new_fval < fit[i]:  # 如果新位置的适应度更优
                pos[i] = new_pos
                fit[i] = new_fval
                memory[i] = new_pos  # 更新记忆位置

                # 更新全局最佳
                if new_fval < best_fval:
                    best_pos = new_pos
                    best_fval = new_fval

        # 记录当前迭代的全局最佳适应度值
        Curve[t] = best_fval

    return best_pos, best_fval, Curve
