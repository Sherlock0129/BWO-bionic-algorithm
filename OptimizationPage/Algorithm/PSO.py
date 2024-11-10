import numpy as np

def pso(Npop, Max_it, lb, ub, nD, fobj):
    # 粒子群优化算法
    # 初始化
    pos = np.random.rand(Npop, nD) * (ub - lb) + lb  # 初始化粒子位置
    vel = np.random.rand(Npop, nD) * 0.1 * (ub - lb)  # 初始化粒子速度
    fit = np.array([fobj(p) for p in pos])  # 计算适应度值
    pbest_pos = pos.copy()  # 记录每个粒子的最佳位置
    pbest_val = fit.copy()  # 记录每个粒子的最佳适应度值
    gbest_pos = pos[np.argmin(fit)]  # 全局最佳位置
    gbest_val = np.min(fit)  # 全局最佳适应度值
    Curve = np.zeros(Max_it)  # 记录每次迭代的全局最佳值

    # PSO 参数
    w = 0.5  # 惯性权重
    c1 = 1.5  # 认知系数
    c2 = 1.5  # 社会系数

    for t in range(Max_it):
        # 更新每个粒子的速度和位置
        for i in range(Npop):
            r1 = np.random.rand(nD)
            r2 = np.random.rand(nD)
            vel[i] = (w * vel[i] +
                      c1 * r1 * (pbest_pos[i] - pos[i]) +
                      c2 * r2 * (gbest_pos - pos[i]))
            pos[i] = pos[i] + vel[i]

            # 边界处理
            pos[i] = np.clip(pos[i], lb, ub)

            # 计算新位置的适应度值
            new_fit = fobj(pos[i])

            # 更新每个粒子的最佳位置和适应度值
            if new_fit < pbest_val[i]:
                pbest_pos[i] = pos[i]
                pbest_val[i] = new_fit

            # 更新全局最佳位置和适应度值
            if new_fit < gbest_val:
                gbest_pos = pos[i]
                gbest_val = new_fit

        # 记录当前迭代的最佳适应度值
        Curve[t] = gbest_val

    return gbest_pos, gbest_val, Curve
