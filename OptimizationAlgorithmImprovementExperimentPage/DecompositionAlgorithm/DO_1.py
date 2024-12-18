import numpy as np


# 随机初始化种群
def initialize_population(Npop, nD, lb, ub):
    return np.random.uniform(lb, ub, (Npop, nD))


# 评估适应度
def evaluate_population(population, fobj):
    return np.array([fobj(ind) for ind in population])


# 全局搜索阶段
def global_search(population, lb, ub, scale_factor=0.5):
    Npop, nD = population.shape
    new_population = population + scale_factor * np.random.uniform(-1, 1, (Npop, nD)) * (ub - lb)
    return np.clip(new_population, lb, ub)


# 局部搜索阶段
def local_search(population, g_best, lb, ub, scale_factor=0.1):
    new_population = population + scale_factor * np.random.uniform(-1, 1, population.shape) * (g_best - population)
    return np.clip(new_population, lb, ub)


# 第一阶段：初始化种群和进行全局搜索
def phase_one(Npop, Max_it, lb, ub, nD, fobj):
    # 初始化种群
    population = initialize_population(Npop, nD, lb, ub)
    fitness = evaluate_population(population, fobj)

    # 找到初始全局最优
    g_best_idx = np.argmin(fitness)
    g_best = population[g_best_idx]
    g_best_score = fitness[g_best_idx]

    curve = []  # 记录收敛曲线

    # 开始迭代
    for it in range(Max_it):
        # 全局搜索
        population = global_search(population, lb, ub)
        fitness = evaluate_population(population, fobj)

        # 更新全局最优解
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < g_best_score:
            g_best = population[best_idx]
            g_best_score = fitness[best_idx]

        # 记录当前最优
        curve.append(g_best_score)
        print(f"Phase 1 - Iteration {it + 1}: Best Fitness = {g_best_score}")

    return g_best, g_best_score, curve


