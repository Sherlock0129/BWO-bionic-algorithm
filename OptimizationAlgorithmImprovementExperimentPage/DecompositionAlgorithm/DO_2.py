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

# 第二阶段：基于上一阶段的结果进行局部搜索
def phase_two(Npop, Max_it, lb, ub, nD, fobj, g_best):
    population = initialize_population(Npop, nD, lb, ub)
    curve = []

    for it in range(Max_it):
        # 局部搜索
        population = local_search(population, g_best, lb, ub)
        fitness = evaluate_population(population, fobj)

        # 更新全局最优解
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < fobj(g_best):
            g_best = population[best_idx]

        # 记录当前最优
        curve.append(fobj(g_best))
        print(f"Phase 2 - Iteration {it + 1}: Best Fitness = {fobj(g_best)}")

    return g_best, fobj(g_best), curve





