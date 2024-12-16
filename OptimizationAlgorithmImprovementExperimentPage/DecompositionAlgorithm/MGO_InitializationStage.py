import numpy as np

def initialization_phase(Npop, Max_it, lb, ub, nD, fobj):
    if len(ub) == 1:
        Positions = np.random.rand(Npop, nD) * (ub - lb) + lb
    else:
        Positions = np.zeros((Npop, nD))
        for i in range(nD):
            high = ub[i]
            low = lb[i]
            Positions[:, i] = np.random.rand(Npop) * (high - low) + low
    fitness = np.array([fobj(Positions[i, :]) for i in range(Npop)])
    Best_fitness = fitness.min()
    Best_position = Positions[fitness.argmin()]
    return Positions, fitness, Best_position, Best_fitness

