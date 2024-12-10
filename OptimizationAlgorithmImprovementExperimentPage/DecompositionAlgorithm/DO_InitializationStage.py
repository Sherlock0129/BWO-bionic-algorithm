import numpy as np

def initialization_phase(Npop, Max_it, lb, ub, nD, fobj):
    Boundary_no = len(ub)
    if Boundary_no == 1:
        Positions = np.random.rand(Npop, nD) * (ub - lb) + lb
    else:
        Positions = np.zeros((Npop, nD))
        for i in range(nD):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(Npop) * (ub_i - lb_i) + lb_i
    fitness = np.array([fobj(Positions[i, :]) for i in range(Npop)])
    Best_fitness = fitness.min()
    Best_position = Positions[fitness.argmin()]
    return Positions, fitness, Best_position, Best_fitness