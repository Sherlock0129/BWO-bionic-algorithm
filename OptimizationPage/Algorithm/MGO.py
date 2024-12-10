import numpy as np
import random


def initialization(N, dim, up, down):
    """
    Initialize the population of gazelles with random values
    """
    if len(up) == 1:
        X = np.random.rand(N, dim) * (up - down) + down
    else:
        X = np.zeros((N, dim))
        for i in range(dim):
            high = up[i]
            low = down[i]
            X[:, i] = np.random.rand(N) * (high - low) + low
    return X


def boundary_check(X, lb, ub):
    """
    Ensure that the solution is within the specified boundaries
    """
    for i in range(X.shape[0]):
        FU = X[i, :] > ub
        FL = X[i, :] < lb
        X[i, :] = (X[i, :] * ~(FU + FL)) + ub * FU + lb * FL
    return X


def coefficient_vector(dim, Iter, MaxIter):
    """
    Calculate the coefficient vector used in the MGO algorithm
    """
    a2 = -1 + Iter * (-1 / MaxIter)
    u = np.random.randn(1, dim)
    v = np.random.randn(1, dim)

    cofi = np.zeros((4, dim))
    cofi[0, :] = np.random.rand(1, dim)
    cofi[1, :] = (a2 + 1) + np.random.rand(1, dim)
    cofi[2, :] = a2 * np.random.randn(1, dim)
    cofi[3, :] = u * v ** 2 * np.cos((np.random.rand(1, dim) * 2) * u)

    return cofi


def solution_imp(X, BestX, lb, ub, N, cofi, M, A, D, i):
    """
    Update the solution based on the MGO algorithm
    """
    NewX = np.zeros((4, X.shape[1]))
    NewX[0, :] = (ub - lb) * np.random.rand(1, X.shape[1]) + lb
    NewX[1, :] = BestX - np.abs((np.random.randint(2) * M - np.random.randint(2) * X[i, :]) * A) * cofi[0, :]
    NewX[2, :] = (M + cofi[0, :]) + (
                np.random.randint(2) * BestX - np.random.randint(2) * X[np.random.randint(N), :]) * cofi[0, :]
    NewX[3, :] = (X[i, :] - D) + (np.random.randint(2) * BestX - np.random.randint(2) * M) * cofi[0, :]

    return NewX


def mgo(N, MaxIter, LB, UB, dim, fobj):
    """
    Main MGO algorithm
    """
    lb = np.ones(dim) * LB
    ub = np.ones(dim) * UB

    # Initialize the first random population of Gazelles
    X = initialization(N, dim, UB, LB)

    # Initialize Best Gazelle
    BestX = None
    BestFitness = float('inf')

    # Calculate initial fitness of the population
    Sol_Cost = np.zeros(N)
    for i in range(N):
        Sol_Cost[i] = fobj(X[i, :])
        if Sol_Cost[i] <= BestFitness:
            BestFitness = Sol_Cost[i]
            BestX = X[i, :]

    cnvg = np.zeros(MaxIter)

    # Main loop
    for Iter in range(MaxIter):
        for i in range(N):
            RandomSolution = np.random.choice(N, size=int(np.ceil(N / 3)), replace=False)
            M = X[np.random.randint(np.ceil(N / 3), N), :] * np.floor(np.random.rand()) + np.mean(X[RandomSolution, :],
                                                                                                  axis=0) * np.ceil(
                np.random.rand())

            # Calculate the vector of coefficients
            cofi = coefficient_vector(dim, Iter, MaxIter)

            A = np.random.randn(1, dim) * np.exp(2 - Iter * (2 / MaxIter))
            D = (np.abs(X[i, :]) + np.abs(BestX)) * (2 * np.random.rand(1, dim) - 1)

            # Update the location
            NewX = solution_imp(X, BestX, lb, ub, N, cofi, M, A, D, i)

            # Boundary check
            NewX = boundary_check(NewX, lb, ub)

            # Calculate fitness and add new solutions to the herd
            Sol_CostNew = np.array([fobj(NewX[j, :]) for j in range(NewX.shape[0])])

            # Adding new gazelles to the herd
            X = np.vstack([X, NewX])
            Sol_Cost = np.hstack([Sol_Cost, Sol_CostNew])

            # Update Best Gazelle
            BestX = X[np.argmin(Sol_Cost), :]

        # Update herd
        Sol_Cost, SortOrder = np.sort(Sol_Cost), np.argsort(Sol_Cost)
        X = X[SortOrder, :]
        BestFitness = Sol_Cost[0]
        BestX = X[0, :]

        # Update only the top N gazelles
        X = X[:N, :]
        Sol_Cost = Sol_Cost[:N]

        cnvg[Iter] = BestFitness

    return BestFitness, BestX, cnvg
